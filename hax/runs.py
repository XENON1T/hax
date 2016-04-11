"""
Runs database utilities
TEMPORARY: These will soon interface with the XENON1T runs database instead
"""
import logging
from glob import glob
import os

from tqdm import tqdm
import pandas as pd
import pymongo
import numpy as np

import hax
from hax.utils import flatten_dict

log = logging.getLogger('hax.runs')

# This will hold the dataframe containing dataset info
# DO NOT import this directly (from hax.runs import datasets), you will just get None!
datasets = None


def update_datasets():
    """Update hax.runs.datasets to contain latest datasets.
    Currently just loads XENON100 run 10 runs from a csv file.
    """
    global datasets
    experiment = hax.config['experiment']
    if experiment == 'XENON100':
        # Fetch runs information from static csv files in runs info
        for rundbfile in glob(os.path.join(hax.config['runs_info_dir'], '*.csv')):
            tpc, run = os.path.splitext(os.path.basename(rundbfile))[0].split('_')
            dsets = pd.read_csv(rundbfile)
            dsets = pd.concat((dsets, pd.DataFrame([{'tpc': tpc, 'run': run}] * len(dsets))),
                               axis=1)
            if datasets is not None and len(datasets):
                datasets = dsets
            else:
                datasets = pd.concat((datasets, dsets))

    elif experiment == 'XENON1T':
        # Connect to the runs database
        if 'mongo_password' in hax.config:
            password = hax.config['mongo_password']
        elif 'MONGO_PASSWORD' in os.environ:
            password = os.environ['MONGO_PASSWORD']
        else:
            raise ValueError('Please set the MONGO_PASSWORD environment variable or the hax.mongo_password option '
                             'to access the runs database.')
        client = pymongo.MongoClient(hax.config['runs_url'].format(password=password))
        db = client[hax.config['runs_database']]
        collection = db[hax.config['runs_collection']]
        if 'detector' in hax.config:
            detector = hax.config['detector']
        else:
            detector = 'tpc'

        docs = []
        for doc in collection.find({'detector':detector},
                                   ['name', 'number', 'reader.self_trigger', 'source']):
            doc = flatten_dict(doc)
            del doc['_id']   # Remove the Mongo document ID
            doc['raw_data_subfolder'] = ''      # For the moment, everything is in one folder
            docs.append(doc)
        datasets = pd.DataFrame(docs)
        client.close()

    # What dataset names do we have?
    dataset_names = datasets['name'].values
    datasets['location'] = [''] * len(dataset_names)
    datasets['raw_data_found'] = [False] * len(dataset_names)

    # Walk through all the main_data_paths, looking for root files
    # This should be more efficient than looking for each dataset separately
    for data_dir in hax.config.get('main_data_paths', []):
        for candidate in glob(os.path.join(data_dir, '*.root')):

            # What dataset is this file for?
            dsetname = os.path.splitext(os.path.basename(candidate))[0]
            bla = np.where(dataset_names == dsetname)[0]

            if len(bla):
                # Dataset was found, index is in bla[0]
                datasets.loc[bla[0], 'location'] = candidate

    # For the raw data, we need to look in all subfolders
    # DO NOT do os.path.exist for each dataset, it will take minutes, at least over sshfs
    if hax.config['raw_data_access_mode'] == 'local':
        for subfolder, dsets_in_subfolder in datasets.groupby('raw_data_subfolder'):
            subfolder_path = os.path.join(hax.config['raw_data_local_path'], subfolder)
            if not os.path.exists(subfolder_path):
                log.debug("Folder %s not found when looking for raw data" % subfolder_path)
                continue
            for candidate in os.listdir(subfolder_path):
                bla = np.where(dataset_names == candidate)[0]
                if len(bla):
                    datasets.loc[bla[0], 'raw_data_found'] = True


def get_dataset_info(dataset_name):
    """Returns a dictionary with the runs database info for a given dataset
    """
    return datasets[datasets['name'] == dataset_name].iloc[0].to_dict()


def datasets_query(query):
    """Return names of datasets matching query"""
    return datasets.query(query)['name'].values
