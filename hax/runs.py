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

rundb_client = None

def get_rundb_collection():
    global rundb_client
    if rundb_client is None:
        # Connect to the runs database
        if 'mongo_password' in hax.config:
            password = hax.config['mongo_password']
        elif 'MONGO_PASSWORD' in os.environ:
            password = os.environ['MONGO_PASSWORD']
        else:
            raise ValueError('Please set the MONGO_PASSWORD environment variable or the hax.mongo_password option '
                             'to access the runs database.')
        rundb_client = pymongo.MongoClient(hax.config['runs_url'].format(password=password))
    db = rundb_client[hax.config['runs_database']]
    return db[hax.config['runs_collection']]


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
        collection = get_rundb_collection()
        docs = []
        for doc in collection.find({'detector': hax.config.get('detector', 'tpc')},
                                   ['name', 'number', 'start', 'end', 'source',
                                    'reader.self_trigger',
                                    'trigger.events_built', 'trigger.status',
                                    'tags.name'
                                    ]):
            doc['tags'] = ','.join([t['name'] for t in doc.get('tags', [])])   # Convert tags to single string
            doc = flatten_dict(doc, sep='__')
            del doc['_id']   # Remove the Mongo document ID
            doc['raw_data_subfolder'] = ''      # For the moment, everything is in one folder
            docs.append(doc)
        datasets = pd.DataFrame(docs)

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
    """Synonym for get_run_info"""
    return get_run_info(dataset_name)


def get_run_info(run_name):
    """Returns a dictionary with the runs database info for a given dataset
    For XENON1T, this queries the runs db to get the complete run doc.
    """
    global datasets
    if hax.config['experiment'] == 'XENON100':
        return datasets[datasets['name'] == run_name].iloc[0].to_dict()
    elif hax.config['experiment'] == 'XENON1T':
        collection = get_rundb_collection()
        result = list(collection.find({'name': run_name}))
        if len(result) == 0:
            raise ValueError("Run named %s not found in run db!" % run_name)
        if len(result) > 1:
            raise ValueError("More than one run named %s found in run db???" % run_name)
        return result[0]


def datasets_query(query):
    """Return names of datasets matching query"""
    return datasets.query(query)['name'].values
