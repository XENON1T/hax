"""
Runs database utilities
TEMPORARY: These will soon interface with the XENON1T runs database instead
"""
from hax.utils import HAX_DIR
from hax.config import CONFIG
import pandas as pd
from glob import glob
import os
import numpy as np
DATASETS = []


# Load the csv files for each run
def update_datasets():

    # Add which datasets should exist
    global DATASETS
    DATASETS = []
    for rundbfile in glob(HAX_DIR + '/runs_info/*.csv'):
        tpc, run = os.path.splitext(os.path.basename(rundbfile))[0].split('_')
        dsets = pd.read_csv(rundbfile)
        dsets = pd.concat((dsets, pd.DataFrame([{'tpc': tpc, 'run': run}] * len(dsets))),
                           axis=1)

        if not len(DATASETS):
            DATASETS = dsets
        else:
            DATASETS = pd.concat((DATASETS, dsets))

    # Add data location for each dataset

    # What dataset names do we have?
    dataset_names = DATASETS['name'].values
    DATASETS['location'] = [''] * len(dataset_names)

    # Walk through all the main_data_paths, looking for root files
    for data_dir in CONFIG.get('main_data_paths', []):
        for candidate in glob(os.path.join(data_dir, '*.root')):

            # What dataset is this file for?
            dsetname = os.path.splitext(os.path.basename(candidate))[0]
            bla = np.where(dataset_names == dsetname)[0]

            if len(bla):
                # Dataset was found, index is in bla[0]
                DATASETS.loc[bla[0], 'location'] = os.path.join(data_dir, candidate)


update_datasets()


def get_dataset_info(dataset_name):
    """Returns a dictionary with the runs database info for a given dataset"""
    return DATASETS[DATASETS['name'] == dataset_name].iloc[0].to_dict()


def datasets_query(query):
    return DATASETS.query(query)['name'].values

