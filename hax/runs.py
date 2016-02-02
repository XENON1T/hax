"""
Runs database utilities
TEMPORARY: These will soon interface with the XENON1T runs database instead
"""
import pandas as pd
from glob import glob
import os

# Load the csv files for each run
def update_datasets():
    global DATASETS
    DATASETS = []
    for rundbfile in glob('runs_info/*.csv'):
        tpc, run = os.path.splitext(os.path.basename(rundbfile))[0].split('_')
        dsets = pd.read_csv(rundbfile)
        dsets = pd.concat((dsets, pd.DataFrame([{'tpc': tpc, 'run': run}] * len(dsets))),
                           axis=1)
        try:
            DATASETS
        except NameError:
            DATASETS = dsets
        else:
            DATASETS = pd.concat((DATASETS, dsets))

update_datasets()


def get_dataset_info(dataset_name):
    """Returns a dictionary with the runs database info for a given dataset"""
    return DATASETS[DATASETS['name'] == dataset_name].iloc[0].to_dict()


def datasets_query(query):
    return DATASETS.query(query)['name'].values

