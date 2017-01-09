"""
Runs database utilities
"""
from distutils.version import LooseVersion
import logging
from glob import glob
import os
import fnmatch
import re

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


def update_datasets(query=None):
    """Update hax.runs.datasets to contain latest datasets.
    Currently just loads XENON100 run 10 runs from a csv file.
    query: custom query, in case you only want to update partially??
    """
    global datasets
    experiment = hax.config['experiment']

    version_policy = hax.config['pax_version_policy']

    if not hax.config.get('use_runs_db', True):
        hax.log.info("Not looking for datasets in runs, db since you put use_runs_db = False")
        return

    if experiment == 'XENON100':
        # Fetch runs information from static csv files in runs info
        for rundbfile in glob(os.path.join(hax.config['runs_info_dir'], '*.csv')):
            tpc, run = os.path.splitext(os.path.basename(rundbfile))[0].split('_')
            dsets = pd.read_csv(rundbfile)
            dsets = pd.concat((dsets, pd.DataFrame([{'tpc': tpc, 'run': run}] * len(dsets))),
                              axis=1)
            if datasets is not None and len(datasets):
                datasets = pd.concat((datasets, dsets), ignore_index=True)
            else:
                datasets = dsets

    elif experiment == 'XENON1T':
        collection = get_rundb_collection()
        docs = []

        if query is None:
            query = {}
        query['detector'] = hax.config.get('detector', hax.config['detector'])

        log.debug("Updating datasets from runs database... ")
        cursor = collection.find(query,
                                ['name', 'number', 'start', 'end', 'source',
                                 'reader.self_trigger', 'reader.ini.name',
                                 'trigger.events_built', 'trigger.status',
                                 'tags.name',
                                 'data'])
        for doc in cursor:
            # Process and flatten the doc
            doc['tags'] = ','.join([t['name'] for t in doc.get('tags', [])])  # Convert tags to single string
            doc = flatten_dict(doc, separator='__')
            del doc['_id']  # Remove the Mongo document ID
            if 'data' in doc:
                data_docs = doc['data']
                del doc['data']
            else:
                data_docs = []
            doc = flatten_dict(doc, separator='__')

            if version_policy != 'loose':

                # Does the run db know where to find the processed data at this host?
                processed_data_docs = [d for d in data_docs
                                       if (d['type'] == 'processed'
                                           and hax.config['cax_key'] in d['host']
                                           and d['status'] == 'transferred')]

                if version_policy != 'latest':
                    # Filter out versions not consistent with the version policy.
                    # We will take the latest of the remaining ones later later.
                    processed_data_docs = [d for d in processed_data_docs
                                           if version_is_consistent_with_policy(d['pax_version'])]

                # If there is a processed data consistent with the version policy, set its location
                doc['location'] = ''
                doc['pax_version'] = ''
                if len(processed_data_docs):
                    # Take the data doc with the most recent policy-consistent pax version
                    data_we_take = max(processed_data_docs,
                                       key=lambda x: LooseVersion(x['pax_version']))
                    doc['location'] = data_we_take['location']
                    doc['pax_version'] = data_we_take['pax_version'][1:]

            docs.append(doc)

        datasets = pd.DataFrame(docs)
        log.debug("... done.")

    # These may or may not have been set already:
    if not 'pax_version' in datasets:
        datasets['pax_version'] = [''] * len(datasets)
    if not 'location' in datasets:
        datasets['location'] = [''] * len(datasets)
    if not 'raw_data_subfolder' in datasets:
        datasets['raw_data_subfolder'] = [''] * len(datasets)
    if not 'raw_data_found' in datasets:
        datasets['raw_data_found'] = [False] * len(datasets)
    dataset_names = datasets['name'].values

    if version_policy == 'loose':
        # Walk through main_data_paths, looking for root files
        # Reversed, since if we find a dataset again, we overwrite, and 
        # usually people put first priority stuff at the front.
        for data_dir in reversed(hax.config.get('main_data_paths', [])):
            for candidate in glob(os.path.join(data_dir, '*.root')):
                # What dataset is this file for?
                dsetname = os.path.splitext(os.path.basename(candidate))[0]
                bla = np.where(dataset_names == dsetname)[0]
                if len(bla):
                    # Dataset was found, index is in bla[0]
                    datasets.loc[bla[0], 'location'] = candidate

    # For the raw data, we may need to look in subfolders ('run_10' etc)
    # don't do os.path.exist for each dataset, it will take minutes, at least over sshfs
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


def version_tuple(v):
    """Convert a version indication string (e.g. "6.2.1") into a tuple of integers"""
    if v.startswith('v'):
        v = v[1:]
    return tuple(map(int, (v.split("."))))


def version_is_consistent_with_policy(version):
    """Returns if the pax version is consistent with the pax version policy.
    If policy is 6.2.1, only '6.2.1' (or 'v6.2.1') gives True
    If policy is 6.2, any of 6.2.0, 6.2.1 etc. gives True
    """
    vp = hax.config['pax_version_policy']
    if vp in ['loose', 'latest']:
        raise RuntimeError("This function should not be called if pax_version_policy is %s" % vp)
    vp = version_tuple(vp)
    version = version_tuple(version)
    for i, x in enumerate(vp):
        if version[i] != vp[i]:
            return False
    return True

def get_run_info(run_id, projection_query=None):
    """Returns a dictionary with the runs database info for a given run_id.
    For XENON1T, this queries the runs db to get the complete run doc.

    :param run_id: name or number, or list of such, of runs to query

    :param projection_query: can be
      - None (default): the entire run doc will be returned
      - string: runs db field name (with dots indicating subfields), we'll query and return only that field.
      - anything else: passed as projection to pymongo.collection.find

    For example 'processor.DEFAULT.electron_lifetime_liquid' returns the electron lifetime.
    """
    if isinstance(projection_query, str):
        single_field_mode = True
        pq = {projection_query: True}
    else:
        single_field_mode = False
        pq = projection_query

    multi_run_mode = isinstance(run_id, (list, tuple))
    if multi_run_mode:
        run_names = [get_run_name(x) for x in run_id]
    else:
        run_names = [get_run_name(run_id)]

    if hax.config['experiment'] == 'XENON100':
        if multi_run_mode or single_field_mode:
            raise NotImplementedError("For XENON100, only single-run, full run info queries are supported")
        return datasets[np.in1d(datasets['name'], run_names)].iloc[0].to_dict()

    elif hax.config['experiment'] == 'XENON1T':
        collection = get_rundb_collection()
        result = list(collection.find({'name': {'$in': run_names},
                                       'detector': hax.config['detector']},
                                      pq))
        if len(result) == 0:
            raise ValueError("No runs matching %s found in run db!" % str(run_names))
        if len(result) > 1:
            if not multi_run_mode:
                raise ValueError("More than one run named %s found in run db???" % run_names[0])

        if single_field_mode:
            # Extract the single field the user requested
            for subkey in projection_query.split('.'):

                for i in range(len(result)):
                    result[i] = result[i][subkey]

        if multi_run_mode:
            return result
        return result[0]

get_dataset_info = get_run_info    # Synonym


def datasets_query(query):
    """Return names of datasets matching query"""
    return datasets.query(query)['name'].values


def get_run_name(run_id):
    """Return run name matching run_id. Returns run_id if run_id is string (presumably already run name)"""
    if isinstance(run_id, str):
        return run_id
    try:
        return datasets_query('number == %d' % run_id)[0]
    except Exception as e:
        print("Could not find run name for %s, got exception %s: %s. Setting run name to 'unknown'" % (
            run_id, type(e), str(e)))
        return "unknown"


def get_run_number(run_id):
    """Return run number matching run_id. Returns run_id if run_id is int (presumably already run int)"""
    if isinstance(run_id, (int, float, np.int, np.int32, np.int64)):
        return int(run_id)
    try:
        if hax.config['experiment'] == 'XENON100':
            # Convert from XENON100 dataset name (like xe100_120402_2000) to number
            if run_id.startswith('xe100_'):
                run_id = run_id[6:]
            run_id = run_id.replace('_', '')
            run_id = run_id[:10]
            return int(run_id)

        return datasets.query('name == "%s"' % run_id)['number'].values[0]
    except Exception as e:
        print("Could not find run number for %s, got exception %s: %s. Setting run number to 0." % (
            run_id, type(e), str(e)))
        return 0


def tags_selection(dsets=None, include=None, exclude=None, pattern_type='fnmatch', ignore_underscore=True):
    """Return runs by tag selection criteria.

    :param dsets: pandas DataFrame, subset of datasets from hax.runs.datasets.
                  If not provided, uses hax.runs.datasets itself (all datasets).

    :param include: String or list of strings of patterns of tags to include

    :param exclude: String or list of strings of patterns of tags to exclude.
                    Exclusion criteria  have higher priority than inclusion criteria.

    :param pattern_type: Type of pattern matching to use. Defaults to 'fnmatch', which means you can use
                         unix shell-style wildcards (?, *). Alternative is 're', which means you can use
                         full python regular expressions.

    :param ignore_underscore: Ignore the underscore at the start of some tags (indicating some degree of
                              officialness or automation) when matching.

    Examples:
     - `tags_selection(include='blinded')` select all datasets with a blinded or _blinded tag.
     - `tags_selection(include='*blinded')` ... with blinded or _blinded, unblinded, blablinded, etc.
     - `tags_selection(include=['blinded', 'unblinded'])` ... with blinded OR unblinded, but not blablinded.
     - `tags_selection(include='blinded', exclude=['bad', 'messy'])` select blinded dsatasets
        that aren't bad or messy
    """
    if dsets is None:
        dsets = hax.runs.datasets

    if include is not None:
        dsets = dsets[_tags_match(dsets, include, pattern_type, ignore_underscore)]
    if exclude is not None:
        dsets = dsets[True ^ _tags_match(dsets, exclude, pattern_type, ignore_underscore)]

    return dsets


def _tags_match(dsets, patterns, pattern_type, ignore_underscore):
    result = np.zeros(len(dsets), dtype=np.bool)

    if isinstance(patterns, str):
        patterns = [patterns]

    for i, tags in enumerate(dsets.tags):
        result[i] = any([any([_tag_match(tag, pattern, pattern_type, ignore_underscore)
                              for tag in tags.split(',')
                              for pattern in patterns])])

    return result


def _tag_match(tag, pattern, pattern_type, ignore_underscore):
    if ignore_underscore and tag.startswith('_'):
        tag = tag[1:]
    if pattern_type == 'fnmatch':
        return fnmatch.fnmatch(tag, pattern)
    elif pattern_type == 're':
        return bool(re.match(pattern, tag))
    raise NotImplementedError

    
def is_blind(run_id):
    """Determine if a dataset should be blinded based on the runDB

    :param run_id: name or number of the run to load

    :returns : True if the blinding cut should be applied, False if not
    """
    if hax.config['experiment'] != 'XENON1T':
        return False

    try:
        tags = get_run_info(run_id, projection_query='tags')
    except KeyError:
        tags = []
    except ValueError:
        # Couldn't find in runDB so blind by default
        log.warning("Blinding by default since cannot find run.")
        return True

    tag_names = [tag['name'] for tag in tags]

    # underscore means that it is a protected tag
    if 'blinded' in tag_names or '_blinded' in tag_names:
        return True

    return False
  