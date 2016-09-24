"""Make small flat root trees with one entry per event from the pax root files.
"""
from datetime import datetime
from distutils.version import LooseVersion
from glob import glob
import inspect
import logging
import json
import pickle
import os
import warnings

import numpy as np
import pandas as pd

try:
    import ROOT
    import root_numpy
except ImportError as e:
    warnings.warn("Error importing ROOT-related libraries: %s. "
                  "If you try to use ROOT-related functions, hax will crash!" % e)

import hax
from hax import runs
from hax.paxroot import loop_over_dataset
from hax.utils import find_file_in_folders, get_user_id

log = logging.getLogger('hax.minitrees')

# update_treemakers() will update this to contain all treemakers included with hax
TREEMAKERS = {}


class TreeMaker(object):
    """Treemaker base class.

    If you're seeing this as the documentation of an actual TreeMaker, somebody forgot to add documentation
    for their treemaker
    """
    cache_size = 1000
    branch_selection = None     # List of branches to load during iteration over events
    extra_branches = tuple()    # If the above is empty, load basic branches (set in hax.config) plus these.
    uses_arrays = False         # Set to True if your treemaker returns array values. This will trigger a different
                                # root file saving code.

    def __init__(self):
        # Support for string arguments
        if isinstance(self.branch_selection, str):
            self.branch_selection = [self.branch_selection]
        if isinstance(self.extra_branches, str):
            self.extra_branches = [self.extra_branches]

        if not self.branch_selection:
            self.branch_selection = hax.config['basic_branches'] + list(self.extra_branches)
        if 'event_number' not in self.branch_selection:
            self.branch_selection += ['event_number']
        self.cache = []

    def extract_data(self, event):
        raise NotImplementedError()

    def process_event(self, event):
        result = self.extract_data(event)
        if not isinstance(result, dict):
            raise ValueError("TreeMakers must always extract dictionary")
        # Add the run and event number to the result. This is required to make joins succeed later on.
        result['event_number'] = event.event_number
        result['run_number'] = self.run_number
        self.cache.append(result)
        self.check_cache()

    def get_data(self, dataset):
        """Return data extracted from running over dataset"""
        self.run_name = runs.get_run_name(dataset)
        self.run_number = runs.get_run_number(dataset)
        loop_over_dataset(dataset, self.process_event,
                          branch_selection=self.branch_selection,
                          desc='Making %s minitree' % self.__class__.__name__)
        self.check_cache(force_empty=True)
        if not hasattr(self, 'data'):
            self.log.warning("Not a single row was extracted from dataset %s!" % dataset)
            return pd.DataFrame([], columns='event_number')
        else:
            return self.data

    def check_cache(self, force_empty=False):
        if not len(self.cache) or (len(self.cache) < self.cache_size and not force_empty):
            return
        if not hasattr(self, 'data'):
            self.data = pd.DataFrame(self.cache)
        else:
            self.data = self.data.append(self.cache, ignore_index=True)
        self.cache = []


class MultipleRowExtractor(TreeMaker):
    """Base class for treemakers that return a list of dictionaries in extract_data.
    These treemakers can produce anywhere from zeroto  or many rows for a single event.
    """

    def process_event(self, event):
        result = self.extract_data(event)
        if not isinstance(result, (list, tuple)):
            raise TypeError("MultipleRowExtractor treemakers must extract "
                            "a list of dictionaries, not a %s" % type(result))
        # Add the run and event number to the result. This is required to make joins succeed later on.
        for i in range(len(result)):
            result[i]['run_number'] = self.run_number
            result[i]['event_number'] = event.event_number
        assert len(result) == 0 or isinstance(result[0], dict)
        self.cache.extend(result)
        self.check_cache()


def update_treemakers():
    """Update the list of treemakers hax knows. Called on hax init, you should never have to call this yourself!"""
    global TREEMAKERS
    TREEMAKERS = {}
    for module_filename in glob(os.path.join(hax.hax_dir + '/treemakers/*.py')):
        module_name = os.path.splitext(os.path.basename(module_filename))[0]
        if module_name.startswith('_'):
            continue

        # Import the module, after which we can do hax.treemakers.blah
        __import__('hax.treemakers.%s' % module_name, globals=globals())

        # Now get all the treemakers defined in the module
        for tm_name, tm in inspect.getmembers(getattr(hax.treemakers, module_name),
                                                      lambda x: type(x) == type and issubclass(x, TreeMaker)):
            if tm_name == 'TreeMaker':
                # This one is the base class; we get it because we did from ... import TreeMaker at the top of the file
                continue
            if tm_name in TREEMAKERS:
                raise ValueError("Two treemakers named %s!" % tm_name)
            TREEMAKERS[tm_name] = tm


def _check_minitree_path(minitree_filename, treemaker, run_name, force_reload=False, use_root=True, use_pickle=False):
    """Return path to minitree_filename if we can find it AND it agrees with the version policy, else returns None.
    If force_reload=True, always returns None.
    """
    if force_reload:
        return None

    version_policy = hax.config['pax_version_policy']

    try:
        minitree_path = find_file_in_folders(minitree_filename, hax.config['minitree_paths'])

    except FileNotFoundError:
        log.debug("Minitree %s not found, will be created" % minitree_filename)
        return None

    log.debug("Found minitree at %s" % minitree_path)
    if use_pickle and not use_root:
        minitree_f = pd.read_pickle(minitree_path)
        minitree_metadata = minitree_f['metadata']
    else:
        minitree_f =  ROOT.TFile(minitree_path)
        minitree_metadata = json.loads(minitree_f.Get('metadata').GetTitle())

    def cleanup():
        if use_root or not use_pickle:
            minitree_f.Close()
        return None

    # Check if the minitree has an outdated treemaker version
    if LooseVersion(minitree_metadata['version']) < treemaker.__version__:
        log.debug("Minitreefile %s is outdated (version %s, treemaker is version %s), will be recreated" % (
            minitree_path, minitree_metadata['version'], treemaker.__version__))
        return cleanup()

    # Check for incompatible hax version (e.g. event_number and run_number columns not yet included in each minitree)
    if (LooseVersion(minitree_metadata.get('hax_version', '0.0')) < hax.config['minimum_minitree_hax_version']):
        log.debug("Minitreefile %s is from an incompatible hax version and must be recreated" % minitree_path)
        return cleanup()

    # Check if pax_version agrees with the version policy
    if version_policy == 'latest':
        try:
            pax_metadata = hax.paxroot.get_metadata(run_name)
        except FileNotFoundError:
            log.warning("Minitree %s was found, but the main data root file was not. "
                        "Your version policy is 'latest', so I guess I'll just this one..." % (minitree_path))
        else:
            if ('pax_version' not in minitree_metadata or
                    LooseVersion(minitree_metadata['pax_version']) <
                        LooseVersion(pax_metadata['file_builder_version'])):
                log.debug("Minitreefile %s is from an outdated pax version (pax %s, %s available), "
                          "will be recreated." % (minitree_path,
                                                  minitree_metadata.get('pax_version', 'not known'),
                                                  pax_metadata['file_builder_version']))
                return cleanup()

    elif version_policy == 'loose':
        pass

    else:
        if not minitree_metadata['pax_version'] == version_policy:
            log.debug("Minitree found from pax version %s, but you required pax version %s. "
                      "Will attempt to create it from the main root file." % (minitree_metadata['pax_version'],
                                                                              version_policy))
            if use_root or not use_pickle:
                minitree_f.Close()
            return None

    if use_root or not use_pickle:
        minitree_f.Close()
    return minitree_path


def load_single(run_name, treemaker, force_reload=False, use_root=True, use_pickle=False):
    """Return pandas DataFrame resulting from running treemaker on run_name (can also be a run number).
    :param run_name: name or number of the run to load
    :param treemaker: TreeMaker class (not instance!) to run
    For other arguments, see load docstring.

    Raises FileNotFoundError if we need the parent pax root file, but can't find it.
    """
    run_name = runs.get_run_name(run_name)
    treemaker_name, treemaker = get_treemaker_name_and_class(treemaker)
    if not hasattr(treemaker, '__version__'):
        raise AttributeError("Please add a __version__ attribute to treemaker %s." % treemaker_name)
    minitree_filename = "%s_%s.root" % (run_name, treemaker_name)
    if use_pickle:
        minitree_pickle_filename = "%s_%s.pkl" % (run_name, treemaker_name)
        if not use_root:
            minitree_filename = minitree_pickle_filename

    # Do we already have this minitree? And is it good?
    minitree_path = _check_minitree_path(minitree_filename, treemaker, run_name,
                                         force_reload=force_reload, use_pickle=use_pickle, use_root=use_root)
    if minitree_path is not None:
        if use_pickle and not use_root:
            loaded_frame = pd.read_pickle(minitree_path)[treemaker_name]
        else:
            loaded_frame = pd.DataFrame.from_records(root_numpy.root2rec(minitree_path))
        return loaded_frame

    # We have to make the minitree file
    # This will raise FileNotFoundError if the root file is not found
    skimmed_data = treemaker().get_data(run_name)
    
    # Custom code is needed to save array fields to a ROOT file. Check if we need to / have permission to use it.
    if not treemaker.uses_arrays and use_root:
        for branch_name in skimmed_data.columns:
            if is_array_field(skimmed_data, branch_name):
                raise NotImplementedError("Column %s is an array field, and you want to save to root. Either "
                         "(1) add a uses_arrays=True attribute to the %s class; or"
                         "(2) use pickle as a minitree caching format; or"
                         "(3) use the DataExtractor class." % (branch_name, treemaker_name))

    log.debug("Created minitree %s for dataset %s" % (treemaker.__name__, run_name))

    # Make a minitree in the first (highest priority) directory from minitree_paths
    # This ensures we will find exactly this file when we load the minitree next.
    creation_dir = hax.config['minitree_paths'][0]
    if not os.path.exists(creation_dir):
        os.makedirs(creation_dir)
    minitree_path = os.path.join(creation_dir, minitree_filename)

    metadata_dict = dict(version=treemaker.__version__,
                         pax_version=hax.paxroot.get_metadata(run_name)['file_builder_version'],
                         hax_version=hax.__version__,
                         created_by=get_user_id(),
                         documentation=treemaker.__doc__,
                         timestamp=str(datetime.now()))
    if use_pickle:
        # Write metadata
        minitree_pickle_path = os.path.join(creation_dir, minitree_pickle_filename)
        pickle_dict = {'metadata': metadata_dict, treemaker.__name__: skimmed_data}
        pickle.dump(pickle_dict, open(minitree_pickle_path, 'wb'))
    if use_root:
        if treemaker.uses_arrays:
            dataframe_to_root(skimmed_data, minitree_path, treename=treemaker.__name__, mode='recreate')
        else:
            root_numpy.array2root(skimmed_data.to_records(), minitree_path,
                                  treename=treemaker.__name__, mode='recreate')
        # Write metadata
        bla = ROOT.TNamed('metadata', json.dumps(metadata_dict))
        minitree_f = ROOT.TFile(minitree_path, 'UPDATE')
        bla.Write()
        minitree_f.Close()
    return skimmed_data


def load(datasets, treemakers=tuple(['Fundamentals', 'Basics']),
         force_reload=False, use_root=True, use_pickle=False):
    """Return pandas DataFrame with minitrees of several datasets and treemakers.
    :param datasets: names or numbers of datasets (without .root) to load
    :param treemakers: treemaker class (or string with name of class) or list of these to load.
    :param force_reload: if True, will force mini-trees to be re-made whether they are outdated or not.
    :param use_root: use ROOT to read/write cached minitrees
    :param use_pickle: use ROOT to read/write cached minitrees
    """
    if isinstance(datasets, (str, int, np.int64, np.int, np.int32)):
        datasets = [datasets]
    if isinstance(treemakers, (type, str)):
        treemakers = [treemakers]

    combined_dataframes = []

    for treemaker in treemakers:

        dataframes = []
        for dataset in datasets:
            dataset_frame = load_single(dataset, treemaker,
                                        force_reload=force_reload, use_root=use_root, use_pickle=use_pickle)
            dataframes.append(dataset_frame)

        # Concatenate mini-trees of this type for all datasets
        combined_dataframes.append(pd.concat(dataframes, ignore_index=True))

    # Merge mini-trees of all types by inner join (propagating cuts)
    if not len(combined_dataframes):
        raise RuntimeError("No data was extracted? What's going on??")
    result = combined_dataframes[0]
    for i in range(1, len(combined_dataframes)):
        d = combined_dataframes[i]
        # To avoid creation of duplicate columns (which will get _x and _y suffixes),
        # look which column names already exist and do not include them in the merge
        cols_to_use = ['run_number', 'event_number'] + d.columns.difference(result.columns).tolist()
        result = pd.merge(d[cols_to_use], result, on=['run_number', 'event_number'], how='inner')

    if 'index' in result.columns:
        # Clean up index, remove 'index' column
        # Probably we're doing something weird with pandas, this doesn't seem like the well-trodden path...
        # TODO: is this still triggered / necessary?
        log.debug("Removing weird index column")
        result.drop('index', axis=1, inplace=True)
        result = result.reset_index()
        result.drop('index', axis=1, inplace=True)

    return result


def get_treemaker_name_and_class(tm):
    """Return (name, class) of treemaker name or class tm"""
    if isinstance(tm, str):
        if not tm in TREEMAKERS:
            raise ValueError("No TreeMaker named %s known to hax!" % tm)
        return tm, TREEMAKERS[tm]
    elif isinstance(tm, type) and issubclass(tm, TreeMaker):
        return tm.__name__, tm
    else:
        raise ValueError("%s is not a TreeMaker child class or name, but a %s" % (tm, type(tm)))


##
# Utilities for saving array fields in pandas dataframes to ROOT files
# This is not supported natively by root_numpy.
##

def is_array_field(test_dataframe, test_field):
    """Tests if the column test_field in test_dataframe is an array field
    :param test_dataframe: dataframe to test
    :param test_field: column name to test
    :return: True or False
    """
    if test_dataframe.empty:
        raise ValueError("No data saved from dataset - DataFrame is empty")
    test_value = test_dataframe[test_field][0]
    return (hasattr(test_value, "__len__") and not isinstance(test_value, (str, bytes)))


def dataframe_to_root(dataframe, root_filename, treename='tree', mode='recreate'):
    branches = {}
    branch_types = {}

    single_value_keys = []
    array_keys = []
    array_root_file = ROOT.TFile(root_filename, mode)
    datatree = ROOT.TTree(treename, "")

    # setting up branches
    for branch_name in dataframe.columns:
        if is_array_field(dataframe, branch_name):
            # This is an array field. Find or create its 'length branch',
            # needed for saving the array to root (why exactly? Wouldn't a vector work?)
            length_branch_name = branch_name + '_length'
            if not length_branch_name in dataframe.columns:
                dataframe[length_branch_name] = np.array([len(x) for x in dataframe[branch_name]], dtype=np.int64)
                single_value_keys.append(length_branch_name)
                branches[length_branch_name] = np.array([0])
                branch_types[length_branch_name] = 'L'
            max_length = dataframe[length_branch_name].max()
            first_element = dataframe[branch_name][0][0]
            array_keys.append(branch_name)

        else:
            # Ordinary scalar field
            max_length = 1
            first_element = dataframe[branch_name][0]
            single_value_keys.append(branch_name)

        # setting branch types
        if isinstance(first_element, (int, np.integer)):
            branch_type = 'L'
            branches[branch_name] = np.zeros(max_length, dtype=np.int64)
        elif isinstance(first_element, (float, np.float)):
            branch_type = 'D'
            branches[branch_name] = np.zeros(max_length, dtype=np.float64)
        else:
            raise TypeError('Branches must contain ints, floats, or arrays of ints or floats' )
        branch_types[branch_name] = branch_type

    # creating branches
    for single_value_key in single_value_keys:
        datatree.Branch(single_value_key, branches[single_value_key],
                        "%s/%s" % (single_value_key, branch_types[single_value_key]))
    for array_key in array_keys:
        assert array_key + '_length' in dataframe.columns
        datatree.Branch(array_key, branches[array_key],
                        "%s[%s]/%s" % (array_key, array_key + "_length", branch_types[array_key]))

    # filling tree
    for event_index in range(len(dataframe.index)):
        for single_value_key in single_value_keys:
            branches[single_value_key][0] = dataframe[single_value_key][event_index]
        for array_key in array_keys:
            branches[array_key][:len(dataframe[array_key][event_index])] = dataframe[array_key][event_index]
        datatree.Fill()
    array_root_file.Write()
    array_root_file.Close()
