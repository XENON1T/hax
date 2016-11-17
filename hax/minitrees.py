"""Make small flat root trees with one entry per event from the pax root files.
"""
from datetime import datetime
from distutils.version import LooseVersion
from glob import glob
import inspect
import logging
import os

import numpy as np
import pandas as pd
import dask
import dask.multiprocessing
import dask.dataframe

import hax
from hax import runs, cuts
from .paxroot import loop_over_dataset
from .utils import find_file_in_folders, get_user_id
from .minitree_formats import get_format

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

    If you're seeing this as the documentation of an actual TreeMaker, somebody forgot to add documentation
    for their treemaker.
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


def _minitree_filename(run_name, treemaker_name, extension):
    return "%s_%s.%s" % (run_name, treemaker_name, extension)


def check(run_id, treemaker, force_reload=False):
    """Return if the minitree exists and where it is found / where to make it.
    :param treemaker: treemaker name or class
    :param run_id: run name or number
    :param force_reload: ignore available minitrees, just tell me where to write the new one.
    :returns : (treemaker, available, path).
      - treemaker_class: class of the treemaker you named.
      - already_made is True if there is an up-to-date minitree we can load, False otherwise (always if force_reload)
      - path is the path to the minitree to load if it is available, otherwise path where we should create the minitree.
    """
    run_name = runs.get_run_name(run_id)
    treemaker_name, treemaker = get_treemaker_name_and_class(treemaker)
    preferred_format = hax.config['preferred_minitree_format']

    # If we need to remake the minitree, where would we place it?
    minitree_filename = _minitree_filename(run_name, treemaker_name, preferred_format)
    creation_dir = hax.config['minitree_paths'][0]
    if not os.path.exists(creation_dir):
        os.makedirs(creation_dir)
    path_to_new = os.path.join(creation_dir, minitree_filename)

    # Value to return if the minitree is not available
    sorry_not_available = treemaker, False, path_to_new

    if force_reload:
        return sorry_not_available

    # Find the file
    try:
        minitree_path = find_file_in_folders(minitree_filename, hax.config['minitree_paths'])
    except FileNotFoundError:
        # Maybe it exists, but was made in a non-preferred file format
        log.debug("Minitree %s not found" % minitree_filename)
        for mt_format in hax.config['other_minitree_formats']:
            if mt_format == preferred_format:
                # Already tried this format
                continue
            else:
                try:
                    minitree_filename = _minitree_filename(run_name, treemaker_name, mt_format)
                    minitree_path = find_file_in_folders(minitree_filename, hax.config['minitree_paths'])
                    log.debug("Minitree found in non-preferred format: %s" % minitree_filename)
                    break
                except FileNotFoundError:
                    log.debug("Not found in non-preferred formats either. Minitree will be created.")
                    pass
        else:
            # Not found in any format
            return sorry_not_available

    log.debug("Found minitree at %s" % minitree_path)

    # Load the metadata ONLY, to see if we can load this file
    minitree_metadata = get_format(minitree_path).load_metadata()

    # Check if the minitree has an outdated treemaker version
    if LooseVersion(minitree_metadata['version']) < treemaker.__version__:
        log.debug("Minitreefile %s is outdated (version %s, treemaker is version %s), will be recreated" % (
            minitree_path, minitree_metadata['version'], treemaker.__version__))
        return sorry_not_available

    # Check for incompatible hax version (e.g. event_number and run_number columns not yet included in each minitree)
    if (LooseVersion(minitree_metadata.get('hax_version', '0.0')) < hax.config['minimum_minitree_hax_version']):
        log.debug("Minitreefile %s is from an incompatible hax version and must be recreated" % minitree_path)
        return sorry_not_available

    # Check if pax_version agrees with the version policy.
    version_policy = hax.config['pax_version_policy']
    if version_policy == 'latest':
        # What the latest pax version is differs per dataset. For now we'll open the root file to find out
        # TODO: we shouldn't need to; the runs db keeps track of this, and we use it in hax.runs for this purpose!
        try:
            pax_metadata = hax.paxroot.get_metadata(run_name)
        except FileNotFoundError:
            log.warning("Minitree %s was found, but the main data root file was not. "
                        "Your version policy is 'latest', but I can't check whether you really have the latest... "
                        "well, let's load it and see what happens." % minitree_path)
        else:
            if ('pax_version' not in minitree_metadata or
                    LooseVersion(minitree_metadata['pax_version']) <
                        LooseVersion(pax_metadata['file_builder_version'])):
                log.debug("Minitreefile %s is from an outdated pax version (pax %s, %s available), "
                          "will be recreated." % (minitree_path,
                                                  minitree_metadata.get('pax_version', 'not known'),
                                                  pax_metadata['file_builder_version']))
                return sorry_not_available

    elif version_policy == 'loose':
        # Anything goes
        pass

    else:
        if not minitree_metadata['pax_version'] == version_policy:
            log.debug("Minitree found from pax version %s, but you required pax version %s. "
                      "Will attempt to create it from the main root file." % (minitree_metadata['pax_version'],
                                                                              version_policy))
            return sorry_not_available

    return treemaker, True, minitree_path


def load_single_minitree(run_id, treemaker, force_reload=False, return_metadata=False):
    """Return pandas DataFrame resulting from running treemaker on run_id (name or number)
    :returns: pandas.DataFrame
    :param run_id: name or number of the run to load
    :param treemaker: TreeMaker class or class name (but not TreeMaker instance!) to run
    :param force_reload: always remake the minitree, never load it from disk.
    :param return_metadata: instead return (metadata_dict, dataframe)
    """
    treemaker, already_made, minitree_path = check(run_id, treemaker, force_reload=force_reload)

    if already_made:
        return get_format(minitree_path).load_data()

    if not hax.config['make_minitrees']:
        # The user didn't want me to make a new minitree :-(
        raise NoMinitreeAvailable("Minitree %s:%s not created since make_minitrees is False." % (
                run_id, treemaker.__name__))

    # We have to make the minitree file
    # This will raise FileNotFoundError if the root file is not found
    skimmed_data = treemaker().get_data(run_id)

    log.debug("Retrieved %s minitree data for dataset %s" % (treemaker.__name__, run_id))

    metadata_dict = dict(version=treemaker.__version__,
                         pax_version=hax.paxroot.get_metadata(run_id)['file_builder_version'],
                         hax_version=hax.__version__,
                         created_by=get_user_id(),
                         documentation=treemaker.__doc__,
                         timestamp=str(datetime.now()))

    if hax.config['minitree_caching']:
        get_format(minitree_path, treemaker).save_data(metadata_dict, skimmed_data)

    if return_metadata:
        return metadata_dict, skimmed_data

    return skimmed_data


def load_single_dataset(run_id, treemakers, preselection, force_reload=False):
    """Return pandas DataFrame resulting from running multiple treemakers on run_id (name or number),
    list of dicts describing cut histories.
    :param run_id: name or number of the run to load
    :param treemakers: list of treemaker class / instances to load
    :param preselection: String or list of strings passed to pandas.eval. Should return bool array, to be used
    for pre-selecting events to load for each dataset.
    :param force_reload: always remake the minitrees, never load any from disk.
    """
    if isinstance(treemakers, (type, str)):
        treemakers = [treemakers]
    if isinstance(preselection, str):
        preselection = [preselection]
    if preselection is None:
        preselection = []
    dataframes = []

    for treemaker in treemakers:
        try:
            dataset_frame = load_single_minitree(run_id, treemaker, force_reload=force_reload)
        except NoMinitreeAvailable as e:
            log.debug(str(e))
            return pd.DataFrame([], columns=['event_number', 'run_number']), []
        dataframes.append(dataset_frame)

    # Merge mini-trees of all types by inner join
    # (propagating "cuts" applied by skipping rows in MultipleRowExtractor)
    if not len(dataframes):
        raise RuntimeError("No data was extracted? What's going on??")
    result = dataframes[0]
    for i in range(1, len(dataframes)):
        d = dataframes[i]
        # To avoid creation of duplicate columns (which will get _x and _y suffixes),
        # look which column names already exist and do not include them in the merge
        cols_to_use = ['run_number', 'event_number'] + d.columns.difference(result.columns).tolist()
        result = pd.merge(d[cols_to_use], result, on=['run_number', 'event_number'], how='inner')

    # Apply pre-selection cuts before moving on to the next dataset
    for ps in preselection:
        result = cuts.eval_selection(result, ps, quiet=True)

    return result, cuts._get_history(result)


def load(datasets, treemakers=tuple(['Fundamentals', 'Basics']), preselection=None, force_reload=False,
         delayed=False, num_workers=1, compute_options=None, cache_file=None, remake_cache=False):
    """Return pandas DataFrame with minitrees of several datasets and treemakers.
    :param datasets: names or numbers of datasets (without .root) to load
    :param treemakers: treemaker class (or string with name of class) or list of these to load.
    :param preselection: string or list of strings parseable by pd.eval. Should return bool array, to be used
    for pre-selecting events to load for each dataset.
    :param force_reload: if True, will force mini-trees to be re-made whether they are outdated or not.
    :param delayed:  Instead of computing a pandas DataFrame, return a dask DataFrame (default False)
    :param num_workers: Number of dask workers to use in computation (if delayed=False)
    :param compute_options: Dictionary of extra options passed to dask.compute
    :param cache_file: Save/load the result to an hdf5 file with filename specified by cahce_file.
                       Useful if you load in a large volume of data with many preselections.
    :param remake_cache: If True, and cache file given, reload (don't remake) minitrees and overwrite the cache file.
    """
    if cache_file and not remake_cache and os.path.exists(cache_file):
        # We don't have to do anything and can just load from the cache file
        store = pd.HDFStore(cache_file)
        result = store['data']
        result.cut_history = store.get_storer('data').attrs.cut_history
        store.close()
        return result

    if isinstance(datasets, (str, int, np.int64, np.int, np.int32)):
        datasets = [datasets]
    if compute_options is None:
        compute_options = {}
    compute_options.setdefault('get', dask.multiprocessing.get)

    partial_results = []
    partial_histories = []
    for dataset in datasets:
        mashup = dask.delayed(load_single_dataset)(dataset, treemakers, preselection, force_reload=force_reload)
        partial_results.append(dask.delayed(lambda x: x[0])(mashup))
        partial_histories.append(dask.delayed(lambda x: x[1])(mashup))

    result = dask.dataframe.from_delayed(partial_results, meta=partial_results[0].compute())

    if not delayed:
        # Dask doesn't seem to want to descend into the lists beyond the first.
        # So we mash things into one list before calling compute, then split it again
        mashedup_result = dask.compute(*([result] + partial_histories),
                                       num_workers=num_workers, **compute_options)
        result = mashedup_result[0]

        if 'index' in result.columns:
            # Clean up index, remove 'index' column
            # Probably we're doing something weird with pandas, this doesn't seem like the well-trodden path...
            log.debug("Removing weird index column")
            result.drop('index', axis=1, inplace=True)
            result = result.reset_index()
            result.drop('index', axis=1, inplace=True)

        # Combine the histories of partial results.
        # For unavailable minitrees, the histories will be empty: filter these empty histories out
        partial_histories = mashedup_result[1:]
        partial_histories = [x for x in partial_histories if len(x)]
        if len(partial_histories):
            cuts.record_combined_histories(result, partial_histories)

    else:
        # Magic for tracking of cut histories while using dask.dataframe here...
        pass

    if cache_file:
        # Save the result to the cache file
        dirname = os.path.dirname(cache_file)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        store = pd.HDFStore(cache_file)
        store.put('data', result)
        # Store the cuts history for the data
        store.get_storer('data').attrs.cut_history = cuts._get_history(result)
        store.close()

    return result


def get_treemaker_name_and_class(tm):
    """Return (name, class) of treemaker name or class tm"""
    if isinstance(tm, str):
        if not tm in TREEMAKERS:
            raise ValueError("No TreeMaker named %s known to hax!" % tm)
        tm_name, tm_class = tm, TREEMAKERS[tm]
    elif isinstance(tm, type) and issubclass(tm, TreeMaker):
        tm_name, tm_class = tm.__name__, tm
    else:
        raise ValueError("%s is not a TreeMaker child class or name, but a %s" % (tm, type(tm)))

    if not hasattr(tm_class, '__version__'):
        raise AttributeError("Please add a __version__ attribute to treemaker %s." % tm_name)
    return tm_name, tm_class


class NoMinitreeAvailable(Exception):
    pass
