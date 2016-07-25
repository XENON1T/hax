"""Make small flat root trees with one entry per event from the pax root files.
"""
from datetime import datetime
from distutils.version import LooseVersion
from glob import glob
import inspect
import logging
import json
import os
log = logging.getLogger('hax.minitrees')

import numpy as np
import pandas as pd
import ROOT
import root_numpy

import hax
from hax import runs
from hax.paxroot import loop_over_dataset
from hax.utils import find_file_in_folders, get_user_id

# Will be updated to contain all treemakers
treemakers = {}


class TreeMaker(object):
    """Treemaker base class.

    If you're seeing this as the documentation of an actual TreeMaker, somebody forgot to add documentation
    for their treemaker
    """
    cache_size = 1000
    branch_selection = None     # List of branches to load during iteration over events
    extra_branches = tuple()    # If the above is empty, load basic branches (set in hax.config) plus these.

    def __init__(self):
        if not self.branch_selection:
            self.branch_selection = hax.config['basic_branches'] + list(self.extra_branches)
        self.cache = []

    def extract_data(self, event):
        raise NotImplementedError()

    def process_event(self, event):
        self.cache.append(self.extract_data(event))
        self.check_cache()

    def get_data(self, dataset):
        """Return data extracted from running over dataset"""
        self.run_name = runs.get_run_name(dataset)
        self.run_number = runs.get_run_number(dataset)
        loop_over_dataset(dataset, self.process_event,
                          branch_selection=self.branch_selection)
        self.check_cache(force_empty=True)
        if not hasattr(self, 'data'):
            raise RuntimeError("Not a single event was extracted from dataset %s!" % dataset)
        else:
            return self.data

    def check_cache(self, force_empty=False):
        if not len(self.cache) or (len(self.cache) < self.cache_size and not force_empty):
            return
        if not hasattr(self, 'data'):
            self.data = pd.DataFrame(self.cache)
        else:
            self.data = self.data.append(self.cache)
        self.cache = []


def update_treemakers():
    """Update the list of treemakers hax knows. Called on hax init, you should never have to call this yourself!"""
    global treemakers
    treemakers = {}
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
            if tm_name in treemakers:
                raise ValueError("Two treemakers named %s!" % tm_name)
            treemakers[tm_name] = tm


def _check_minitree_path(minitree_filename, treemaker, run_name, force_reload=False):
    """Return path to minitree_filename if we can find it and it agrees with the version policy, else returns None.
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
    minitree_f =  ROOT.TFile(minitree_path)
    minitree_metadata = json.loads(minitree_f.Get('metadata').GetTitle())

    # Check if the minitree has an outdated treemaker version
    if LooseVersion(minitree_metadata['version']) < treemaker.__version__:
        log.debug("Minitreefile %s is outdated (version %s, treemaker is version %s), will be recreated" % (
            minitree_path, minitree_metadata['version'], treemaker.__version__))
        minitree_f.Close()
        return None

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
                minitree_f.Close()
                return None

    elif version_policy == 'loose':
        pass

    else:
        if not minitree_metadata['pax_version'] == version_policy:
            log.debug("Minitree found from pax version %s, but you required pax version %s. "
                      "Will attempt to create it from the main root file." % (minitree_metadata['pax_version'],
                                                                              version_policy))
            minitree_f.Close()
            return None

    minitree_f.Close()
    return minitree_path


def get(run_name, treemaker, force_reload=False):
    """Return path to minitree file from treemaker for run_name (can also be a run number).
    The file will be re-created if it is not present, outdated, or force_reload is True (default False)
    Raises FileNotFoundError if we have to create the minitree, but the root file is not found.
    """
    global treemakers
    run_name = runs.get_run_name(run_name)
    treemaker_name, treemaker = get_treemaker_name_and_class(treemaker)
    if not hasattr(treemaker, '__version__'):
        raise AttributeError("Please add a __version__ attribute to treemaker %s." % treemaker_name)
    minitree_filename = "%s_%s.root" % (run_name, treemaker_name)

    # Do we already have this minitree? And is it good?
    minitree_path = _check_minitree_path(minitree_filename, treemaker, run_name,
                                         force_reload=force_reload)
    if minitree_path is not None:
        return minitree_path

    # We have to make the minitree file
    # This will raise FileNotFoundError if the root file is not found
    skimmed_data = treemaker().get_data(run_name)
    log.debug("Created minitree %s for dataset %s" % (treemaker.__name__, run_name))

    # Make a minitree in the first (highest priority) directory from minitree_paths
    # This ensures we will find exactly this file when we load the minitree next.
    creation_dir = hax.config['minitree_paths'][0]
    if not os.path.exists(creation_dir):
        os.makedirs(creation_dir)
    minitree_path = os.path.join(creation_dir, minitree_filename)
    root_numpy.array2root(skimmed_data.to_records(), minitree_path,
                          treename=treemaker.__name__, mode='recreate')

    # Write metadata
    bla = ROOT.TNamed('metadata', json.dumps(dict(version=treemaker.__version__,
                                            pax_version=hax.paxroot.get_metadata(run_name)['file_builder_version'],
                                            created_by=get_user_id(),
                                            documentation=treemaker.__doc__,
                                            timestamp=str(datetime.now()))))
    minitree_f = ROOT.TFile(minitree_path, 'UPDATE')
    bla.Write()
    minitree_f.Close()

    return minitree_path


def load(datasets, treemakers='Basics', force_reload=False):
    """Return pandas DataFrame with minitrees of several datasets.
      datasets: names or numbers of datasets (without .root) to load
      treemakers: treemaker class (or string with name of class) or list of these to load. Defaults to 'Basics'.
      force_reload: if True, will force mini-trees to be re-made whether they are outdated or not.
    """
    if isinstance(datasets, (str, int, np.int64, np.int, np.int32)):
        datasets = [datasets]
    if isinstance(treemakers, (type, str)):
        treemakers = [treemakers]

    # Add the "Fundamentals" treemaker to the beginning; we always want to load this.
    treemakers = ['Fundamentals'] + treemakers

    combined_dataframes = []

    for treemaker in treemakers:

        dataframes = []
        for dataset in datasets:
            minitree_path = get(dataset, treemaker, force_reload=force_reload)
            new_df = pd.DataFrame.from_records(root_numpy.root2rec(minitree_path))
            dataframes.append(new_df)

        # Concatenate mini-trees of this type for all datasets
        combined_dataframes.append(pd.concat(dataframes))

    # Concatenate mini-trees of all types
    if not len(combined_dataframes):
        raise RuntimeError("No data was extracted? What's going on??")
    result = pd.concat(combined_dataframes, axis=1)

    # Clean up index, remove 'index' column
    # Probably we're doing something weird with pandas, this doesn't seem like the well-trodden path...
    result.drop('index', axis=1, inplace=True)
    result = result.reset_index()
    result.drop('index', axis=1, inplace=True)
    return result


def get_treemaker_name_and_class(tm):
    """Return (name, class) of treemaker name or class tm"""
    global treemakers
    if isinstance(tm, str):
        if not tm in treemakers:
            raise ValueError("No TreeMaker named %s known to hax!" % tm)
        return tm, treemakers[tm]
    elif isinstance(tm, type) and issubclass(tm, TreeMaker):
        return tm.__name__, tm
    else:
        raise ValueError("%s is not a TreeMaker child class or name, but a %s" % (tm, type(tm)))
