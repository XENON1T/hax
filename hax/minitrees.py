"""Make small flat root trees with one entry per event from the pax root files.
"""


from datetime import datetime
from distutils.version import LooseVersion
from glob import glob
import inspect
import json
import os

import pandas as pd
import ROOT
import root_numpy

import hax
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
    extra_branches = tuple()

    def __init__(self):
        self.cache = []

    def extract_data(self, event):
        raise NotImplementedError()

    def process_event(self, event):
        self.cache.append(self.extract_data(event))
        self.check_cache()

    def get_data(self, dataset):
        """Return data extracted from running over dataset"""
        loop_over_dataset(dataset, self.process_event,
                          branch_selection=hax.config['basic_branches'] + list(self.extra_branches))
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


def get(dataset, treemaker, force_reload=False):
    """Return path to minitree file from treemaker for dataset.
    The file will be re-created if it is not present, outdated, or force_reload is True (default False)
    """
    global treemakers
    treemaker_name, treemaker = get_treemaker_name_and_class(treemaker)
    if not hasattr(treemaker, '__version__'):
        raise RuntimeError("Please add a __version__ attribute to treemaker %s" % treemaker_name)
    minitree_filename = "%s_%s.root" % (dataset,
                                        treemaker_name)

    try:
        minitree_path = find_file_in_folders(minitree_filename, hax.config['mini_tree_paths'])
        print("Found minitree at %s" % minitree_path)

        # Check the version of the minitree file
        f = ROOT.TFile(minitree_path, 'UPDATE')
        metadata = json.loads(f.Get('metadata').GetTitle())
        if LooseVersion(metadata['version']) < treemaker.__version__:
            print("Minitreefile %s is outdated (version %s, treemaker is version %s), will be recreated" % (minitree_path,
                                                                                                            metadata['version'],
                                                                                                            treemaker.__version__))
            minitree_path = None
        f.Close()

    except FileNotFoundError:
        minitree_path = None

    if minitree_path is None or force_reload:
        # We have to make the minitree file
        # This will raise FileNotFoundError if the root file is not found
        skimmed_data = treemaker().get_data(dataset)
        print("Created minitree %s for dataset %s" % (treemaker.__name__, dataset))

        # Make a minitree in the current directory
        minitree_path = './' + minitree_filename
        root_numpy.array2root(skimmed_data.to_records(), minitree_path,
                              treename=treemaker.__name__, mode='recreate')

        # Write metadata
        f = ROOT.TFile(minitree_path, 'UPDATE')
        ROOT.TNamed('metadata', json.dumps(dict(version=treemaker.__version__,
                                                created_by=get_user_id(),
                                                documentation=treemaker.__doc__,
                                                timestamp=str(datetime.now())))).Write()
        f.Close()

    return minitree_path


def load(datasets, treemakers='Basics', force_reload=False):
    """Return pandas DataFrame with minitrees of several datasets.
      datasets: names of datasets (without .root) to load
      treemakers: treemaker class (or string with name of class) or list of these to load. Defaults to 'Basics'.
      force_reload: if True, will force mini-trees to be re-made whether they are outdated or not.
    """
    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(treemakers, (type, str)):
        treemakers = [treemakers]

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
    return pd.concat(combined_dataframes, axis=1)


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
