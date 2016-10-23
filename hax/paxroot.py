"""Utility functions for loading and looping over a pax root file
"""
import logging
import os
import numpy as np
import json
import warnings

from tqdm import tqdm
from pax.exceptions import MaybeOldFormatException

try:
    import ROOT
    from pax.plugins.io.ROOTClass import load_event_class, load_pax_event_class_from_root
except ImportError as e:
    warnings.warn("Error importing ROOT-related libraries: %s. "
                  "If you try to use ROOT-related functions, hax will crash!" % e)

import hax
from hax import runs
from hax.utils import find_file_in_folders


log = logging.getLogger('hax.paxroot')


def open_pax_rootfile(run_id, load_class=True):
    """Opens pax root file for run_id, compiling classes/dictionaries as needed. Returns TFile object.
    """
    try:
        run_name = runs.get_run_name(run_id)
        filename = runs.datasets.loc[runs.datasets['name'] == run_name].iloc[0].location
    except IndexError:
        print("Don't know a run named %s, trying to find it anyway..." % run_id)
        filename = find_file_in_folders(run_id + '.root', hax.config['main_data_paths'])
    if not filename:
        raise ValueError("Cannot find processed data for run name %s." % run_name)
    return _open_pax_rootfile(filename, load_class=True)

def _open_pax_rootfile(filename, load_class):
    """Opens pax root file filename, compiling classes/dictionaries as needed. Returns TFile object.
    """
    if not os.path.exists(filename):
        raise ValueError("%s does not exist!" % filename)
    if load_class:
        try:
            load_pax_event_class_from_root(filename)
        except MaybeOldFormatException:
            log.warning("Root file %s does not include pax event class. Normal for pax < 4.5."
                        "Falling back to event class for pax %s" % (filename, hax.config['old_pax_class_version']))
            # Load the pax class for the data format version
            load_event_class(os.path.join(hax.config['old_pax_classes_dir'],
                                          'pax_event_class_%d.cpp' % hax.config['old_pax_class_version']))
    return ROOT.TFile(filename)


def get_metadata(run_id):
    """Returns the metadata dictionary stored in the pax root file for run_id.
    """
    f = open_pax_rootfile(run_id, load_class=False)
    metadata = f.Get('pax_metadata').GetTitle()
    metadata = json.loads(metadata)
    f.Close()
    return metadata


# An exception you can raise to stop looping over the current dataset
class StopEventLoop(Exception):
    pass


def function_results_datasets(datasets_names, event_function=lambda event, **kwargs: None,
                              branch_selection=None, kwargs=None, desc=''):
    """Returns a generator which yields the return values of event_function(event) over the datasets specified in
    datasets_names.

    Parameters are equivalent to loop_over_datasets, except for the addition of kwargs, which allows for additional
    arguments to be passed to event_function if the function is able to take more than one. kwargs should be a dict
    mapping argument names to their respective values.
    Example: kwargs={'x': 2, 'y': 3} --> function called like: event_function(event, x=2, y=3)
    """
    if kwargs is None:
        kwargs = {}

    if not isinstance(datasets_names, (list, tuple, np.ndarray)):
        datasets_names = [datasets_names]

    for run_id in datasets_names:
        rootfile = open_pax_rootfile(run_id)
        # If you get "'TObject' object has no attribute 'GetEntries'" here,
        # we renamed the tree to T1 or TPax or something... or you're trying to load a Xerawdp root file!
        t = rootfile.Get('tree')
        n_events = t.GetEntries()

        if branch_selection == 'basic':
            branch_selection = hax.config['basic_branches']

        # Activate the desired branches
        if branch_selection:
            t.SetBranchStatus("*", 0)
            for bn in branch_selection:
                t.SetBranchStatus(bn, 1)

        try:
            source = range(n_events)
            if hax.config.get('tqdm_on', True):
                source = tqdm(source, desc='Run %s: %s' % (run_id, desc))
            for event_i in source:
                t.GetEntry(event_i)
                event = t.events
                yield event_function(event, **kwargs)

        except StopEventLoop:
            rootfile.Close()
        except Exception as e:
            rootfile.Close()
            raise e


def loop_over_datasets(datasets_names, event_function=lambda event: None, branch_selection=None, desc=''):
    """Execute event_function(event) over all events in the dataset(s)
    Does not return anything: use function_results_dataset or pass a class method as event_function if you want results.
     - list of datataset names or numbers
     - event_function: function to run
     - branch selection: can be
        None (all branches are read),
        'basic' (hax.config['basic_branches'] are read), or
        a list of branches to read.
    """
    for _ in function_results_datasets(datasets_names, event_function, branch_selection, desc=desc):
        # do nothing with the results
        pass

# For backward compatibility
loop_over_dataset = loop_over_datasets
