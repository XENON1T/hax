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
    from pax.plugins.io.ROOTClass import load_event_class, load_pax_event_class_from_root, ShutUpROOT
except ImportError as e:
    warnings.warn("Error importing ROOT-related libraries: %s. "
                  "If you try to use ROOT-related functions, hax will crash!" % e)

import hax
from hax import runs
from hax.utils import find_file_in_folders


log = logging.getLogger('hax.paxroot')


def get_filename(run_id):
    try:
        run_name = runs.get_run_name(run_id)
        filename = runs.datasets.loc[runs.datasets['name'] == run_name].iloc[0].location
    except IndexError:
        print("Don't know a run named %s, trying to find it anyway..." % run_id)
        filename = find_file_in_folders(run_id + '.root', hax.config['main_data_paths'])
    if not filename:
        raise ValueError("Cannot find processed data for run name %s." % run_id)
    return filename


def open_pax_rootfile(run_id, load_class=True):
    """Opens pax root file for run_id, compiling classes/dictionaries as needed. Returns TFile object.
    if load_class is False, will not load the event class. You'll only be able to read metadata from the file.
    """
    return _open_pax_rootfile(get_filename(run_id), load_class=load_class)


def _open_pax_rootfile(filename, load_class=True):
    """Opens pax root file filename, compiling classes/dictionaries as needed. Returns TFile object.
    if load_class is False, will not load the event class. You'll only be able to read metadata from the file.
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
    return _get_metadata(get_filename(run_id))


def _get_metadata(filename):
    # Suppress warning about classes not being loaded (we're doing that on purpose)
    with ShutUpROOT():
        f = _open_pax_rootfile(filename, load_class=False)
    metadata = f.Get('pax_metadata').GetTitle()
    metadata = json.loads(metadata)
    f.Close()
    return metadata


# An exception you can raise to stop looping over the current dataset
class StopEventLoop(Exception):
    pass


def function_results_datasets(datasets_names,
                              event_function=lambda event, **kwargs: None,
                              event_lists=None,
                              branch_selection=None,
                              kwargs=None,
                              desc=''):
    """Returns a generator which yields the return values of event_function(event) over the datasets specified in
    datasets_names.

    :param dataset_names: list of datataset names or numbers, or string/int of a single dataset name/number

    :param event_function: function to run over each event

    :param event_lists: a list of event numbers (if you're loading in a single dataset) to visit,
                        or a list of lists of event numbers for each of the datasets passed in datasets_names.

    :param branch_selection: can be
     - None (all branches are read),
     - 'basic' (hax.config['basic_branches'] are read), or
     - a list of branches to read.

    :param kwargs: dictionary of extra arguments to pass to event_function.
                   For example: kwargs={'x': 2, 'y': 3} --> function called like: event_function(event, x=2, y=3)

    :param desc: Description used in the tqdm progressbar
    """
    if kwargs is None:
        kwargs = {}

    if not isinstance(datasets_names, (list, tuple, np.ndarray)):
        datasets_names = [datasets_names]
        if event_lists is not None:
            event_lists = [event_lists]

    for dset_i, run_id in enumerate(datasets_names):
        rootfile = open_pax_rootfile(run_id)
        # If you get "'TObject' object has no attribute 'GetEntries'" here,
        # we renamed the tree to T1 or TPax or something... or you're trying to load a Xerawdp root file!
        t = rootfile.Get('tree')

        if branch_selection == 'basic':
            branch_selection = hax.config['basic_branches']

        # Activate the desired branches
        if branch_selection:
            t.SetBranchStatus("*", 0)
            for bn in branch_selection:
                t.SetBranchStatus(bn, 1)

        try:
            if event_lists is None:
                # Visit all events
                n_events = t.GetEntries()
                source = range(n_events)
            else:
                # Visit only the desired events
                source = event_lists[dset_i]
                n_events = len(source)
            if hax.config.get('tqdm_on', True):
                source = tqdm(source,
                              desc='Run %s: %s' % (run_id, desc),
                              total=n_events)
            for event_i in source:
                t.GetEntry(event_i)
                event = t.events
                yield event_function(event, **kwargs)

        except StopEventLoop:
            rootfile.Close()
        except Exception as e:
            rootfile.Close()
            raise e


def loop_over_datasets(*args, **kwargs):
    """Execute a function over all events in the dataset(s)
    Does not return anything: use function_results_dataset or pass a class method as event_function if you want results.
    See function_results_datasets for possible options.
    """
    for _ in function_results_datasets(*args, **kwargs):
        # do nothing with the results
        pass

# For backward compatibility
loop_over_dataset = loop_over_datasets
