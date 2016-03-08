"""Utility functions for loading and looping over a pax root file
"""
import logging
import os

from tqdm import tqdm
import ROOT
from pax.plugins.io.ROOTClass import load_event_class, load_pax_event_class_from_root, ShutUpROOT
from pax.exceptions import MaybeOldFormatException

import hax
from hax import runs
from hax.utils import find_file_in_folders


log = logging.getLogger('hax.paxroot')


def open_pax_rootfile(filename):
    """Opens pax root file filename, compiling classes/dictionaries as needed. Returns TFile object
    pax_class_version is only needed for old pax files, for pax >= 4.5.0 the class is contained inside the root file
    """
    if not os.path.exists(filename):
        raise ValueError("%s does not exist!" % filename)
    try:
        load_pax_event_class_from_root(filename)
    except MaybeOldFormatException:
        log.warning("Root file %s does not include pax event class. Normal for pax < 4.5."
                    "Falling back to event class for pax %s" % (filename, hax.config['old_pax_class_version']))
        # Load the pax class for the data format version
        load_event_class(os.path.join(hax.config['old_pax_classes_dir'],
                                      'pax_event_class_%d.cpp' % hax.config['old_pax_class_version']))
    return ROOT.TFile(filename)




# An exception you can raise to stop looping over the current dataset
class StopEventLoop(Exception):
    pass


def loop_over_datasets(datasets_names, event_function=lambda event: None, branch_selection='basic'):
    """Execute event_function(event) over all events in the dataset(s)
    Does not return anything: you have to keep track of results yourself (global vars, function attrs, classes, ...)
    branch selection: can be None (all branches are read), 'basic' (hax.config['basic_branches'] are read), or a list of branches to read.
    """
    if isinstance(datasets_names, str):
        datasets_names = [datasets_names]

    for dataset_name in datasets_names:
        # Open the file, load the tree
        # If you get "'TObject' object has no attribute 'GetEntries'" here,
        # we renamed the tree to T1 or TPax or something
        try:
            dataset = runs.datasets.loc[runs.datasets['name'] == dataset_name].iloc[0]
            filename = dataset.location
        except IndexError:
            print("Don't know a dataset named %s, trying to find it anyway..." % dataset_name)
            filename = find_file_in_folders(dataset_name + '.root', hax.config['main_data_paths'])
        if not filename:
            raise ValueError("Cannot loop over dataset %s, we don't know where it is." % dataset_name)

        rootfile = open_pax_rootfile(filename)
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
            for event_i in tqdm(range(n_events)):
                t.GetEntry(event_i)
                event = t.events
                event_function(event)
        except StopEventLoop:
            rootfile.Close()
        except Exception as e:
            rootfile.Close()
            raise e

# For backward compatibility
loop_over_dataset = loop_over_datasets