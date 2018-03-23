import os
import json
import warnings

import numpy as np
import pandas as pd

try:
    import ROOT
    import root_numpy
except ImportError as e:
    warnings.warn("Error importing ROOT-related libraries: %s. "
                  "If you try to use ROOT-related functions, hax will crash!" % e)

from hax.utils import save_pickles, load_pickles


def get_format(path, treemaker=None):
    _, ext = os.path.splitext(path)
    if ext not in MINITREE_FORMATS:
        raise ValueError("Unknown minitree extension in filename %s" % path)
    return MINITREE_FORMATS[ext](path, treemaker)


class MinitreeDataFormat():
    def __init__(self, path, treemaker):
        self.path = path
        self.treemaker = treemaker

    def load_metadata(self):
        raise NotImplementedError


class PickleFormat(MinitreeDataFormat):
    def load_metadata(self):
        return load_pickles(self.path, load_first=1)[0]

    def load_data(self):
        return load_pickles(self.path)[1]

    def save_data(self, metadata, data):
        save_pickles(self.path, metadata, data)


class ROOTFormat(MinitreeDataFormat):
    def load_metadata(self):
        # This is NOT the same as paxroot.get_metadata, that's for pax ROOT files...
        minitree_f = ROOT.TFile(self.path)

        metadata_object = minitree_f.Get('metadata')

        # Corrupt file where metadata did not get saved
        if metadata_object is None:
            minitree_f.Close()
            raise RuntimeError("Metadata non-existent/corrupt file: %s" % self.path)

        minitree_metadata = json.loads(metadata_object.GetTitle())
        minitree_f.Close()
        return minitree_metadata

    def load_data(self):
        return pd.DataFrame.from_records(root_numpy.root2array(self.path).view(np.recarray))

    def save_data(self, metadata, data):
        if self.treemaker.uses_arrays:
            # Activate Joey's array saving code
            dataframe_to_root(data, self.path, treename=self.treemaker.__name__, mode='recreate')

        else:
            # Check we really aren't using arrays, otherwise we'll crash with a very uninformative message
            for branch_name in data.columns:
                if is_array_field(data, branch_name):
                    raise TypeError("Column %s is an array field, and you want to save to root. Either "
                                    "(1) use MultipleRowExtractor-based minitrees; or "
                                    "(2) add a uses_arrays=True attribute to the %s class; or "
                                    "(3) use pickle as your minitree format." % (branch_name,
                                                                                 self.treemaker.__class__.__name__))
            root_numpy.array2root(data.to_records(), self.path,
                                  treename=self.treemaker.__name__, mode='recreate')

        # Add metadata as JSON in a TNamed in the same ROOT file
        bla = ROOT.TNamed('metadata', json.dumps(metadata))
        minitree_f = ROOT.TFile(self.path, 'UPDATE')
        bla.Write()
        minitree_f.Close()


MINITREE_FORMATS = {'.root': ROOTFormat, '.pklz': PickleFormat}


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
            if length_branch_name not in dataframe.columns:
                try:
                    dataframe[length_branch_name] = np.array([len(x) for x in dataframe[branch_name]], dtype=np.int64)
                except TypeError:
                    raise TypeError('Array branch %s has at least one element that is not a list or array'
                                    % branch_name)
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
            raise TypeError('Branches must contain ints, floats, or arrays of ints or floats')
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
