"""Extract peak or hit info from processed root file
"""
import warnings

import numpy as np
import hax


def root_to_numpy(base_object, field_name, attributes):
    """Convert objects stored in base_object.field_name to numpy array
    Will query attributes for each of the objects in base_object.field_name
    No, root_numpy does not do this for you, that's for trees...
    """
    objects_to_convert = getattr(base_object, field_name)
    if not len(objects_to_convert):
        return None
    return np.array([tuple([getattr(p, pa) for pa in attributes]) for p in objects_to_convert])


def build_cut_string(cut_list, obj):
    '''
    Build a string of cuts that can be applied using eval() function.
    '''
    # If no cut is specified, always pass cut
    if len(cut_list) == 0:
        return 'True'
    # Check if user entered range_50p_area, since this won't work
    for cut in cut_list:
        if cut[:14] == 'range_50p_area':
            raise ValueError('You cannot use range_50p_area in your cut, use range_area_decile[5] instead!')
    cut_string = '('
    for cut in cut_list[:-1]:
        cut_string += obj + '.' + cut + ') & ('
    cut_string += obj + '.' + cut_list[-1] + ')'
    return cut_string


def make_branch_selection(level, event_fields, peak_fields, added_branches):
    """Make the list of branches that have to be selected.
    """
    branch_selection_events = event_fields
    branch_selection_peaks = ['peaks.' + field for field in peak_fields]

    # For hits, just select the whole hit branch.
    # Unfortunately, specifically setting one variable does not seem to work.
    branch_selection_hits = ['peaks.hits*']
    if level == 'hit':
        branch_selection = branch_selection_events + branch_selection_peaks + branch_selection_hits + added_branches
    if level == 'peak':
        branch_selection = branch_selection_events + branch_selection_peaks + added_branches
    # Hack to enable range_50p_area
    branch_selection = [field.replace('peaks.range_50p_area', 'peaks.range_area_decile*') for field in branch_selection]
    return branch_selection


def make_named_array(array, field_names):
    """Make a named array from a numpy array.
    """
    import pandas as pd
    df = pd.DataFrame(array, columns=field_names)
    array = df.to_records()
    return array


class DataExtractor():
    """This class is meant for extracting properties that are *not* on the event level, such as peak or hit properties.
    For more information, check the docs of DataExtractor.get_data().
    """

    def __init__(self):
        # Initialize empty data list
        warnings.warn(
            "DataExtractor is deprecated, please switch to multi-row minitrees instead.",
            DeprecationWarning)
        self.data = []

    def loop_body(self, event):
        """Function that extracts data from each event and adds array with that data to the data list.
        """
        # Check if event passes event cut
        if eval(self.event_cut_string):
            event_entry = np.array([getattr(event, field) for field in self.event_fields])

            for peak in event.peaks:
                # Check if peak passes the cut
                if eval(self.peak_cut_string):
                    # Get peak information, which is stored in _temp_data
                    _temp_data = []
                    for field in self.peak_fields:
                        if field == 'range_50p_area':
                            _x = list(peak.range_area_decile)[5]
                        elif field in ('x', 'y'):
                            # In case of x and y need to get position from reconstructed_positions
                            for rp in peak.reconstructed_positions:
                                if rp.algorithm == 'PosRecTopPatternFit':
                                    _x = getattr(rp, field)
                                    break
                            else:
                                _x = float('nan')
                        elif field[-1] == ']':
                            # This means that the parameter is a list element.
                            # We need a slightly different approach
                            parsed_field = field.split(sep='[')
                            field_list_name = parsed_field[0]
                            field_number = int(parsed_field[1][:-1])
                            _list = getattr(peak, field_list_name)
                            _x = list(_list)[field_number]
                        else:
                            # Default case for 'normal'  peak variable
                            _x = getattr(peak, field)
                        _temp_data.append(_x)
                    peak_entry = np.array(_temp_data)
                    if self.level == 'hit':
                        # Extract hit info. Numpy array with n_hit_channel entries x number of hit properties
                        hit_entry = root_to_numpy(
                            peak, 'hits', self.hit_fields)
                        if hit_entry is None:
                            # If there is no hit data: crash loudly
                            raise ValueError("Unable to read hit info. "
                                             "Is it in the root file? (Try: peak_cuts = ['type == 's1''])")

                        # Append all properties. First event properties, then
                        # peak, then hits
                        entry = np.c_[np.zeros((len(hit_entry), len(event_entry) + len(peak_entry))), hit_entry]
                        entry[:, 0:len(event_entry)] = event_entry
                        entry[:, len(event_entry):(
                            len(peak_entry) + len(event_entry))] = peak_entry
                    elif self.level == 'peak':
                        # Extra brackets needed for concatenation in the end, else we'll get flat array
                        entry = np.c_[[event_entry], [peak_entry]]
                    else:
                        # We should actually never reach this since the checking has been done before
                        raise ValueError(
                            "Enter either 'peak' of 'hit' for level!")
                    self.data.append(entry)
        # Check if user-defined event number limit is reached
        if event.event_number >= self.stop_after:
            print("User-defined limit of %d events reached, stopping..." % self.stop_after)
            raise hax.paxroot.StopEventLoop
        return None

    def get_data(self, dataset, level='peak', event_fields=['event_number'],
                 peak_fields=['area', 'hit_time_std'], hit_fields=[], event_cuts=[],
                 peak_cuts=[], stop_after=np.inf, added_branches=[]):
        """Extract peak or hit data from a dataset.
        Peak or hit can be toggled by specifying level = 'peak' or level = 'hit'.
        Example useage:
            d = DataExtractor.get_data(dataset=run_name,level='peak',event_fields = ['event_number'],
                peak_fields=['area'],event_cuts=['event_number > 5', 'event_number < 10'],
                peak_cuts=['area > 100', 'type = "s1"'],stop_after=10000,added_branches= ['peak.type'])
        """
        # Sanity checking
        if (level != 'peak') and (level != 'hit'):
            raise SyntaxError("Enter either 'peak' of 'hit' for level!")
        if (hit_fields != []) and (level == 'peak'):
            print("Warning: You set hit properties, but your input will be ignored since you specified peak level!")

        branch_selection = make_branch_selection(level, event_fields, peak_fields, added_branches)
        self.event_cut_string = build_cut_string(event_cuts, 'event')
        self.peak_cut_string = build_cut_string(peak_cuts, 'peak')
        self.event_fields = event_fields
        self.peak_fields = peak_fields
        self.hit_fields = hit_fields
        self.stop_after = stop_after
        self.level = level

        hax.paxroot.loop_over_dataset(dataset, self.loop_body, branch_selection=branch_selection)

        # Now reshape data
        # list of arrays -> one array -> named array
        self.data = np.concatenate(self.data)
        # Build list of strings with field names.
        if level == 'hit':
            # For the hit level, we have to be careful. For example, 'area' can be peak or hit area.
            # How to solve this? Well, just add hit_ or peak_ before the property
            field_names = (event_fields + ['peak_' + field for field in peak_fields] +
                           ['hit_' + field for field in hit_fields])
        if level == 'peak':
            # For peak level no such problem exists (yet) so just keep normal names
            field_names = event_fields + peak_fields
        self.data = make_named_array(self.data, field_names)

        return self.data
