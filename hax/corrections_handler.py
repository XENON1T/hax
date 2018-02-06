import hax
from hax import runs
from pax.InterpolatingMap import InterpolatingMap
import pax.utils
import numpy as np
from scipy.interpolate import interp1d
import logging
import copy

log = logging.getLogger('hax.runs')

"""
Corrections handler class
"""


class CorrectionsHandler():
    """
    This class will hold and handle all corrections.
    It reads the ini file and loads up appropriate corrections
    files as needed.

    This is to avoid individual treemakers opening files and
    doing this logic on their own.
    """

    def __init__(self):
        """Declare data structures.
        We won't actually load any files until they're requested.
        """

        # This one for interp maps
        self.maps = {}

        # Special handling for e lifetime
        self.elifedoc = None
        self.elife_functions = {}

        # For things that aren't maps
        self.misc = {}

    def get_correction_from_map(self, correction_name, run, var, map_name='map'):
        """Get a correctrion from a correction map
        var: array with proper dimensions for map
        map_name: if you store multiple maps in one pax object
        run: integer run number
        correction_name: must match the one in hax config
        """
        correction = self.get_correction(correction_name, run, ismap=True)
        if correction is None:
            raise ValueError("Didn't find your correction %s for run %i" % (
                correction_name, run))

        # 2D case
        if len(var) == 2:
            return correction['value'].get_value(var[0], var[1], map_name=map_name)
        else:
            return correction['value'].get_value(var[0], var[1], var[2], map_name=map_name)

    def get_misc_correction(self, correction_name, run):
        """Get a non-map and non-elifetime correction
        """
        correction = self.get_correction(correction_name, run)
        if correction is None:
            raise ValueError("Didn't find your correction %s for run %i" % (
                correction_name, run))
        return correction['value']

    def get_correction(self, correction_name, run, ismap=False):
        """Get a correction from a map. var must have same dimension
        as the value expected by the map or will throw.

        Will load correction as needed
        """
        if correction_name not in self.maps and ismap:
            self.maps[correction_name] = []
        elif correction_name not in self.misc and not ismap:
            self.misc[correction_name] = []

        # Look to see if we have an existing correction covering
        iterdict = self.maps
        if not ismap:
            iterdict = self.misc
        for entry in iterdict[correction_name]:
            if 'run_min' in entry and run < entry['run_min']:
                continue
            if 'run_max' in entry and run > entry['run_max']:
                continue
            if 'value' in entry:
                return entry

        # If we make it to this point then the map hasn't been loaded yet
        # load it and return the correction value
        if correction_name not in hax.config['corrections_definitions']:
            raise ValueError("Can't find correction %s in hax.ini" % correction_name)

        for entry in hax.config['corrections_definitions'][correction_name]:
            if 'run_min' in entry and run < entry['run_min']:
                continue
            if 'run_max' in entry and run > entry['run_max']:
                continue
            if 'correction' not in entry:
                continue

            newcorr = copy.deepcopy(entry)
            if ismap:
                map_path = pax.utils.data_file_name(newcorr['correction'])
                newcorr['value'] = InterpolatingMap(map_path)
                self.maps[correction_name].append(newcorr)
            else:
                newcorr['value'] = newcorr['correction']
                self.misc[correction_name].append(newcorr)

            return newcorr

        # If we get here something is wrong and we didn't find the correction
        raise ValueError("Didn't find a correction for %s in run %i" % (correction_name, run))

    def get_electron_lifetime_correction(self, run_number, run_start, drift_time, mc_data, value='DEFAULT'):
        """Wrapper that does the exponential calculation for you
        """
        if mc_data:
            elifetime = self.get_misc_correction("mc_electron_lifetime_liquid", run_number)

        else:
            elifetime = self.get_electron_lifetime(run_start, value)

        return np.exp((drift_time / 1e3) / elifetime)

    def get_electron_lifetime(self, run_start, value='DEFAULT'):
        """Gets the electron lifetime for this run
        """
        if self.elifedoc is None:
            self.elifedoc = runs.corrections_docs['hax_electron_lifetime']

        if value not in self.elife_functions.keys():
            if value == 'DEFAULT':  # Kr83m trend
                self.elife_functions[value] = interp1d(self.elifedoc['times'],
                                                       self.elifedoc['electron_lifetimes'])

            elif value == 'alpha':
                self.elife_functions[value] = interp1d(self.elifedoc['times_alpha'],
                                                       self.elifedoc['electron_lifetimes_alpha'])

            else:
                self.elife_functions[value] = interp1d(self.elifedoc['times'],
                                                       self.elifedoc[value])

        ts = ((run_start - np.datetime64('1970-01-01T00:00:00Z')) /
              np.timedelta64(1, 's'))

        return self.elife_functions[value](ts)
