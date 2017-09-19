import hax
from hax.minitrees import TreeMaker
from hax.treemakers.common import get_largest_indices
from hax import runs
from pax.InterpolatingMap import InterpolatingMap
import pax.utils
import numpy as np
from scipy.interpolate import interp1d


class Corrections(TreeMaker):
    """Applies high level corrections which are used in standard analyses.

    Provides:
    - Position correction (please note this will change soon!):
      - r_observed: the observed interaction r coordinate (before the r, z correction).
      - x_observed: the observed interaction x coordinate (before the r, z correction).
      - y_observed: the observed interaction y coordinate (before the r, z correction).
      - z_observed: the observed interaction z coordinate (before the r, z correction).
      - r: the corrected interaction r coordinate
      - x: the corrected interaction x coordinate
      - y: the corrected interaction y coordinate
      - z: the corrected interaction z coordinate

    - Correction values for 'un-doing' single corrections:
      - s2_xy_correction_tot
      - s2_xy_correction_top
      - s2_xy_correction_bottom
      - s2_lifetime_correction
      - r_correction (r_observed + r_correction = r)
    - Corrected S2 contains xy-correction and electron lifetime:
      - cs2: The corrected area in pe of the main interaction's S2
      - cs2_top: The corrected area in pe of the main interaction's S2 from the top array.
      - cs2_bottom: The corrected area in pe of the main interaction's S2 from the bottom array.

    Notes:
    - The cs2, cs2_top and cs2_bottom variables are corrected
    for electron lifetime and x, y dependence.

    """
    __version__ = '1.4'
    extra_branches = ['peaks.s2_saturation_correction',
                      'interactions.s2_lifetime_correction',
                      'peaks.area_fraction_top',
                      'peaks.area',
                      'interactions.x',
                      'interactions.y',
                      'interactions.z',
                      'interactions.r_correction',
                      'interactions.z_correction',
                      'interactions.drift_time',
                      'start_time']

    extra_metadata = hax.config['corrections_definitions']

    # Electron Lifetime: hopefully doc was pulled in hax.init.
    # Otherwise get it here at significantly higher DB cost
    try:
        elife_correction_doc = runs.corrections_docs['hax_electron_lifetime']
        extra_metadata['electron_lifetime_version'] = elife_correction_doc['version']
        elife_interpolation = interp1d(elife_correction_doc['times'],
                                       elife_correction_doc['electron_lifetimes'])
    except Exception as e:
        elife_interpolation = None
        print("No electron lifetime document found. Continuing without.")
        print(e)

    loaded_xy_map_name = None
    loaded_fdc_map_name = None
    loaded_lce_map_name = None
    xy_map = None
    fdc_map = None
    lce_map = None

    def get_correction_filename(self, correction_name):
        """Return the file to use for a correction"""
        if ('corrections_definitions' not in hax.config or
            correction_name not in hax.config['corrections_definitions']):
            return None

        for entry in hax.config['corrections_definitions'][correction_name]:
            if 'run_min' not in entry or self.run_number < entry['run_min']:
                continue
            if 'run_max' not in entry or self.run_number <= entry['run_max']:
                if 'file_name' in entry:
                    return entry['file_name']
        return None

    def extract_data(self, event):
        result = dict()

        # If there are no interactions cannot do anything
        if not len(event.interactions):
            return result

        # Workaround for blinding cut. S2 area and largest_other_s2 needed.
        interaction = event.interactions[0]
        s2 = event.peaks[interaction.s2]
        s1 = event.peaks[interaction.s1]
        largest_other_indices = get_largest_indices(
            event.peaks, exclude_indices=(interaction.s1, interaction.s2))
        largest_area_of_type = {ptype: event.peaks[i].area
                                for ptype, i in largest_other_indices.items()}
        result['largest_other_s2'] = largest_area_of_type.get('s2', 0)
        result['s2'] = s2.area

        # Check that the correct S2 map is loaded and change if not
        wanted_map_name = self.get_correction_filename("s2_xy_map")
        if self.loaded_xy_map_name != wanted_map_name:
            map_path = pax.utils.data_file_name(wanted_map_name)
            self.xy_map = InterpolatingMap(map_path)
            self.loaded_xy_map_name = wanted_map_name

        # Load the FDC map
        wanted_fdc_map_name = self.get_correction_filename("fdc")
        if wanted_fdc_map_name != self.loaded_fdc_map_name:
            fdc_map_path = pax.utils.data_file_name(wanted_fdc_map_name)
            self.fdc_map = InterpolatingMap(fdc_map_path)
            self.loaded_fdc_map_name = wanted_fdc_map_name

        # Load the LCE map
        wanted_lce_map_name = self.get_correction_filename("s1_lce_map")
        if wanted_lce_map_name != self.loaded_lce_map_name:
            lce_map_path = pax.utils.data_file_name(wanted_lce_map_name)
            self.lce_map = InterpolatingMap(lce_map_path)
            self.loaded_lce_map_name = wanted_lce_map_name

        # Need the observed ('uncorrected') position; pax gives
        # corrected positions (where the interaction happens)
        interaction_r = np.sqrt(interaction.x ** 2 + interaction.y ** 2)
        r_observed = interaction_r - interaction.r_correction
        z_observed = interaction.z - interaction.z_correction
        phi = np.arctan2(y_observed, x_observed) 
        x_observed = r_observed * np.cos(phi)
        y_observed = r_observed * np.sin(phi)
        result['r_correction_pax'] = interaction.r_correction

        result['s2_xy_correction_tot'] = (1.0 /
                                          self.xy_map.get_value(x_observed, y_observed))
        result['s2_xy_correction_top'] = (1.0 /
                                          self.xy_map.get_value(
                                              x_observed, y_observed, map_name='map_top'))
        result['s2_xy_correction_bottom'] = (1.0 /
                                             self.xy_map.get_value(
                                                 x_observed, y_observed, map_name='map_bottom'))

        # include electron lifetime correction
        if self.elife_interpolation is not None:
            # Ugh, numpy time types...
            ts = ((self.run_start - np.datetime64('1970-01-01T00:00:00Z')) /
                  np.timedelta64(1, 's'))
            result['ts'] = ts
            self.electron_lifetime = self.elife_interpolation(ts)
            result['s2_lifetime_correction'] = np.exp((interaction.drift_time/1e3) /
                                                      self.electron_lifetime)
        else:
            result['s2_lifetime_correction'] = 1.

        # Combine all the s2 corrections
        s2_correction = (result['s2_lifetime_correction'] *
                         result['s2_xy_correction_tot'])
        s2_top_correction = (result['s2_lifetime_correction'] *
                             result['s2_xy_correction_top'])
        s2_bottom_correction = (result['s2_lifetime_correction'] *
                                result['s2_xy_correction_bottom'])

        result['r_observed'] = r_observed
        result['z_observed'] = z_observed
        result['x_observed'] = x_observed
        result['y_observed'] = y_observed

        result['cs2'] = s2.area * s2_correction
        result['cs2_top'] = s2.area * s2.area_fraction_top * s2_top_correction
        result['cs2_bottom'] = s2.area * (1.0 - s2.area_fraction_top) * s2_bottom_correction

        # Apply FDC (field distortion correction to position)
        result['r_correction'] = self.fdc_map.get_value(r_observed, z_observed, map_name='to_true_r')
        result['z_correction'] = self.fdc_map.get_value(r_observed, z_observed, map_name='to_true_z')

        result['r'] = r_observed + result['r_correction']
        result['x'] = result['r'] * np.cos(phi)
        result['y'] = result['r'] * np.sin(phi)
        result['z'] = z_observed + result['z_correction']

        # Apply LCE (light collection efficiency correction to s1)
        result['s1_xyz_correction'] = 1 / self.lce_map.get_value(result['x'], result['y'], result['z'])

        result['cs1'] = s1.area * result['s1_xyz_correction']

        return result
