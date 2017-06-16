from hax.minitrees import TreeMaker
from pax.InterpolatingMap import InterpolatingMap
import pax.utils
import numpy as np

class Corrections(TreeMaker):
    """Applies high level corrections which are used in standard analyses.

    Provides:
    - Position correction:
      - r_observed: the observed interaction r coordinate (before the r, z correction).
      - x_observed: the observed interaction x coordinate (before the r, z correction).
      - y_observed: the observed interaction y coordinate (before the r, z correction).
      - z_observed: the observed interaction z coordinate (before the r, z correction).
    - S2 x, y correction:
      - cs2: The corrected area in pe of the main interaction's S2
      - cs2_top: The corrected area in pe of the main interaction's S2 from the top array.
      - cs2_bottom: The corrected area in pe of the main interaction's S2 from the bottom array.

    Notes:
    - The cs2, cs2_top and cs2_bottom variables are corrected
    for electron lifetime and x, y dependence.

    """
    __version__ = '1.0'
    extra_branches = ['peaks.s2_saturation_correction',
                      'interactions.s2_lifetime_correction',
                      'peaks.area_fraction_top',
                      'peaks.area',
                      'interactions.x',
                      'interactions.y',
                      'interactions.z',
                      'interactions.r_correction',
                      'interactions.z_correction']

    def get_s2_map_name(self):
        """Return the name of the S2 map file to use for this run."""
        if self.run_number < 6386:
            return 's2_xy_XENON1T_24Feb2017.json'
        else:
            return 's2_xy_map_v2.0.json'

    xy_map = None
    loaded_map_name = None

    def extract_data(self, event):
        result = dict()

        # If there are no interactions cannot do anything
        if not len(event.interactions):
            return result

        interaction = event.interactions[0]
        s2 = event.peaks[interaction.s2]

        # Check that the correct S2 map is loaded and change if not
        wanted_map_name = self.get_s2_map_name()
        if self.loaded_map_name != wanted_map_name:
            map_path = pax.utils.data_file_name(wanted_map_name)
            self.xy_map = InterpolatingMap(map_path)
            self.loaded_map_name = wanted_map_name

        # Need the observed ('uncorrected') position; pax gives
        # corrected positions (where the interaction happens)
        interaction_r = np.sqrt(interaction.x ** 2 + interaction.y ** 2)
        r_observed = interaction_r - interaction.r_correction
        z_observed = interaction.z - interaction.z_correction
        x_observed = (r_observed / interaction_r) * interaction.x
        y_observed = (r_observed / interaction_r) * interaction.y

        s2_spatial_correction = 1.0 / self.xy_map.get_value(x_observed, y_observed)
        s2_top_spatial_correction = 1.0 / self.xy_map.get_value(x_observed, y_observed, map_name='map_top')
        s2_bottom_spatial_correction = 1.0 / self.xy_map.get_value(x_observed, y_observed, map_name='map_bottom')

        s2_non_spatial_correction = interaction.s2_lifetime_correction
        s2_correction = s2_non_spatial_correction * s2_spatial_correction
        s2_top_correction = s2_non_spatial_correction * s2_top_spatial_correction
        s2_bottom_correction = s2_non_spatial_correction * s2_bottom_spatial_correction

        result['r_observed'] = r_observed
        result['z_observed'] = z_observed
        result['x_observed'] = x_observed
        result['y_observed'] = y_observed

        result['cs2'] = s2.area * s2_correction
        result['cs2_top'] = s2.area * s2.area_fraction_top * s2_top_correction
        result['cs2_bottom'] = s2.area * (1.0 - s2.area_fraction_top) * s2_bottom_correction

        return result
