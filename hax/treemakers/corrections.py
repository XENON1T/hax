import hax
from hax.minitrees import TreeMaker
from hax.treemakers.common import get_largest_indices
import numpy as np
from hax.corrections_handler import CorrectionsHandler


class Corrections(TreeMaker):
    """Applies high level corrections which are used in standard analyses.

    Provides:
    - Corrected S1 contains xyz-correction:
      - cs1: The corrected area of the main interaction's S1 using NN 3D FDC after correction of electric field effects
      - cs1_no_field_corr: The corrected area in pe of the main interaction's S1 using NN 3D FDC (no field effects corrected)
      - cs1_tpf_2dfdc: Same as cs1_no_field_corr but for TPF 2D FDC

    - Corrected S2 contains xy-correction and electron lifetime:
      - cs2: The corrected area in pe of the main interaction's S2
      - cs2_top: The corrected area in pe of the main interaction's S2 from the top array.
      - cs2_bottom: The corrected area in pe of the main interaction's S2 from the bottom array.

    - Observed positions, not corrected with FDC maps, for both NN and TPF:
      - r_observed_tpf: the observed interaction r coordinate (using TPF).
      - x_observed_tpf: the observed interaction x coordinate (using TPF).
      - y_observed_tpf: the observed interaction y coordinate (using TPF).
      - r_observed_nn: the observed interaction r coordinate (using NN).
      - x_observed_nn: the observed interaction x coordinate (using NN).
      - y_observed_nn: the observed interaction y coordinate (using NN).
      - z_observed: the observed interaction z coordinate (before the r, z correction).

    - Position correction (based on TPF, (old) 2D FDC map):
      - r: the corrected interaction r coordinate
      - x: the corrected interaction x coordinate
      - y: the corrected interaction y coordinate
      - z: the corrected interaction z coordinate

    - Data-driven 3D position correction (applied to both NN and TPF observed positions):
      - r_3d_nn: the corrected interaction r coordinate (using NN).
      - x_3d_nn: the corrected interaction x coordinate (using NN).
      - y_3d_nn: the corrected interaction y coordinate (using NN).
      - z_3d_nn: the corrected interaction z coordinate (using NN).
      - r_3d_tpf: the corrected interaction r coordinate (using TPF).
      - x_3d_tpf: the corrected interaction x coordinate (using TPF).
      - y_3d_tpf: the corrected interaction y coordinate (using TPF).
      - z_3d_tpf: the corrected interaction z coordinate (using TPF).

    - Correction values for 'un-doing' single corrections:
      - s1_xyz_correction_tpf_fdc_2d
      - s1_xyz_correction_nn_fdc_3d
      - s2_xy_correction_tot
      - s2_xy_correction_top
      - s2_xy_correction_bottom
      - s2_lifetime_correction
      - r_correction_3d_nn
      - r_correction_3d_tpf
      - r_correction_2d
      - z_correction_3d_nn
      - z_correction_3d_tpf
      - z_correction_2d

    Notes:
    - The cs2, cs2_top and cs2_bottom variables are corrected
    for electron lifetime and x, y dependence.

    """
    __version__ = '1.9'
    extra_branches = ['peaks.s2_saturation_correction',
                      'interactions.s2_lifetime_correction',
                      'peaks.area_fraction_top',
                      'peaks.area',
                      'peaks.reconstructed_positions*',
                      'interactions.x',
                      'interactions.y',
                      'interactions.z',
                      'interactions.r_correction',
                      'interactions.z_correction',
                      'interactions.drift_time',
                      'start_time']

    extra_metadata = hax.config['corrections_definitions']
    corrections_handler = CorrectionsHandler()

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

        # Need the observed ('uncorrected') position.
        # pax Interaction positions are corrected so lookup the
        # uncorrected positions in the ReconstructedPosition objects
        for rp in s2.reconstructed_positions:
            if rp.algorithm == 'PosRecNeuralNet':
                result['x_observed_nn'] = rp.x
                result['y_observed_nn'] = rp.y
                result['r_observed_nn'] = np.sqrt(rp.x ** 2 + rp.y ** 2)
            if rp.algorithm == 'PosRecTopPatternFit':
                result['x_observed_tpf'] = rp.x
                result['y_observed_tpf'] = rp.y
                result['r_observed_tpf'] = np.sqrt(rp.x ** 2 + rp.y ** 2)
                r_observed = np.sqrt(rp.x ** 2 + rp.y ** 2)
                result['r_observed'] = r_observed
                x_observed = rp.x
                y_observed = rp.y

        z_observed = interaction.z - interaction.z_correction
        result['z_observed'] = z_observed

        # Correct S2
        cvals = [x_observed, y_observed]
        result['s2_xy_correction_tot'] = (1.0 /
                                          self.corrections_handler.get_correction_from_map(
                                              "s2_xy_map", self.run_number, cvals))
        result['s2_xy_correction_top'] = (1.0 /
                                          self.corrections_handler.get_correction_from_map(
                                              "s2_xy_map", self.run_number, cvals, map_name='map_top'))
        result['s2_xy_correction_bottom'] = (1.0 /
                                             self.corrections_handler.get_correction_from_map(
                                                 "s2_xy_map", self.run_number, cvals, map_name='map_bottom'))

        # include electron lifetime correction
        if self.mc_data:
            wanted_electron_lifetime = self.corrections_handler.get_misc_correction(
                "mc_electron_lifetime_liquid", self.run_number)
            result['s2_lifetime_correction'] = np.exp((interaction.drift_time / 1e3) /
                                                      wanted_electron_lifetime)

        else:
            try:
                result['s2_lifetime_correction'] = (
                    self.corrections_handler.get_electron_lifetime_correction(
                        self.run_start, interaction.drift_time))
            except Exception as e:
                print(e)
                result['s2_lifetime_correction'] = 1.

        # Combine all the s2 corrections
        s2_correction = (result['s2_lifetime_correction'] *
                         result['s2_xy_correction_tot'])
        s2_top_correction = (result['s2_lifetime_correction'] *
                             result['s2_xy_correction_top'])
        s2_bottom_correction = (result['s2_lifetime_correction'] *
                                result['s2_xy_correction_bottom'])

        result['cs2'] = s2.area * s2_correction
        result['cs2_top'] = s2.area * s2.area_fraction_top * s2_top_correction
        result['cs2_bottom'] = s2.area * (1.0 - s2.area_fraction_top) * s2_bottom_correction

        # FDC
        # Apply the (old) 2D FDC (field distortion correction to position)
        # Because we have different 2D correction maps for different runs we need
        # to reapply the 2D FDC here (if not we could simply take the Interaction positions
        # which have already the 2D FDC applied).
        result['r_correction_2d'] = self.corrections_handler.get_correction_from_map(
            "fdc_2d", self.run_number, [r_observed, z_observed], map_name='to_true_r')
        result['z_correction_2d'] = self.corrections_handler.get_correction_from_map(
            "fdc_2d", self.run_number, [r_observed, z_observed], map_name='to_true_z')

        result['r'] = r_observed + result['r_correction_2d']
        result['x'] = (result['r'] / result['r_observed']) * x_observed
        result['y'] = (result['r'] / result['r_observed']) * y_observed
        result['z'] = z_observed + result['z_correction_2d']

        # FDC
        # Apply the (new) 3D data driven FDC, using NN positions and TPF positions
        for algo in ['nn', 'tpf']:
            cvals = [result['x_observed_' + algo], result['y_observed_' + algo], z_observed]
            result['r_correction_3d_' + algo] = self.corrections_handler.get_correction_from_map(
                "fdc_3d", self.run_number, cvals)

            result['r_3d_' + algo] = result['r_observed_' + algo] + result['r_correction_3d_' + algo]
            result['x_3d_' + algo] =\
                result['x_observed_' + algo] * (result['r_3d_' + algo] / result['r_observed_' + algo])
            result['y_3d_' + algo] =\
                result['y_observed_' + algo] * (result['r_3d_' + algo] / result['r_observed_' + algo])

            if abs(z_observed) > abs(result['r_correction_3d_' + algo]):
                result['z_3d_' + algo] = -np.sqrt(z_observed ** 2 -
                                                  result['r_correction_3d_' + algo] ** 2)
            else:
                result['z_3d_' + algo] = z_observed

            result['z_correction_3d_' + algo] = result['z_3d_' + algo] - z_observed

        # Apply LCE (light collection efficiency correction to s1 without field effects considered)
        cvals = [result['x'], result['y'], result['z']]
        result['s1_xyz_correction_tpf_fdc_2d'] = (
            1 / self.corrections_handler.get_correction_from_map(
                "s1_lce_map_tpf_fdc_2d", self.run_number, cvals)
        )
        result['cs1_tpf_2dfdc'] = s1.area * result['s1_xyz_correction_tpf_fdc_2d']

        cvals = [result['x_3d_nn'], result['y_3d_nn'], result['z_3d_nn']]
        result['s1_xyz_correction_nn_fdc_3d'] = (
            1 / self.corrections_handler.get_correction_from_map(
                "s1_lce_map_nn_fdc_3d", self.run_number, cvals)
        )
        result['cs1_no_field_corr'] = s1.area * result['s1_xyz_correction_nn_fdc_3d']

        # Apply corrected LCE (light collection efficiency correction to s1, including field effects)
        result['s1_xyz_true_correction_nn_fdc_3d'] = (
            1 / self.corrections_handler.get_correction_from_map(
                "s1_corrected_lce_map_nn_fdc_3d", self.run_number, cvals)
        )
        result['cs1'] = s1.area * result['s1_xyz_true_correction_nn_fdc_3d']

        return result
