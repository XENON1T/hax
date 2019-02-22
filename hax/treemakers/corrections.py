import hax
import json
from hax.minitrees import TreeMaker
from pax import utils
from pax.configuration import load_configuration
from hax.treemakers.common import get_largest_indices
import numpy as np
from hax.corrections_handler import CorrectionsHandler


class Corrections(TreeMaker):
    """Applies high level corrections which are used in standard analyses.

    Provides:
    - Corrected S1 contains xyz-correction:
      - cs1: The corrected area of the main interaction's S1 using w/  correction of electric field effects
      - cs1_no_field_corr_nn_tf: The corrected S1 using w/o correction of electric field effects,
            using position from nn_tf
      - cs1_no_field_corr_nn: The corrected S1 using w/o correction of electric field effects,
            using position from nn
      - cs1_no_field_corr_tpf: The corrected S1 using w/o correction of electric field effects,
            using position from tpf
      - cs1_nn_tf: The corrected S1 using w/ correction of electric field effects,
            using position from nn_tf
      - cs1_nn: The corrected S1 using w/ correction of electric field effects,
            using position from nn
      - cs1_tpf: The corrected S1 using w/ correction of electric field effects,
            using position from tpf

    - Corrected S2 contains xy-correction and electron lifetime based on Kr83m trend:
      - cs2: The corrected area in pe of the main interaction's S2
      - cs2_top: The corrected area in pe of the main interaction's S2 from the top array.
      - cs2_bottom: The corrected area in pe of the main interaction's S2 from the bottom array.
      - cs2_nn_tf: The corrected area in pe of the main interaction's S2,
        using position from nn_tf
      - cs2_top_nn_tf: The corrected area in pe of the main interaction's S2 from the top array,
        using position from nn_tf
      - cs2_bottom_nn_tf: The corrected area in pe of the main interaction's S2 from the bottom array,
        using position from nn_tf
      - cs2_nn: The corrected area in pe of the main interaction's S2,
        using position from nn
      - cs2_top_nn: The corrected area in pe of the main interaction's S2 from the top array,
        using position from nn
      - cs2_bottom_nn: The corrected area in pe of the main interaction's S2 from the bottom array,
        using position from nn
      - cs2_tpf: The corrected area in pe of the main interaction's S2,
        using position from nn_tpf
      - cs2_top_tpf: The corrected area in pe of the main interaction's S2 from the top array,
        using position from nn_tpf
      - cs2_bottom_tpf: The corrected area in pe of the main interaction's S2 from the bottom array,
        using position from nn_tpf

    - Observed positions, not corrected with FDC maps, for both NN and TPF:
      - r_observed_tpf: the observed interaction r coordinate (using TPF).
      - x_observed_tpf: the observed interaction x coordinate (using TPF).
      - y_observed_tpf: the observed interaction y coordinate (using TPF).
      - r_observed_nn: the observed interaction r coordinate (using NN).
      - x_observed_nn: the observed interaction x coordinate (using NN).
      - y_observed_nn: the observed interaction y coordinate (using NN).
      - r_observed_nn_tf: the observed interaction r coordinate (using NN_TF).
      - x_observed_nn_tf: the observed interaction x coordinate (using NN_TF).
      - y_observed_nn_tf: the observed interaction y coordinate (using NN_TF).
      - z_observed: the observed interaction z coordinate (before the r, z correction).

    - Data-driven 3D position correction (applied to both NN and TPF observed positions):
      - r_3d_nn: the corrected interaction r coordinate (using NN).
      - x_3d_nn: the corrected interaction x coordinate (using NN).
      - y_3d_nn: the corrected interaction y coordinate (using NN).
      - z_3d_nn: the corrected interaction z coordinate (using NN).
      - r_3d_tpf: the corrected interaction r coordinate (using TPF).
      - x_3d_tpf: the corrected interaction x coordinate (using TPF).
      - y_3d_tpf: the corrected interaction y coordinate (using TPF).
      - z_3d_tpf: the corrected interaction z coordinate (using TPF).
      - r_3d_nn_tf: the corrected interaction r coordinate (using NN_TF).
      - x_3d_nn_tf: the corrected interaction x coordinate (using NN_TF).
      - y_3d_nn_tf: the corrected interaction y coordinate (using NN_TF).
      - z_3d_nn_tf: the corrected interaction z coordinate (using NN_TF).

    - Correction values for 'un-doing' single corrections:
      - s2_lifetime_correction
      - s2_xy_correction_tot_nn_tf
      - s2_xy_correction_top_nn_tf
      - s2_xy_correction_bottom_nn_tf
      - s2_xy_correction_tot_nn
      - s2_xy_correction_top_nn
      - s2_xy_correction_bottom_nn
      - s2_xy_correction_tot_tpf
      - s2_xy_correction_top_tpf
      - s2_xy_correction_bottom_tpf
      - r_correction_3d_nn_tf
      - r_correction_3d_nn
      - r_correction_3d_tpf
      - z_correction_3d_nn_tf
      - z_correction_3d_nn
      - z_correction_3d_tpf


    Notes:
    - The cs2, cs2_top and cs2_bottom variables are corrected
    for electron lifetime and x, y dependence.

    """
    __version__ = '2.1'
    extra_branches = ['peaks.area_per_channel[260]',
                      'peaks.n_saturated_per_channel[260]',
                      'peaks.s2_saturation_correction',
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

    def __init__(self):
        hax.minitrees.TreeMaker.__init__(self)
        self.extra_metadata = hax.config['corrections_definitions']
        self.corrections_handler = CorrectionsHandler()
        self.tfnn_posrec = tfnn_position_reconstruction()

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

        z_observed = interaction.z - interaction.z_correction
        result['z_observed'] = z_observed

        # Position reconstruction based on NN from TensorFlow
        # First Check for MC data, and avoid Tensor Flow if MC.
        if not self.mc_data:  # Temporary for OSG production
            # Calculate TF_NN reconstructed position
            predicted_xy_tensorflow = self.tfnn_posrec(list(s2.area_per_channel), self.run_number)

            result['x_observed_nn_tf'] = predicted_xy_tensorflow[0, 0] / 10.
            result['y_observed_nn_tf'] = predicted_xy_tensorflow[0, 1] / 10.
            result['r_observed_nn_tf'] =\
                np.sqrt(result['x_observed_nn_tf']**2 + result['y_observed_nn_tf']**2)

            # 3D FDC NN_TF
            algo = 'nn_tf'
            cvals = [result['x_observed_' + algo], result['y_observed_' + algo], z_observed]
            result['r_correction_3d_' + algo] = self.corrections_handler.get_correction_from_map(
                "fdc_3d_tfnn", self.run_number, cvals)

            result['r_3d_' + algo] = (result['r_observed_' + algo] +
                                      result['r_correction_3d_' + algo])
            result['x_3d_' + algo] = (result['x_observed_' + algo] *
                                      (result['r_3d_' + algo] / result['r_observed_' + algo]))
            result['y_3d_' + algo] = (result['y_observed_' + algo] *
                                      (result['r_3d_' + algo] / result['r_observed_' + algo]))

            if abs(z_observed) > abs(result['r_correction_3d_' + algo]):
                result['z_3d_' + algo] = -np.sqrt(z_observed ** 2 -
                                                  result['r_correction_3d_' + algo] ** 2)
            else:
                result['z_3d_' + algo] = z_observed

            result['z_correction_3d_' + algo] = result['z_3d_' + algo] - z_observed

        # Apply the 3D data driven NN_FDC, for NN positions and TPF positions
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

        # electron lifetime correction
        result['s2_lifetime_correction'] = (
            self.corrections_handler.get_electron_lifetime_correction(
                self.run_number, self.run_start, interaction.drift_time, self.mc_data))

        for algo in ['nn_tf', 'nn', 'tpf']:
            # Correct S2
            result['r_observed_' + algo] = \
                np.sqrt(result['x_observed_' + algo] ** 2 + result['y_observed_' + algo] ** 2)

            cvals = [result['x_observed_' + algo], result['y_observed_' + algo]]
            result['s2_xy_correction_tot_' + algo] = \
                (1.0 / self.corrections_handler.get_correction_from_map("s2_xy_map", self.run_number, cvals))
            result['s2_xy_correction_top_' + algo] = (1.0 /
                                                      self.corrections_handler.get_correction_from_map(
                                                          "s2_xy_map", self.run_number, cvals, map_name='map_top'))
            result['s2_xy_correction_bottom_' + algo] = (1.0 /
                                                         self.corrections_handler.get_correction_from_map(
                                                             "s2_xy_map", self.run_number,
                                                             cvals, map_name='map_bottom'))

            # Combine all the s2 corrections
            result['cs2_' + algo] = \
                s2.area * result['s2_lifetime_correction'] * result['s2_xy_correction_tot_' + algo]
            result['cs2_top_' + algo] = \
                s2.area * s2.area_fraction_top * \
                result['s2_lifetime_correction'] * result['s2_xy_correction_top_' + algo]
            result['cs2_bottom_' + algo] = \
                s2.area * (1.0 - s2.area_fraction_top) * \
                result['s2_lifetime_correction'] * result['s2_xy_correction_bottom_' + algo]

            # Correct S1
            cvals = [result['x_3d_' + algo], result['y_3d_' + algo], result['z_3d_' + algo]]
            result['s1_xyz_correction_fdc_3d_' + algo] = \
                (1 / self.corrections_handler.get_correction_from_map(
                    "s1_lce_map_nn_fdc_3d", self.run_number, cvals))
            result['cs1_no_field_corr_' + algo] = s1.area * result['s1_xyz_correction_fdc_3d_' + algo]

            # Apply corrected LCE (light collection efficiency correction to s1, including field effects)
            result['s1_xyz_true_correction_fdc_3d' + algo] = \
                (1 / self.corrections_handler.get_correction_from_map(
                    "s1_corrected_lce_map_nn_fdc_3d", self.run_number, cvals))
            result['cs1_' + algo] = s1.area * result['s1_xyz_true_correction_fdc_3d' + algo]

        # default cS1 and cS2 values
        default_algo = 'nn_tf'
        result['cs1'] = result['cs1_' + default_algo]
        result['cs2'] = result['cs2_' + default_algo]
        result['cs2_top'] = result['cs2_top_' + default_algo]
        result['cs2_bottom'] = result['cs2_bottom_' + default_algo]

        return result


class tfnn_position_reconstruction(object):
    """
    Class for loading and calculate tfnn position given area per channel
    """
    def __init__(self):
        self.corrections_handler = CorrectionsHandler()
        # We need to pull some stuff from the pax config
        self.pax_config = load_configuration("XENON1T")
        self.tpc_channels = self.pax_config['DEFAULT']['channels_in_detector']['tpc']
        self.top_channels = self.pax_config['DEFAULT']['channels_top']

        self.run_number = -1

    def __call__(self, per_channel_area, run_number):
        # per_channel_area as a list of channel area
        # OR as an array of shape (#peaks, #channels w/o bad pmt removal)
        # If we haven't loaded weights or it's another run reload nn
        if (('tfnn_weights' not in self.__dict__.keys()) or
                ('tfnn_model' not in self.__dict__.keys()) or
                (run_number != self.run_number)):
            self.load_nn(run_number)

        if isinstance(per_channel_area, list):
            per_channel_area = np.array([per_channel_area])

        # Take channel clean (top but not bad)
        per_channel_area = np.take(per_channel_area, self.clean_channels, axis=-1)
        # Divide by the sum area of each peak, preform transform due to np division law
        # Final transform due to nn predict input requires (none, #channels)
        per_channel_area = (per_channel_area.T / np.sum(per_channel_area, axis=-1)).T

        return self.loaded_nn.predict(per_channel_area)

    def load_nn(self, run_number):
        from keras.models import model_from_json

        self.tfnn_weights = self.corrections_handler.get_misc_correction(
            "tfnn_weights", run_number)
        self.tfnn_model = self.corrections_handler.get_misc_correction(
            "tfnn_model", run_number)
        self.run_number = run_number

        # Get tfnn bat pmt list from JSON file
        with open(utils.data_file_name(self.tfnn_model), 'r') as json_file_nn:
            loaded_model_json_dict = json.load(json_file_nn)
            self.list_bad_pmts = loaded_model_json_dict['badPMTList']

        self.clean_channels = list(set(self.top_channels).difference(self.list_bad_pmts))

        # Get tfnn model from JSON file
        with open(utils.data_file_name(self.tfnn_model), 'r') as json_file_nn:
            loaded_model_json = json_file_nn.read()
            self.loaded_nn = model_from_json(loaded_model_json)

        weights_file = utils.data_file_name(self.tfnn_weights)
        self.loaded_nn.load_weights(weights_file)
