import hax
import numpy as np
import json
from hax.minitrees import TreeMaker
from pax.PatternFitter import PatternFitter
from pax.configuration import load_configuration
from pax import utils
from pax import exceptions
from pax.plugins.interaction_processing.S1AreaFractionTopProbability import s1_area_fraction_top_probability
from keras.models import model_from_json
from hax.corrections_handler import CorrectionsHandler


class PositionReconstruction(TreeMaker):
    """Stores position-reconstruction-related variables.

    Provides:
       - s1_pattern_fit_hax: S1 pattern likelihood computed with corrected
                             position and areas
       - s1_pattern_fit_hits_hax: S1 pattern likelihood computed with corrected
                                  position and hits
       - s1_pattern_fit_bottom_hax: S1 pattern likelihood computed with corrected
                                    position and bottom array area
       - s1_pattern_fit_bottom_hits_hax: S1 pattern likelihood computed with corrected
                                         position and bottom array hits

       - s1_area_fraction_top_probability_hax: S1 AFT p-value computed with corrected position
       - s1_area_fraction_top_probability_nothresh: computed using area below S1=10 (instead of hits)
       - s1_area_fraction_top_binomial: Binomial probability for given S1 AFT
       - s1_area_fraction_top_binomial_nothresh: Same except using area below S1=10

       - x_observed_nn_tf: TensorFlow NN reconstructed x position
       - y_observed_nn_tf: TensorFlow NN reconstructed y position
       - r_observed_nn_tf: TensorFlow NN reconstructed r position

       - r_3d_nn_tf: the corrected interaction r coordinate (data-driven 3d fdc)
       - x_3d_nn_tf: the corrected interaction x coordinate (data-driven 3d fdc)
       - y_3d_nn_tf: the corrected interaction y coordinate (data-driven 3d fdc)
       - z_3d_nn_tf: the corrected interaction z coordinate (data-driven 3d fdc)
       - r_correction_3d_nn_tf: r_3d_nn_tf - r_observed_nn_tf
       - z_correction_3d_nn_tf: z_3d_nn_tf - z_observed

       - s1_area_upper_injection_fraction: s1 area fraction near Rn220 injection points (near PMT 131)
       - s1_area_lower_injection_fraction: s1 area fraction near Rn220 injection points (near PMT 243)
    """
    __version__ = '0.19'
    extra_branches = ['peaks.area_per_channel[260]',
                      'peaks.hits_per_channel[260]',
                      'peaks.n_saturated_per_channel[260]',
                      'interactions.x', 'interactions.y', 'interactions.z']

    def __init__(self):

        hax.minitrees.TreeMaker.__init__(self)
        self.extra_metadata = hax.config['corrections_definitions']
        self.corrections_handler = CorrectionsHandler()

        # We need to pull some stuff from the pax config
        self.pax_config = load_configuration("XENON1T")
        self.tpc_channels = self.pax_config['DEFAULT']['channels_in_detector']['tpc']
        self.confused_s1_channels = []
        self.statistic = (
            self.pax_config['BuildInteractions.BasicInteractionProperties']['s1_pattern_statistic']
        )
        qes = np.array(self.pax_config['DEFAULT']['quantum_efficiencies'])

        self.pattern_fitter = PatternFitter(
            filename=utils.data_file_name(self.pax_config['WaveformSimulator']['s1_patterns_file']),
            zoom_factor=self.pax_config['WaveformSimulator'].get('s1_patterns_zoom_factor', 1),
            adjust_to_qe=qes[self.tpc_channels],
            default_errors=(self.pax_config['DEFAULT']['relative_qe_error'] +
                            self.pax_config['DEFAULT']['relative_gain_error'])
        )

        # Threshold for s1_aft probability calculation
        self.low_pe_threshold = 10

        self.ntop_pmts = len(self.pax_config['DEFAULT']['channels_top'])

        # Declare nn stuff
        self.tfnn_weights = None
        self.tfnn_model = None
        self.loaded_nn = None

    def load_nn(self):
        """For loading NN files"""

        # If we already loaded it up then skip
        if ((self.tfnn_weights == self.corrections_handler.get_misc_correction(
                "tfnn_weights", self.run_number)) and
            (self.tfnn_model == self.corrections_handler.get_misc_correction(
                "tfnn_model", self.run_number))):
            return

        self.tfnn_weights = self.corrections_handler.get_misc_correction(
            "tfnn_weights", self.run_number)
        self.tfnn_model = self.corrections_handler.get_misc_correction(
            "tfnn_model", self.run_number)

        json_file_nn = open(utils.data_file_name(self.tfnn_model), 'r')
        loaded_model_json = json_file_nn.read()
        self.loaded_nn = model_from_json(loaded_model_json)
        json_file_nn.close()

        # Get bad PMT List in JSON file:
        json_file_nn = open(utils.data_file_name(self.tfnn_model), 'r')
        loaded_model_json_dict = json.load(json_file_nn)
        self.list_bad_pmts = loaded_model_json_dict['badPMTList']
        json_file_nn.close()

        weights_file = utils.data_file_name(self.tfnn_weights)
        self.loaded_nn.load_weights(weights_file)

    def get_data(self, dataset, event_list=None):
        # If we do switch to new NN later get rid of this stuff and directly use those positions!
        data, _ = hax.minitrees.load_single_dataset(dataset, ['Corrections', 'Fundamentals'])
        self.x = data.x_3d_nn.values
        self.y = data.y_3d_nn.values
        self.z = data.z_3d_nn.values

        self.indices = list(data.event_number.values)

        return hax.minitrees.TreeMaker.get_data(self, dataset, event_list)

    def extract_data(self, event):

        event_data = {
            "s1_pattern_fit_hax": None,
            "s1_pattern_fit_hits_hax": None,
            "s1_pattern_fit_bottom_hax": None,
            "s1_pattern_fit_bottom_hits_hax": None,
            "s1_area_fraction_top_probability_hax": None,
            "s1_area_fraction_top_probability_nothresh": None,
            "s1_area_fraction_top_binomial": None,
            "s1_area_fraction_top_binomial_nothresh": None,
            "x_observed_nn_tf": None,
            "y_observed_nn_tf": None,
            "r_observed_nn_tf": None,
            "x_3d_nn_tf": None,
            "y_3d_nn_tf": None,
            "r_3d_nn_tf": None,
            "z_3d_nn_tf": None,
            "r_correction_3d_nn_tf": None,
            "z_correction_3d_nn_tf": None,
            "s1_area_upper_injection_fraction": None,
            "s1_area_lower_injection_fraction": None,
        }

        # We first need the positions. This minitree is only valid when loading
        # Corrections since you need that to get the corrected positions
        if not len(event.interactions):
            return event_data

        # Check that correct NN is loaded and change if not
        self.load_nn()

        event_num = event.event_number

        try:
            event_index = self.indices.index(event_num)
        except Exception:
            return event_data

        interaction = event.interactions[0]
        s1 = event.peaks[interaction.s1]
        s2 = event.peaks[interaction.s2]

        # Position reconstruction based on NN from TensorFlow
        s2apc = np.array(list(s2.area_per_channel))
        s2apc_clean = []

        for ipmt, s2_t in enumerate(s2apc):
            if ipmt not in self.list_bad_pmts and ipmt < self.ntop_pmts:
                s2apc_clean.append(s2_t)

        s2apc_clean = np.asarray(s2apc_clean)
        s2apc_clean_norm = s2apc_clean / s2apc_clean.sum()
        s2apc_clean_norm = s2apc_clean_norm.reshape(1, len(s2apc_clean_norm))

        predicted_xy_tensorflow = self.loaded_nn.predict(s2apc_clean_norm)
        event_data['x_observed_nn_tf'] = predicted_xy_tensorflow[0, 0] / 10.
        event_data['y_observed_nn_tf'] = predicted_xy_tensorflow[0, 1] / 10.
        event_data['r_observed_nn_tf'] =\
            np.sqrt(event_data['x_observed_nn_tf']**2 + event_data['y_observed_nn_tf']**2)

        # 3D FDC
        algo = 'nn_tf'
        z_observed = interaction.z - interaction.z_correction
        cvals = [event_data['x_observed_' + algo], event_data['y_observed_' + algo], z_observed]
        event_data['r_correction_3d_' + algo] = self.corrections_handler.get_correction_from_map(
            "fdc_3d_tfnn", self.run_number, cvals)

        event_data['r_3d_' + algo] = event_data['r_observed_' + algo] + event_data['r_correction_3d_' + algo]
        event_data['x_3d_' + algo] =\
            event_data['x_observed_' + algo] * (event_data['r_3d_' + algo] / event_data['r_observed_' + algo])
        event_data['y_3d_' + algo] =\
            event_data['y_observed_' + algo] * (event_data['r_3d_' + algo] / event_data['r_observed_' + algo])

        if abs(z_observed) > abs(event_data['r_correction_3d_' + algo]):
            event_data['z_3d_' + algo] = -np.sqrt(z_observed ** 2 -
                                                  event_data['r_correction_3d_' + algo] ** 2)
        else:
            event_data['z_3d_' + algo] = z_observed

        event_data['z_correction_3d_' + algo] = event_data['z_3d_' + algo] - z_observed

        # s1 area fraction near injection points for Rn220 source
        area_upper_injection = (s1.area_per_channel[131] + s1.area_per_channel[138] +
                                s1.area_per_channel[146] + s1.area_per_channel[147])
        area_lower_injection = (s1.area_per_channel[236] + s1.area_per_channel[237] +
                                s1.area_per_channel[243])

        event_data['s1_area_upper_injection_fraction'] = area_upper_injection / s1.area
        event_data['s1_area_lower_injection_fraction'] = area_lower_injection / s1.area

        # Want S1 AreaFractionTop Probability
        aft_prob = self.corrections_handler.get_correction_from_map(
            "s1_aft_map", self.run_number, [self.x[event_index], self.y[event_index], self.z[event_index]])

        aft_args = aft_prob, s1.area, s1.area_fraction_top, s1.n_hits, s1.hits_fraction_top

        event_data['s1_area_fraction_top_probability_hax'] = s1_area_fraction_top_probability(*aft_args)
        event_data['s1_area_fraction_top_binomial'] = s1_area_fraction_top_probability(*(aft_args + (10, 'pmf')))

        event_data['s1_area_fraction_top_probability_nothresh'] = s1_area_fraction_top_probability(*(aft_args + (0,)))
        event_data['s1_area_fraction_top_binomial_nothresh'] = s1_area_fraction_top_probability(*(aft_args + (0, 'pmf')))

        # Now do s1_pattern_fit
        apc = np.array(list(s1.area_per_channel))
        hpc = np.array(list(s1.hits_per_channel))

        # Get saturated channels
        confused_s1_channels = []
        for a, c in enumerate(s1.n_saturated_per_channel):
            if c > 0:
                confused_s1_channels.append(a)

        try:

            # Create PMT array of booleans for use in likelihood calculation
            is_pmt_in = np.ones(len(self.tpc_channels), dtype=bool)  # Default True
            is_pmt_in[confused_s1_channels] = False  # Ignore saturated channels

            event_data['s1_pattern_fit_hax'] = self.pattern_fitter.compute_gof(
                (self.x[event_index], self.y[event_index], self.z[event_index]),
                apc[self.tpc_channels],
                pmt_selection=is_pmt_in,
                statistic=self.statistic)

            event_data['s1_pattern_fit_hits_hax'] = self.pattern_fitter.compute_gof(
                (self.x[event_index], self.y[event_index], self.z[event_index]),
                hpc[self.tpc_channels],
                pmt_selection=is_pmt_in,
                statistic=self.statistic)

            # Switch to bottom PMTs only
            is_pmt_in[self.pax_config['DEFAULT']['channels_top']] = False

            event_data['s1_pattern_fit_bottom_hax'] = self.pattern_fitter.compute_gof(
                (self.x[event_index], self.y[event_index], self.z[event_index]),
                apc[self.tpc_channels],
                pmt_selection=is_pmt_in,
                statistic=self.statistic)

            event_data['s1_pattern_fit_bottom_hits_hax'] = self.pattern_fitter.compute_gof(
                (self.x[event_index], self.y[event_index], self.z[event_index]),
                hpc[self.tpc_channels],
                pmt_selection=is_pmt_in,
                statistic=self.statistic)

        except exceptions.CoordinateOutOfRangeException as _:
            # pax does this too. happens when event out of TPC (usually z)
            return event_data

        return event_data
