import hax
import numpy as np
import json
from hax.minitrees import TreeMaker
from hax.runs import get_run_info
from pax.PatternFitter import PatternFitter
from pax.configuration import load_configuration
from pax import utils
from pax import exceptions
from pax.plugins.interaction_processing.S1AreaFractionTopProbability import s1_area_fraction_top_probability
from hax.corrections_handler import CorrectionsHandler

class PositionReconstructionDoubleScatter(TreeMaker):
    """Stores position-reconstruction-related variables.
    Provides:
        X = a or b
       - s1_X_pattern_fit_hax: S1 pattern likelihood computed with corrected
                             position and areas
       - s1_X_pattern_fit_hits_hax: S1 pattern likelihood computed with corrected
                                  position and hits
       - s1_X_pattern_fit_bottom_hax: S1 pattern likelihood computed with corrected
                                    position and bottom array area
       - s1_X_pattern_fit_bottom_hits_hax: S1 pattern likelihood computed with corrected
                                         position and bottom array hits
       - s1_X_area_fraction_top_probability_hax: S1 AFT p-value computed with corrected position
       - s1_X_area_fraction_top_probability_nothresh: computed using area below S1=10 (instead of hits)
       - s1_X_area_fraction_top_binomial: Binomial probability for given S1 AFT
       - s1_X_area_fraction_top_binomial_nothresh: Same except using area below S1=10
       - int_X_x_observed_nn_tf: TensorFlow NN reconstructed x position
       - int_X_y_observed_nn_tf: TensorFlow NN reconstructed y position
       - int_X_r_observed_nn_tf: TensorFlow NN reconstructed r position
       - int_X_r_3d_nn_tf: the corrected interaction r coordinate (data-driven 3d fdc)
       - int_X_x_3d_nn_tf: the corrected interaction x coordinate (data-driven 3d fdc)
       - int_X_y_3d_nn_tf: the corrected interaction y coordinate (data-driven 3d fdc)
       - int_X_z_3d_nn_tf: the corrected interaction z coordinate (data-driven 3d fdc)
       - int_X_r_correction_3d_nn_tf: r_3d_nn_tf - r_observed_nn_tf
       - int_X_z_correction_3d_nn_tf: z_3d_nn_tf - z_observed
       - s1_X_area_upper_injection_fraction: s1 area fraction near Rn220 injection points (near PMT 131)
       - s1_X_area_lower_injection_fraction: s1 area fraction near Rn220 injection points (near PMT 243)
       - s2_X_pattern_fit_nn: s2 pattern fit using nn position
    """
    __version__ = '1.0'
    extra_branches = ['peaks.area_per_channel[260]',
                      'peaks.hits_per_channel[260]',
                      'peaks.n_saturated_per_channel[260]',
                      'peaks.n_hits', 'peaks.hits_fraction_top', 'peaks.reconstructed_positions',
                      'interactions.x', 'interactions.y', 'interactions.z']

    def __init__(self):

        hax.minitrees.TreeMaker.__init__(self)
        self.extra_metadata = hax.config['corrections_definitions']
        self.corrections_handler = CorrectionsHandler()
        # We need to pull some stuff from the pax config
        self.pax_config = load_configuration("XENON1T")
        self.tpc_channels = self.pax_config['DEFAULT']['channels_in_detector']['tpc']
        self.confused_s1_channels = []
        self.s1_statistic = (
            self.pax_config['BuildInteractions.BasicInteractionProperties']['s1_pattern_statistic']
        )
        qes = np.array(self.pax_config['DEFAULT']['quantum_efficiencies'])

        self.s1_pattern_fitter = PatternFitter(
            filename=utils.data_file_name(self.pax_config['WaveformSimulator']['s1_patterns_file']),
            zoom_factor=self.pax_config['WaveformSimulator'].get('s1_patterns_zoom_factor', 1),
            adjust_to_qe=qes[self.tpc_channels],
            default_errors=(self.pax_config['DEFAULT']['relative_qe_error'] +
                            self.pax_config['DEFAULT']['relative_gain_error'])
        )

        self.top_channels = self.pax_config['DEFAULT']['channels_top']
        self.ntop_pmts = len(self.top_channels)
        # Declare nn stuff
        self.tfnn_weights = None
        self.tfnn_model = None
        self.loaded_nn = None
        # Run doc
        self.loaded_run_doc = None
        self.run_doc = None

    def load_nn(self):
        """For loading NN files"""
        from keras.models import model_from_json
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
        data, _ = hax.minitrees.load_single_dataset(dataset, ['CorrectedDoubleS1Scatter', 'Fundamentals'])
        self.int_a_x = data.int_a_x_3d_nn.values
        self.int_a_y = data.int_a_y_3d_nn.values
        self.int_a_z = data.int_a_z_3d_nn.values
        self.int_b_x = data.int_b_x_3d_nn.values
        self.int_b_y = data.int_b_y_3d_nn.values
        self.int_b_z = data.int_b_z_3d_nn.values
        self.indices = list(data.event_number.values)
        return hax.minitrees.TreeMaker.get_data(self, dataset, event_list)

    def load_run_doc(self, run):
        if run != self.loaded_run_doc:
            self.run_doc = get_run_info(run)
            self.loaded_run_doc = run

    def extract_data(self, event):
        event_data = {
            "s1_a_pattern_fit_hax": None,
            "s1_a_pattern_fit_hits_hax": None,
            "s1_a_pattern_fit_bottom_hax": None,
            "s1_a_pattern_fit_bottom_hits_hax": None,
            "s2_a_pattern_fit_nn": None,
            "s2_a_pattern_fit_tpf": None,
            "s1_a_area_fraction_top_probability_hax": None,
            "s1_a_area_fraction_top_probability_nothresh": None,
            "s1_a_area_fraction_top_binomial": None,
            "s1_a_area_fraction_top_binomial_nothresh": None,
            "int_a_x_observed_nn_tf": None,
            "int_a_y_observed_nn_tf": None,
            "int_a_r_observed_nn_tf": None,
            "int_a_x_3d_nn_tf": None,
            "int_a_y_3d_nn_tf": None,
            "int_a_r_3d_nn_tf": None,
            "int_a_z_3d_nn_tf": None,
            "int_a_r_correction_3d_nn_tf": None,
            "int_a_z_correction_3d_nn_tf": None,
            "s1_a_area_upper_injection_fraction": None,
            "s1_a_area_lower_injection_fraction": None,
            "s1_b_pattern_fit_hax": None,
            "s1_b_pattern_fit_hits_hax": None,
            "s1_b_pattern_fit_bottom_hax": None,
            "s1_b_pattern_fit_bottom_hits_hax": None,
            "s2_b_pattern_fit_nn": None,
            "s2_b_pattern_fit_tpf": None,
            "s1_b_area_fraction_top_probability_hax": None,
            "s1_b_area_fraction_top_probability_nothresh": None,
            "s1_b_area_fraction_top_binomial": None,
            "s1_b_area_fraction_top_binomial_nothresh": None,
            "int_b_x_observed_nn_tf": None,
            "int_b_y_observed_nn_tf": None,
            "int_b_r_observed_nn_tf": None,
            "int_b_x_3d_nn_tf": None,
            "int_b_y_3d_nn_tf": None,
            "int_b_r_3d_nn_tf": None,
            "int_b_z_3d_nn_tf": None,
            "int_b_r_correction_3d_nn_tf": None,
            "int_b_z_correction_3d_nn_tf": None,
            "s1_b_area_upper_injection_fraction": None,
            "s1_b_area_lower_injection_fraction": None,
        }

        # We first need the positions. This minitree is only valid when loading
        # Corrections since you need that to get the corrected positions
        if not len(event.interactions):
            return event_data
        # shortcuts for pax classes
        peaks = event.peaks
        interactions = event.interactions

        # Select Interactions for DoubleScatter Event
        # assume one scatter is interactions[0]
        int_0 = 0
        s1_0 = peaks[interactions[int_0].s1]
        s2_0 = peaks[interactions[int_0].s2]
        s1_0_int = interactions[int_0].s1
        s2_0_int = interactions[int_0].s2
        # find another scatter
        otherInts = [0, 0]
        for i, interaction in enumerate(interactions):
            if (interaction.s1 != s1_0_int and interaction.s2 == s2_0_int and otherInts[0] == 0):
                otherInts[0] = i
            elif (interaction.s1 != s1_0_int and interaction.s2 != s2_0_int and otherInts[1] == 0):
                otherInts[1] = i

        # Distinction b/w single and double s2 scatters
        # Cut events without second s1
        if otherInts[1] != 0:
            s1_1 = peaks[interactions[otherInts[1]].s1]
            s2_1 = peaks[interactions[otherInts[1]].s2]       
            s1_1_int = interactions[otherInts[1]].s1
            s2_1_int = interactions[otherInts[1]].s2
            int_1 = otherInts[1]
        elif otherInts[0] != 0:
            s1_1 = peaks[interactions[otherInts[0]].s1]
            s2_1 = peaks[interactions[otherInts[0]].s2]    
            s1_1_int = interactions[otherInts[0]].s1
            s2_1_int = interactions[otherInts[0]].s2   
            int_1 = otherInts[0]
        else:
            return dict()
        # order s1s/interactions by time
        if peaks[s1_0_int].center_time <= peaks[s1_1_int].center_time:
            s1_a = s1_0
            s1_b = s1_1
            s2_a = s2_0
            s2_b = s2_1
            int_a = int_0
            int_b = int_1
        else:
            s1_a = s1_1
            s1_b = s1_0
            s2_a = s2_1
            s2_b = s2_0
            int_a = int_1
            int_b = int_0

        event_num = event.event_number

        try:
            event_index = self.indices.index(event_num)
        except Exception:
            return event_data
        
        for rp in s2_a.reconstructed_positions:
            if rp.algorithm == "PosRecNeuralNet":
                event_data['s2_a_pattern_fit_nn'] = rp.goodness_of_fit
            elif rp.algorithm == "PosRecTopPatternFit":
                event_data['s2_a_pattern_fit_tpf'] = rp.goodness_of_fit

        for rp in s2_b.reconstructed_positions:
            if rp.algorithm == "PosRecNeuralNet":
                event_data['s2_b_pattern_fit_nn'] = rp.goodness_of_fit
            elif rp.algorithm == "PosRecTopPatternFit":
                event_data['s2_b_pattern_fit_tpf'] = rp.goodness_of_fit
                
        # Position reconstruction based on NN from TensorFlow
        # First Check for MC data, and avoid Tensor Flow if MC.
        if not self.mc_data:  # Temporary for OSG production
            # Check that correct NN is loaded and change if not
            self.load_nn()

            s2_a_apc = np.array(list(s2_a.area_per_channel))
            s2_a_apc_clean = []
            
            s2_b_apc = np.array(list(s2_b.area_per_channel))
            s2_b_apc_clean = []
            
            for ipmt, s2_t in enumerate(s2_a_apc):
                if ipmt not in self.list_bad_pmts and ipmt < self.ntop_pmts:
                    s2_a_apc_clean.append(s2_t)
            s2_a_apc_clean = np.asarray(s2_a_apc_clean)
            s2_a_apc_clean_norm = s2_a_apc_clean / s2_a_apc_clean.sum()
            s2_a_apc_clean_norm = s2_a_apc_clean_norm.reshape(1, len(s2_a_apc_clean_norm))

            predicted_xy_tensorflow_a = self.loaded_nn.predict(s2_a_apc_clean_norm)
            event_data['int_a_x_observed_nn_tf'] = predicted_xy_tensorflow_a[0, 0] / 10.
            event_data['int_a_y_observed_nn_tf'] = predicted_xy_tensorflow_a[0, 1] / 10.
            event_data['int_a_r_observed_nn_tf'] =\
                np.sqrt(event_data['int_a_x_observed_nn_tf']**2 + event_data['int_a_y_observed_nn_tf']**2)
                
            for ipmt, s2_t in enumerate(s2_b_apc):
                if ipmt not in self.list_bad_pmts and ipmt < self.ntop_pmts:
                    s2_b_apc_clean.append(s2_t)    
                
            s2_b_apc_clean = np.asarray(s2_b_apc_clean)
            s2_b_apc_clean_norm = s2_b_apc_clean / s2_b_apc_clean.sum()
            s2_b_apc_clean_norm = s2_b_apc_clean_norm.reshape(1, len(s2_b_apc_clean_norm))

            predicted_xy_tensorflow_b = self.loaded_nn.predict(s2_b_apc_clean_norm)
            event_data['int_b_x_observed_nn_tf'] = predicted_xy_tensorflow_b[0, 0] / 10.
            event_data['int_b_y_observed_nn_tf'] = predicted_xy_tensorflow_b[0, 1] / 10.
            event_data['int_b_r_observed_nn_tf'] =\
                np.sqrt(event_data['int_b_x_observed_nn_tf']**2 + event_data['int_b_y_observed_nn_tf']**2)
                         
            # 3D FDC
            algo = 'nn_tf'
            z_observed = interactions[int_a].z - interactions[int_a].z_correction
            cvals = [event_data['int_a_x_observed_' + algo], event_data['int_a_y_observed_' + algo], z_observed]
            event_data['int_a_r_correction_3d_' + algo] = self.corrections_handler.get_correction_from_map(
                "fdc_3d_tfnn", self.run_number, cvals)

            event_data['int_a_r_3d_' + algo] = (event_data['int_a_r_observed_' + algo] +
                                          event_data['int_a_r_correction_3d_' + algo])
            event_data['int_a_x_3d_' + algo] = (event_data['int_a_x_observed_' + algo] *
                                          (event_data['int_a_r_3d_' + algo] /
                                           event_data['int_a_r_observed_' + algo]))
            event_data['int_a_y_3d_' + algo] = (event_data['int_a_y_observed_' + algo] *
                                          (event_data['int_a_r_3d_' + algo] /
                                           event_data['int_a_r_observed_' + algo]))

            if abs(z_observed) > abs(event_data['int_a_r_correction_3d_' + algo]):
                event_data['int_a_z_3d_' + algo] = -np.sqrt(z_observed ** 2 -
                                                      event_data['int_a_r_correction_3d_' + algo] ** 2)
            else:
                event_data['int_a_z_3d_' + algo] = z_observed

            event_data['int_a_z_correction_3d_' + algo] = event_data['int_a_z_3d_' + algo] - z_observed
            # int_b
            z_observed = interactions[int_b].z - interactions[int_b].z_correction
            cvals = [event_data['int_b_x_observed_' + algo], event_data['int_b_y_observed_' + algo], z_observed]
            event_data['int_b_r_correction_3d_' + algo] = self.corrections_handler.get_correction_from_map(
                "fdc_3d_tfnn", self.run_number, cvals)

            event_data['int_b_r_3d_' + algo] = (event_data['int_b_r_observed_' + algo] +
                                          event_data['int_b_r_correction_3d_' + algo])
            event_data['int_b_x_3d_' + algo] = (event_data['int_b_x_observed_' + algo] *
                                          (event_data['int_b_r_3d_' + algo] /
                                           event_data['int_b_r_observed_' + algo]))
            event_data['int_b_y_3d_' + algo] = (event_data['int_b_y_observed_' + algo] *
                                          (event_data['int_b_r_3d_' + algo] /
                                           event_data['int_b_r_observed_' + algo]))

            if abs(z_observed) > abs(event_data['int_b_r_correction_3d_' + algo]):
                event_data['int_b_z_3d_' + algo] = -np.sqrt(z_observed ** 2 -
                                                      event_data['int_b_r_correction_3d_' + algo] ** 2)
            else:
                event_data['int_b_z_3d_' + algo] = z_observed

            event_data['int_b_z_correction_3d_' + algo] = event_data['int_b_z_3d_' + algo] - z_observed
            
        # s1 area fraction near injection points for Rn220 source
        area_upper_injection = (s1_a.area_per_channel[131] + s1_a.area_per_channel[138] +
                                s1_a.area_per_channel[146] + s1_a.area_per_channel[147])
        area_lower_injection = (s1_a.area_per_channel[236] + s1_a.area_per_channel[237] +
                                s1_a.area_per_channel[243])

        event_data['s1_a_area_upper_injection_fraction'] = area_upper_injection / s1_a.area
        event_data['s1_a_area_lower_injection_fraction'] = area_lower_injection / s1_a.area

        # Want S1 AreaFractionTop Probability
        aft_prob = self.corrections_handler.get_correction_from_map(
            "s1_aft_map", self.run_number, [self.int_a_x[event_index], self.int_a_y[event_index], self.int_a_z[event_index]])

        aft_args = aft_prob, s1_a.area, s1_a.area_fraction_top, s1_a.n_hits, s1_a.hits_fraction_top

        event_data['s1_a_area_fraction_top_probability_hax'] = s1_area_fraction_top_probability(*aft_args)
        event_data['s1_a_area_fraction_top_binomial'] = s1_area_fraction_top_probability(*(aft_args + (10, 'pmf')))

        event_data['s1_a_area_fraction_top_probability_nothresh'] = s1_area_fraction_top_probability(*(aft_args + (0,)))
        event_data['s1_a_area_fraction_top_binomial_nothresh'] = s1_area_fraction_top_probability(*(aft_args + (0, 'pmf')))

        # Now do s1_pattern_fit
        apc = np.array(list(s1_a.area_per_channel))
        hpc = np.array(list(s1_a.hits_per_channel))

        # Get saturated channels
        confused_s1_channels = []
        self.load_run_doc(self.run_number)

        # The original s1 pattern calculation had a bug where dead PMTs were
        # included. They are not included here.
        for a, c in enumerate(self.run_doc['processor']['DEFAULT']['gains']):
            if c == 0:
                confused_s1_channels.append(a)
        for a, c in enumerate(s1_a.n_saturated_per_channel):
            if c > 0:
                confused_s1_channels.append(a)
        try:
            # Create PMT array of booleans for use in likelihood calculation
            is_pmt_in = np.ones(len(self.tpc_channels), dtype=bool)  # Default True
            is_pmt_in[confused_s1_channels] = False  # Ignore saturated channels

            event_data['s1_a_pattern_fit_hax'] = self.s1_pattern_fitter.compute_gof(
                (self.int_a_x[event_index], self.int_a_y[event_index], self.int_a_z[event_index]),
                apc[self.tpc_channels],
                pmt_selection=is_pmt_in,
                statistic=self.s1_statistic)

            event_data['s1_a_pattern_fit_hits_hax'] = self.s1_pattern_fitter.compute_gof(
                (self.int_a_x[event_index], self.int_a_y[event_index], self.int_a_z[event_index]),
                hpc[self.tpc_channels],
                pmt_selection=is_pmt_in,
                statistic=self.s1_statistic)
            # Switch to bottom PMTs only
            is_pmt_in[self.top_channels] = False

            event_data['s1_a_pattern_fit_bottom_hax'] = self.s1_pattern_fitter.compute_gof(
                (self.int_a_x[event_index], self.int_a_y[event_index], self.int_a_z[event_index]),
                apc[self.tpc_channels],
                pmt_selection=is_pmt_in,
                statistic=self.s1_statistic)

            event_data['s1_a_pattern_fit_bottom_hits_hax'] = self.s1_pattern_fitter.compute_gof(
                (self.int_a_x[event_index], self.int_a_y[event_index], self.int_a_z[event_index]),
                hpc[self.tpc_channels],
                pmt_selection=is_pmt_in,
                statistic=self.s1_statistic)

        except exceptions.CoordinateOutOfRangeException as _:
            # pax does this too. happens when event out of TPC (usually z)
            return event_data
        
        ## INT_B
        # s1 area fraction near injection points for Rn220 source
        area_upper_injection = (s1_b.area_per_channel[131] + s1_b.area_per_channel[138] +
                                s1_b.area_per_channel[146] + s1_b.area_per_channel[147])
        area_lower_injection = (s1_b.area_per_channel[236] + s1_b.area_per_channel[237] +
                                s1_b.area_per_channel[243])

        event_data['s1_b_area_upper_injection_fraction'] = area_upper_injection / s1_b.area
        event_data['s1_b_area_lower_injection_fraction'] = area_lower_injection / s1_b.area

        # Want S1 AreaFractionTop Probability
        aft_prob = self.corrections_handler.get_correction_from_map(
            "s1_aft_map", self.run_number, [self.int_b_x[event_index], self.int_b_y[event_index], self.int_b_z[event_index]])

        aft_args = aft_prob, s1_b.area, s1_b.area_fraction_top, s1_b.n_hits, s1_b.hits_fraction_top

        event_data['s1_b_area_fraction_top_probability_hax'] = s1_area_fraction_top_probability(*aft_args)
        event_data['s1_b_area_fraction_top_binomial'] = s1_area_fraction_top_probability(*(aft_args + (10, 'pmf')))

        event_data['s1_b_area_fraction_top_probability_nothresh'] = s1_area_fraction_top_probability(*(aft_args + (0,)))
        event_data['s1_b_area_fraction_top_binomial_nothresh'] = s1_area_fraction_top_probability(*(aft_args + (0, 'pmf')))

        # Now do s1_pattern_fit
        apc = np.array(list(s1_b.area_per_channel))
        hpc = np.array(list(s1_b.hits_per_channel))

        # Get saturated channels
        confused_s1_channels = []
        self.load_run_doc(self.run_number)

        # The original s1 pattern calculation had a bug where dead PMTs were
        # included. They are not included here.
        for a, c in enumerate(self.run_doc['processor']['DEFAULT']['gains']):
            if c == 0:
                confused_s1_channels.append(a)
        for a, c in enumerate(s1_b.n_saturated_per_channel):
            if c > 0:
                confused_s1_channels.append(a)
        try:
            # Create PMT array of booleans for use in likelihood calculation
            is_pmt_in = np.ones(len(self.tpc_channels), dtype=bool)  # Default True
            is_pmt_in[confused_s1_channels] = False  # Ignore saturated channels

            event_data['s1_b_pattern_fit_hax'] = self.s1_pattern_fitter.compute_gof(
                (self.int_b_x[event_index], self.int_b_y[event_index], self.int_b_z[event_index]),
                apc[self.tpc_channels],
                pmt_selection=is_pmt_in,
                statistic=self.s1_statistic)

            event_data['s1_b_pattern_fit_hits_hax'] = self.s1_pattern_fitter.compute_gof(
                (self.int_b_x[event_index], self.int_b_y[event_index], self.int_b_z[event_index]),
                hpc[self.tpc_channels],
                pmt_selection=is_pmt_in,
                statistic=self.s1_statistic)
            # Switch to bottom PMTs only
            is_pmt_in[self.top_channels] = False

            event_data['s1_b_pattern_fit_bottom_hax'] = self.s1_pattern_fitter.compute_gof(
                (self.int_b_x[event_index], self.int_b_y[event_index], self.int_b_z[event_index]),
                apc[self.tpc_channels],
                pmt_selection=is_pmt_in,
                statistic=self.s1_statistic)

            event_data['s1_b_pattern_fit_bottom_hits_hax'] = self.s1_pattern_fitter.compute_gof(
                (self.int_b_x[event_index], self.int_b_y[event_index], self.int_b_z[event_index]),
                hpc[self.tpc_channels],
                pmt_selection=is_pmt_in,
                statistic=self.s1_statistic)
        except exceptions.CoordinateOutOfRangeException as _:
            # pax does this too. happens when event out of TPC (usually z)
            return event_data
        
        return event_data
