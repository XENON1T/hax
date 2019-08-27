import hax
import numpy as np
from hax.minitrees import TreeMaker
from hax.runs import get_run_info
from pax.PatternFitter import PatternFitter
from pax.configuration import load_configuration
from pax import utils
from pax import exceptions
from pax.plugins.interaction_processing.S1AreaFractionTopProbability import s1_area_fraction_top_probability
from hax.corrections_handler import CorrectionsHandler

from pax import configuration, datastructure
pax_config = configuration.load_configuration('XENON1T')

class PatternReconstruction(TreeMaker):
    """Stores interaction pattern-related variables.

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
       - s1_area_upper_injection_fraction: s1 area fraction near Rn220 injection points (near PMT 131)
       - s1_area_lower_injection_fraction: s1 area fraction near Rn220 injection points (near PMT 243)

       - s2_pattern_fit_nn: s2 pattern fit using nn position
    """
    __version__ = '1.3'
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

        # Run doc
        self.loaded_run_doc = None
        self.run_doc = None

    def get_data(self, dataset, event_list=None):
        # If we do switch to new NN later get rid of this stuff and directly use those positions!
        # WARNING: This 'bypass_blinding' flag should only be used for production and never for analysis (see #211)
        data, _ = hax.minitrees.load_single_dataset(dataset, ['Corrections', 'Fundamentals'], bypass_blinding=True)
        self.x = data.x_3d_nn_tf.values
        self.y = data.y_3d_nn_tf.values
        self.z = data.z_3d_nn_tf.values

        self.indices = list(data.event_number.values)

        return hax.minitrees.TreeMaker.get_data(self, dataset, event_list)

    def load_run_doc(self, run):
        if run != self.loaded_run_doc:
            self.run_doc = get_run_info(run)
            self.loaded_run_doc = run

    def extract_data(self, event):

        event_data = {
            "s1_pattern_fit_hax": None,
            "s1_pattern_fit_hits_hax": None,
            "s1_pattern_fit_bottom_hax": None,
            "s1_pattern_fit_bottom_hits_hax": None,
            "s2_pattern_fit_nn": None,
            "s2_pattern_fit_tpf": None,
            "s1_area_fraction_top_probability_hax": None,
            "s1_area_fraction_top_probability_nothresh": None,
            "s1_area_fraction_top_binomial": None,
            "s1_area_fraction_top_binomial_nothresh": None,
            "s1_area_upper_injection_fraction": None,
            "s1_area_lower_injection_fraction": None,
            "s2_pattern_fit_nn": None,
            "s2_pattern_fit_tpf": None
        }

        # We first need the positions. This minitree is only valid when loading
        # Corrections since you need that to get the corrected positions
        if not len(event.interactions):
            return event_data

        event_num = event.event_number

        try:
            event_index = self.indices.index(event_num)
        except Exception:
            return event_data

        interaction = event.interactions[0]
        s1 = event.peaks[interaction.s1]
        s2 = event.peaks[interaction.s2]

        for rp in s2.reconstructed_positions:
            if rp.algorithm == "PosRecNeuralNet":
                event_data['s2_pattern_fit_nn'] = rp.goodness_of_fit
            elif rp.algorithm == "PosRecTopPatternFit":
                event_data['s2_pattern_fit_tpf'] = rp.goodness_of_fit

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
        event_data['s1_area_fraction_top_binomial_nothresh'] = \
            s1_area_fraction_top_probability(*(aft_args + (0, 'pmf')))

        # Now do s1_pattern_fit
        apc = np.array(list(s1.area_per_channel))
        hpc = np.array(list(s1.hits_per_channel))

        # Get saturated channels
        confused_s1_channels = []
        self.load_run_doc(self.run_number)

        # The original s1 pattern calculation had a bug where dead PMTs were
        # included. They are not included here.
        for ch in self.tpc_channels:
            gain = self.run_doc['processor']['DEFAULT']['gains'][ch]
            if gain == 0:
                confused_s1_channels.append(ch)
        for a, c in enumerate(s1.n_saturated_per_channel):
            if c > 0:
                confused_s1_channels.append(a)

        try:

            # Create PMT array of booleans for use in likelihood calculation
            is_pmt_in = np.ones(len(self.tpc_channels), dtype=bool)  # Default True
            is_pmt_in[confused_s1_channels] = False  # Ignore saturated channels

            event_data['s1_pattern_fit_hax'] = self.s1_pattern_fitter.compute_gof(
                (self.x[event_index], self.y[event_index], self.z[event_index]),
                apc[self.tpc_channels],
                pmt_selection=is_pmt_in,
                statistic=self.s1_statistic)

            event_data['s1_pattern_fit_hits_hax'] = self.s1_pattern_fitter.compute_gof(
                (self.x[event_index], self.y[event_index], self.z[event_index]),
                hpc[self.tpc_channels],
                pmt_selection=is_pmt_in,
                statistic=self.s1_statistic)

            # Switch to bottom PMTs only
            is_pmt_in[self.top_channels] = False

            event_data['s1_pattern_fit_bottom_hax'] = self.s1_pattern_fitter.compute_gof(
                (self.x[event_index], self.y[event_index], self.z[event_index]),
                apc[self.tpc_channels],
                pmt_selection=is_pmt_in,
                statistic=self.s1_statistic)

            event_data['s1_pattern_fit_bottom_hits_hax'] = self.s1_pattern_fitter.compute_gof(
                (self.x[event_index], self.y[event_index], self.z[event_index]),
                hpc[self.tpc_channels],
                pmt_selection=is_pmt_in,
                statistic=self.s1_statistic)

        except exceptions.CoordinateOutOfRangeException:
            # pax does this too. happens when event out of TPC (usually z)
            return event_data

        return event_data


class S2PatternReducedAP(TreeMaker):
    '''
    Determination of the S2PatternLikelihood value when excluding PMTs that show large After-Pulse during SR1. 
    TO DO : Check for SR2.
    Allow to reduce the time dependance of the S2PatternLikelihood value used for cuts 
    (see:https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:chloetherreau:0vbb_s2_likelihood_cut_he_update)
    Need Correction and Fundamentals minitrees.
    '''
    __version__ = '1.0'
    extra_branches = ['peaks.area_per_channel*']
    
    def __init__(self):
        hax.minitrees.TreeMaker.__init__(self)
        # We need to pull some stuff from the pax config
        self.pax_config = load_configuration("XENON1T")
        self.channels_tot = self.pax_config['DEFAULT']['channels_in_detector']

        self.channels_top = self.pax_config['DEFAULT']['channels_top']
        self.is_pmt_alive = np.array(self.pax_config['DEFAULT']['gains']) > 1

        qes = np.array(self.pax_config['DEFAULT']['quantum_efficiencies'])

        # Create PMT array of booleans for use in likelihood calculation
        self.is_pmt_in = self.is_pmt_alive[self.channels_top]
        # After Pulse Channel
        self.apc = self.pax_config['DesaturatePulses.DesaturatePulses']['large_after_pulsing_channels']
        self.apc_top = list(set(self.apc).intersection(
            self.pax_config['DEFAULT']['channels_top']))
        self.s2_pattern_fitter = PatternFitter(
                    filename=utils.data_file_name(self.pax_config['WaveformSimulator']['s2_fitted_patterns_file']),
                    zoom_factor=self.pax_config['WaveformSimulator'].get('s2_fitted_patterns_zoom_factor', 1),
                    adjust_to_qe=qes[self.channels_top],
                    default_errors=(self.pax_config['DEFAULT']['relative_qe_error'] + 
                                    self.pax_config['DEFAULT']['relative_gain_error']))

    def get_data(self, dataset, event_list=None):
        # If we do switch to new NN later get rid of this stuff and directly use those positions!
        # WARNING: This 'bypass_blinding' flag should only be used for production and never for analysis (see #211)
        data, _ = hax.minitrees.load_single_dataset(dataset, ['Corrections', 'Fundamentals'])
        self.x = data.x_observed_nn_tf.values
        self.y = data.y_observed_nn_tf.values
        self.z = data.z_observed.values
        self.indices = list(data.event_number.values)
        return hax.minitrees.TreeMaker.get_data(self, dataset, event_list)
    
    def extract_data(self, event):
        result = dict()
        # If there are no interactions cannot do anything
        if not len(event.interactions):
            return result
        interaction = event.interactions[0]
        s2 = event.peaks[interaction.s2]

        result['s2_area_from_top_ap_pmt'] = np.sum(
            np.array(list(s2.area_per_channel))[self.apc_top])
        event_num = event.event_number
        try:
            event_index = self.indices.index(event_num)
        except Exception:
            return result
        area_per_channel = np.array(list(s2.area_per_channel))[self.channels_top]

        # Pattern fit with all PMTs
        try:
            result['s2_pattern_fit_top'] = self.s2_pattern_fitter.compute_gof(
                coordinates=(self.x[event_index], self.y[event_index]),
                areas_observed=area_per_channel,
                pmt_selection=self.is_pmt_in,
                statistic='likelihood_poisson')
        except exceptions.CoordinateOutOfRangeException as _:
            # pax does this too. happens when event out of TPC (usually z)
            return result

        # Remove all PMT AP channel
        self.is_pmt_in[self.apc_top] = False

        try:
            result['s2_pattern_fit_top_reduced_ap'] = self.s2_pattern_fitter.compute_gof(
                coordinates=(self.x[event_index], self.y[event_index]),
                areas_observed=area_per_channel,
                pmt_selection=self.is_pmt_in,
                statistic='likelihood_poisson')
        except exceptions.CoordinateOutOfRangeException as _:
            # pax does this too. happens when event out of TPC (usually z)
            return result
        return result
