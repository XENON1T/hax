# Get pax values
import hax
from hax.minitrees import TreeMaker
import numpy as np
from hax.corrections_handler import CorrectionsHandler
from hax.treemakers.corrections import tfnn_position_reconstruction

class CorrectedDoubleS1Scatter(TreeMaker):
    """Applies high level corrections which are used in DoubleS1Scatter analyses.
    Be carefull, this treemaker was developed for Kr83m analysis. It will probably need modifications
     for other analysis.
    The search for double scatter events: made by Ted Berger
        double decays, afterpulses, and anything else that gets in our way
        if you have any questions contact Ted Berger (berget2@rpi.edu)
    The search proceeds as follows:
      * interaction[0] (int_0) provides s1_0 and s2_0
      * find additional interaction (int_1) to provide s1_1 and s2_1
      - loop through interactions (interactions store s1/s2 pairs in
      descending size order, s2s on fast loop)
      Choice A) select first interaction with s1 != s1_0 AND s2 != s2_0
      Choice B) if Choice A doesn't exist, select first interaction with s1 != s1_0 AND s2 == s2_0
      Choice C) if Choice A and B don't exist ignore, this isn't a double scatter event
      * int_0 and int_1 ordered by s1.center_time to int_a and int_b (int_a has s1 that happened first)
      The output provides the following variables attributed to specific peaks
      (s1_a, s2_a, s1_b, s2_b), as well as specific interactions (int_a, int_b).
    ### Peak Output (for PEAK in [s1_a, s2_a, s1_b, s2_b]):
     - PEAK: The uncorrected area in pe of the peak
     - PEAK_center_time: The center_time in ns of the peak
    ### DoubleScatter Specific Output:
    - ds_s1_b_n_distinct_channels: number of PMTs contributing to s1_b distinct from the PMTs that contributed to s1_a
    - ds_s1_dt : delay time between s1_a_center_time and s1_b_center_time
    - ds_second_s2: 1 if selected interactions have distinct s2s
    ### Position Corrections : Same as corrections minitree but for int_a and int_b position
    ### Signal corrections: Same as corrections minitree using int_a positions for position dependant corrections
    """
    __version__ = '2.1'

    extra_branches = ['peaks.n_contributing_channels',
                      'peaks.center_time',
                      'peaks.area_per_channel[260]',
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
                      'start_time',
                      'peaks.hits*',
                      'interactions.s1_pattern_fit'
                      ]

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

        # shortcuts for pax classes
        peaks = event.peaks
        interactions = event.interactions

        # Select Interactions for DoubleScatter Event
        # assume one scatter is interactions[0]
        int_0 = 0
        s1_0 = interactions[int_0].s1
        s2_0 = interactions[int_0].s2

        # find another scatter
        otherInts = [0, 0]
        for i, interaction in enumerate(interactions):
            if (interaction.s1 != s1_0 and interaction.s2 == s2_0 and otherInts[0] == 0):
                otherInts[0] = i
            elif (interaction.s1 != s1_0 and interaction.s2 != s2_0 and otherInts[1] == 0):
                otherInts[1] = i

        # Distinction b/w single and double s2 scatters
        # Cut events without second s1
        if otherInts[1] != 0:
            s1_1 = interactions[otherInts[1]].s1
            s2_1 = interactions[otherInts[1]].s2
            int_1 = otherInts[1]
            ds_second_s2 = 1
        elif otherInts[0] != 0:
            s1_1 = interactions[otherInts[0]].s1
            s2_1 = interactions[otherInts[0]].s2
            int_1 = otherInts[0]
            ds_second_s2 = 0
        else:
            return dict()

        # order s1s/interactions by time
        if peaks[s1_0].center_time <= peaks[s1_1].center_time:
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

        # Additional s1s and s2s removed! see v0.1.0
        result['s1_a'] = peaks[s1_a].area
        result['s1_a_center_time'] = peaks[s1_a].center_time
        result['s1_a_area_fraction_top'] = peaks[s1_a].area_fraction_top
        result['s1_a_range_50p_area'] = peaks[s1_a].range_area_decile[5]
        result['s2_a'] = peaks[s2_a].area
        result['s2_a_center_time'] = peaks[s2_a].center_time
        result['s2_a_bottom'] = (1.0 - peaks[s2_a].area_fraction_top) * peaks[s2_a].area
        result['s2_a_area_fraction_top'] = peaks[s2_a].area_fraction_top
        result['s2_a_range_50p_area'] = peaks[s2_a].range_area_decile[5]
        result['s1_b'] = peaks[s1_b].area
        result['s1_b_center_time'] = peaks[s1_b].center_time
        result['s1_b_area_fraction_top'] = peaks[s1_b].area_fraction_top
        result['s1_b_range_50p_area'] = peaks[s1_b].range_area_decile[5]
        result['s2_b'] = peaks[s2_b].area
        result['s2_b_center_time'] = peaks[s2_b].center_time
        result['s2_b_bottom'] = (1.0 - peaks[s2_b].area_fraction_top) * peaks[s2_b].area
        result['s2_b_area_fraction_top'] = peaks[s2_b].area_fraction_top
        result['s2_b_range_50p_area'] = peaks[s2_b].range_area_decile[5]
        result['ds_second_s2'] = ds_second_s2

        # Drift Time
        result['int_a_drift_time'] = result['s2_a_center_time'] - result['s1_a_center_time']
        result['int_b_drift_time'] = result['s2_b_center_time'] - result['s1_b_center_time']

        # Compute DoubleScatter Specific Variables
        # Select largest hits on each channel in s10 and s11 peaks
        s1_a_hitChannels = []
        s1_a_hitAreas = []
        for hit in peaks[s1_a].hits:
            if hit.is_rejected:
                continue
            if hit.channel not in s1_a_hitChannels:
                s1_a_hitChannels.append(hit.channel)
                s1_a_hitAreas.append(hit.area)
            else:
                hitChannel_i = s1_a_hitChannels.index(hit.channel)
                if hit.area > s1_a_hitAreas[hitChannel_i]:
                    s1_a_hitAreas[hitChannel_i] = hit.area
        s1_b_hitChannels = []
        s1_b_hitAreas = []

        for hit in peaks[s1_b].hits:
            if hit.is_rejected:
                continue
            if hit.channel not in s1_b_hitChannels:
                s1_b_hitChannels.append(hit.channel)
                s1_b_hitAreas.append(hit.area)
            else:
                hitChannel_i = s1_b_hitChannels.index(hit.channel)
                if hit.area > s1_b_hitAreas[hitChannel_i]:
                    s1_b_hitAreas[hitChannel_i] = hit.area

        # count largest-hit channels in s1_b distinct from s1_a
        ds_s1_b_n_distinct_channels = 0
        for i, channel in enumerate(s1_b_hitChannels):
            if channel not in s1_a_hitChannels:
                ds_s1_b_n_distinct_channels += 1
        result['ds_s1_b_n_distinct_channels'] = ds_s1_b_n_distinct_channels
        result['ds_s1_dt'] = peaks[s1_b].center_time - peaks[s1_a].center_time
        # Need the observed ('uncorrected') position.
        # pax Interaction positions are corrected so lookup the
        # uncorrected positions in the ReconstructedPosition objects
        for rp in peaks[s2_a].reconstructed_positions:
            if rp.algorithm == 'PosRecNeuralNet':
                result['int_a_x_observed_nn'] = rp.x
                result['int_a_y_observed_nn'] = rp.y
                result['int_a_r_observed_nn'] = np.sqrt(rp.x ** 2 + rp.y ** 2)
            elif rp.algorithm == 'PosRecTopPatternFit':
                result['int_a_x_observed_tpf'] = rp.x
                result['int_a_y_observed_tpf'] = rp.y
                result['int_a_r_observed_tpf'] = np.sqrt(rp.x ** 2 + rp.y ** 2)
        for rp in peaks[s2_b].reconstructed_positions:
            if rp.algorithm == 'PosRecNeuralNet':
                result['int_b_x_observed_nn'] = rp.x
                result['int_b_y_observed_nn'] = rp.y
                result['int_b_r_observed_nn'] = np.sqrt(rp.x ** 2 + rp.y ** 2)
            elif rp.algorithm == 'PosRecTopPatternFit':
                result['int_b_x_observed_tpf'] = rp.x
                result['int_b_y_observed_tpf'] = rp.y
                result['int_b_r_observed_tpf'] = np.sqrt(rp.x ** 2 + rp.y ** 2)
        int_a_z = interactions[int_a].z - interactions[int_a].z_correction
        result['int_a_z_observed'] = int_a_z
        int_b_z = interactions[int_b].z - interactions[int_b].z_correction
        result['int_b_z_observed'] = int_b_z

        int_signal = ['int_a_', 'int_b_']
        for int_s in int_signal:
            # Position reconstruction based on NN from TensorFlow
            # First Check for MC data, and avoid Tensor Flow if MC.
            if not self.mc_data:  # Temporary for OSG production
                # Calculate TF_NN reconstructed position
                predicted_xy_tensorflow = self.tfnn_posrec(list(peaks[s2_a].area_per_channel), self.run_number)

                result[int_s+'x_observed_nn_tf'] = predicted_xy_tensorflow[0, 0] / 10.
                result[int_s+'y_observed_nn_tf'] = predicted_xy_tensorflow[0, 1] / 10.
                result[int_s+'r_observed_nn_tf'] =\
                    np.sqrt(result[int_s+'x_observed_nn_tf']**2 + result[int_s+'y_observed_nn_tf']**2)

                # 3D FDC NN_TF
                algo = 'nn_tf'
                cvals = [result[int_s+'x_observed_' + algo],
                         result[int_s+'y_observed_' + algo], result[int_s+'z_observed']]
                result[int_s+'r_correction_3d_' + algo] = self.corrections_handler.get_correction_from_map(
                    "fdc_3d_tfnn", self.run_number, cvals)

                result[int_s+'r_3d_' + algo] = (result[int_s+'r_observed_' + algo] +
                                                result[int_s+'r_correction_3d_' + algo])
                result[int_s+'x_3d_' + algo] = (result[int_s+'x_observed_' + algo] *
                                                (result[int_s+'r_3d_' + algo] / result[int_s+'r_observed_' + algo]))
                result[int_s+'y_3d_' + algo] = (result[int_s+'y_observed_' + algo] *
                                                (result[int_s+'r_3d_' + algo] / result[int_s+'r_observed_' + algo]))

                if abs(result[int_s+'z_observed']) > abs(result[int_s+'r_correction_3d_' + algo]):
                    result[int_s + 'z_3d_' + algo] = -np.sqrt(result[int_s+'z_observed'] ** 2 -
                                                              result[int_s+'r_correction_3d_' + algo] ** 2)
                else:
                    result[int_s + 'z_3d_' + algo] = result[int_s+'z_observed']

                result[int_s + 'z_correction_3d_' + algo] = result[int_s+'z_3d_' + algo] - result[int_s+'z_observed']

            # Apply the 3D data driven NN_FDC, for NN positions and TPF positions
            for algo in ['nn', 'tpf']:
                cvals = [result[int_s+'x_observed_' + algo], result[int_s+'y_observed_' + algo],
                         result[int_s+'z_observed']]
                result[int_s+'r_correction_3d_' + algo] = self.corrections_handler.get_correction_from_map(
                    "fdc_3d", self.run_number, cvals)

                result[int_s+'r_3d_' + algo] =\
                    result[int_s+'r_observed_' + algo] + result[int_s+'r_correction_3d_' + algo]
                result[int_s+'x_3d_' + algo] = result[int_s+'x_observed_' + algo] \
                    * (result[int_s+'r_3d_' + algo] / result[int_s+'r_observed_' + algo])
                result[int_s+'y_3d_' + algo] = result[int_s+'y_observed_' + algo] \
                    * (result[int_s+'r_3d_' + algo] / result[int_s+'r_observed_' + algo])

                if abs(result[int_s+'z_observed']) > abs(result[int_s+'r_correction_3d_' + algo]):
                    result[int_s+'z_3d_' + algo] = -np.sqrt(result[int_s+'z_observed'] ** 2 -
                                                            result[int_s+'r_correction_3d_' + algo] ** 2)
                else:
                    result[int_s+'z_3d_' + algo] = result[int_s+'z_observed']

                result[int_s+'z_correction_3d_' + algo] = result[int_s+'z_3d_' + algo] - result[int_s+'z_observed']
        # include electron lifetime correction
        result['s2_lifetime_correction'] = (
            self.corrections_handler.get_electron_lifetime_correction(
                self.run_number, self.run_start, result['int_a_drift_time'], self.mc_data))
        # Correction only with int_a
        int_s_default = 'int_a_'
        for algo in ['nn_tf', 'nn', 'tpf']:
            # Correct S2
            result[int_s_default+'r_observed_' + algo] = np.sqrt(result[int_s_default+'x_observed_' + algo] ** 2
                                                                 + result[int_s_default+'y_observed_' + algo] ** 2)

            cvals = [result[int_s_default+'x_observed_' + algo], result[int_s_default+'y_observed_' + algo]]
            result[int_s_default+'s2_xy_correction_tot_' + algo] = \
                (1.0 / self.corrections_handler.get_correction_from_map("s2_xy_map", self.run_number, cvals))
            result[int_s_default+'s2_xy_correction_top_' + algo] = (1.0 /
                                                                    self.corrections_handler.get_correction_from_map(
                                                                        "s2_xy_map", self.run_number, cvals,
                                                                        map_name='map_top'))
            result[int_s_default+'s2_xy_correction_bottom_' + algo] = (1.0 /
                                                                       self.corrections_handler.get_correction_from_map(
                                                                           "s2_xy_map", self.run_number,
                                                                           cvals, map_name='map_bottom'))

            # Combine all the s2 corrections
            result['cs2_a_' + algo] = peaks[s2_a].area * result['s2_lifetime_correction'] \
                * result[int_s_default+'s2_xy_correction_tot_' + algo]
            result['cs2_a_top_' + algo] = \
                peaks[s2_a].area * peaks[s2_a].area_fraction_top * \
                result['s2_lifetime_correction'] * result[int_s_default+'s2_xy_correction_top_' + algo]
            result['cs2_a_bottom_' + algo] = \
                peaks[s2_a].area * (1.0 - peaks[s2_a].area_fraction_top) * \
                result['s2_lifetime_correction'] * result[int_s_default+'s2_xy_correction_bottom_' + algo]

            # Correct S1_a
            cvals = [result[int_s_default+'x_3d_' + algo], result[int_s_default+'y_3d_' + algo],
                     result[int_s_default+'z_3d_' + algo]]
            result[int_s_default+'s1_xyz_correction_fdc_3d_' + algo] = \
                (1 / self.corrections_handler.get_correction_from_map(
                    "s1_lce_map_nn_fdc_3d", self.run_number, cvals))
            result['cs1_a_no_field_corr_' + algo] = peaks[s1_a].area * \
                result[int_s_default+'s1_xyz_correction_fdc_3d_' + algo]
            # Apply corrected LCE (light collection efficiency correction to s1, including field effects)
            result[int_s_default+'s1_xyz_true_correction_fdc_3d' + algo] = \
                (1 / self.corrections_handler.get_correction_from_map(
                    "s1_corrected_lce_map_nn_fdc_3d", self.run_number, cvals))
            result['cs1_a_' + algo] = peaks[s1_a].area * result[int_s_default+'s1_xyz_true_correction_fdc_3d' + algo]
            # Correct S1_b
            result[int_s_default+'s1_xyz_correction_fdc_3d_' + algo] = \
                (1 / self.corrections_handler.get_correction_from_map(
                    "s1_lce_map_nn_fdc_3d", self.run_number, cvals))
            result['cs1_b_no_field_corr_' + algo] = peaks[s1_b].area * \
                result[int_s_default+'s1_xyz_correction_fdc_3d_' + algo]

            # Apply corrected LCE (light collection efficiency correction to s1, including field effects)
            result[int_s_default+'s1_xyz_true_correction_fdc_3d' + algo] = \
                (1 / self.corrections_handler.get_correction_from_map(
                    "s1_corrected_lce_map_nn_fdc_3d", self.run_number, cvals))
            result['cs1_b_' + algo] = peaks[s1_b].area * result[int_s_default+'s1_xyz_true_correction_fdc_3d' + algo]

        # default cS1 and cS2 values
        default_algo = 'nn_tf'
        result['cs1_a'] = result['cs1_a_' + default_algo]
        result['cs1_b'] = result['cs1_b_' + default_algo]
        result['cs2_a'] = result['cs2_a_' + default_algo]
        result['cs2_a_top'] = result['cs2_a_top_' + default_algo]
        result['cs2_a_bottom'] = result['cs2_a_bottom_' + default_algo]

        return result
