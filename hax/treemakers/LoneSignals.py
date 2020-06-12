import hax
from hax.minitrees import TreeMaker
import numpy as np
from hax.corrections_handler import CorrectionsHandler


# Lone signal in pre_s1 window

from hax.corrections_handler import CorrectionsHandler

class LoneSignalsPreS1(hax.minitrees.TreeMaker):
    __version__ = '0.3'
    extra_branches = ['peaks.*']

    def extract_data(self, event):
        peaks = event.peaks
        if not len(peaks):
            return dict()
        main_s1 = list(sorted([p for p in event.peaks if p.type == 's1' and p.detector == 'tpc'], key=lambda p: p.area,
                              reverse=True))
        main_s2 = list(sorted([p for p in event.peaks if p.type == 's2' and p.detector == 'tpc'], key=lambda p: p.area,
                              reverse=True))
        if not len(main_s1):
            return dict()
        if not len(main_s2):
            return dict()
        s1_sorted = list(sorted([p for p in event.peaks if p.type == 's1'
                                 and p.detector == 'tpc'
                                 and (p.center_time < main_s1[0].center_time - 4000 or p.area == main_s1[0].area)],
                                key=lambda p: p.area, reverse=True))
        s2_sorted = list(sorted([p for p in event.peaks if p.type == 's2' and p.detector == 'tpc' and (
            p.center_time < main_s1[0].center_time - 4000 or p.area == main_s2[0].area)], key=lambda p: p.area,
                                reverse=True))
        unknown = [peak for peak in peaks if peak.type == 'unknown' and peak.detector == "tpc"]
        result = dict(n_pulses=event.n_pulses, n_peaks=len(peaks), n_interactions=len(event.interactions))
        result['unknown_tot'] = np.sum([peak.area for peak in unknown])
        result['s1_area_tot'] = np.sum([peak.area for peak in s1_sorted])
        result['s2_area_tot'] = np.sum([peak.area for peak in s2_sorted])
        result['n_s1'] = len(s1_sorted)
        result['n_s2'] = len(s2_sorted)

        if len(s1_sorted):
            result['area_before_largest_s1'] = np.sum(
                [p.area for p in peaks if p.center_time < s1_sorted[0].center_time])
            s1_0_recpos = s1_sorted[0].reconstructed_positions
            for rp in s1_0_recpos:
                if (rp.algorithm == 'PosRecTopPatternFit'):
                    s1_0_recpos_pf = rp
                    result['s1_0_x'] = s1_0_recpos_pf.x
                    result['s1_0_y'] = s1_0_recpos_pf.y

            result['s1_0_area'] = s1_sorted[0].area
            result['s1_0_center_time'] = s1_sorted[0].center_time
            result['s1_0_aft'] = s1_sorted[0].area_fraction_top
            result['s1_0_50p_width'] = s1_sorted[0].range_area_decile[5]

        if len(s1_sorted) > 1:
            s1_1_recpos = s1_sorted[1].reconstructed_positions
            for rp in s1_1_recpos:
                if (rp.algorithm == 'PosRecTopPatternFit'):
                    s1_1_recpos_pf = rp
                    result['s1_1_x'] = s1_1_recpos_pf.x
                    result['s1_1_y'] = s1_1_recpos_pf.y
                    result['s1_1_posrec_goodness_of_fit'] = s1_1_recpos_pf.goodness_of_fit

            result['s1_1_area'] = s1_sorted[1].area
            result['s1_1_n_contributing_channels'] = s1_sorted[1].n_contributing_channels
            result['s1_1_tight_coincidence'] = s1_sorted[1].tight_coincidence
            result['s1_1_center_time'] = s1_sorted[1].center_time
            result['s1_1_aft'] = s1_sorted[1].area_fraction_top
            result['s1_1_n_hits'] = s1_sorted[1].n_hits
            result['s1_1_hit_aft'] = s1_sorted[1].hits_fraction_top
            result['s1_1_50p_width'] = s1_sorted[1].range_area_decile[5]
            result['s1_1_90p_width'] = s1_sorted[1].range_area_decile[9]
            result['s1_1_rise_time'] = -s1_sorted[1].area_decile_from_midpoint[1]
            result['s1_1_largest_hit_area'] = s1_sorted[1].largest_hit_area
            result['s1_1_largest_hit_channel'] = s1_sorted[1].largest_hit_channel
            
            # Shingo's new variable
            area_upper_injection = (s1_sorted[1].area_per_channel[131] + s1_sorted[1].area_per_channel[138] +
                                    s1_sorted[1].area_per_channel[146] + s1_sorted[1].area_per_channel[147])
            area_lower_injection = (s1_sorted[1].area_per_channel[236] + s1_sorted[1].area_per_channel[237] +
                                    s1_sorted[1].area_per_channel[243])
            result['s1_area_upper_injection_fraction'] = area_upper_injection / s1_sorted[1].area
            result['s1_area_lower_injection_fraction'] = area_lower_injection / s1_sorted[1].area

        if len(s2_sorted) > 0:
            result['area_before_largest_s2'] = np.sum(p.area for p in peaks if p.center_time < s2_sorted[0].center_time)
            s2_0_recpos = s2_sorted[0].reconstructed_positions
            for rp in s2_0_recpos:
                if (rp.algorithm == 'PosRecTopPatternFit'):
                    s2_0_recpos_pf = rp
                    result['s2_0_x'] = s2_0_recpos_pf.x
                    result['s2_0_y'] = s2_0_recpos_pf.y
                    result['s2_0_posrec_goodness_of_fit'] = s2_0_recpos_pf.goodness_of_fit

                if (rp.algorithm == 'PosRecNeuralNet'):
                    s2_0_recpos_pf = rp
                    result['s2_0_x_nn'] = s2_0_recpos_pf.x
                    result['s2_0_y_nn'] = s2_0_recpos_pf.y
                    result['s2_0_posrec_goodness_of_fit_nn'] = s2_0_recpos_pf.goodness_of_fit

            result['s2_0_area'] = s2_sorted[0].area
            result['s2_0_center_time'] = s2_sorted[0].center_time
            result['s2_0_aft'] = s2_sorted[0].area_fraction_top
            result['s2_0_50p_width'] = s2_sorted[0].range_area_decile[5]

        if len(s2_sorted) > 1:
            s2_1_recpos = s2_sorted[1].reconstructed_positions
            for rp in s2_1_recpos:
                if (rp.algorithm == 'PosRecTopPatternFit'):
                    s2_1_recpos_pf = rp
                    result['s2_1_x'] = s2_1_recpos_pf.x
                    result['s2_1_y'] = s2_1_recpos_pf.y
                    result['s2_1_posrec_goodness_of_fit'] = s2_1_recpos_pf.goodness_of_fit

                if (rp.algorithm == 'PosRecNeuralNet'):
                    s2_1_recpos_pf = rp
                    result['s2_1_x_nn'] = s2_1_recpos_pf.x
                    result['s2_1_y_nn'] = s2_1_recpos_pf.y
                    result['s2_1_posrec_goodness_of_fit_nn'] = s2_1_recpos_pf.goodness_of_fit

            result['s2_1_area'] = s2_sorted[1].area
            result['s2_1_center_time'] = s2_sorted[1].center_time
            result['s2_1_aft'] = s2_sorted[1].area_fraction_top
            result['s2_1_50p_width'] = s2_sorted[1].range_area_decile[5]
            result['s2_1_rise_time'] = -s2_sorted[1].area_decile_from_midpoint[1]
            result['s2_1_largest_hit_area'] = s2_sorted[1].largest_hit_area

        return result

# Extraction of peak information for Lone-S2/S1 studies
from hax.corrections_handler import CorrectionsHandler
from hax.treemakers.corrections import tfnn_position_reconstruction


class LoneSignals(TreeMaker):
    __version__ = '1.0'
    extra_branches = ['peaks.*']
    extra_metadata = hax.config['corrections_definitions']
    corrections_handler = CorrectionsHandler()

    def __init__(self):
        hax.minitrees.TreeMaker.__init__(self)
        self.tfnn_posrec = tfnn_position_reconstruction()

    def extract_data(self, event):
        peaks = event.peaks
        if not len(peaks):
            return dict()
        s1_sorted = list(
            sorted([p for p in event.peaks if p.type == 's1' and p.detector == 'tpc'], key=lambda p: p.area,
                   reverse=True))
        s2_sorted = list(
            sorted([p for p in event.peaks if p.type == 's2' and p.detector == 'tpc'], key=lambda p: p.area,
                   reverse=True))
        unknown = [peak for peak in peaks if peak.type == 'unknown' and peak.detector == "tpc"]
        result = dict(n_pulses=event.n_pulses, n_peaks=len(peaks), n_interactions=len(event.interactions))
        result['unknown_tot'] = np.sum([peak.area for peak in unknown])
        result['s1_area_tot'] = np.sum([peak.area for peak in s1_sorted])
        result['s2_area_tot'] = np.sum([peak.area for peak in s2_sorted])
        result['n_s1'] = len(s1_sorted)
        result['n_s2'] = len(s2_sorted)

        if len(s1_sorted):
            result['area_before_largest_s1'] = np.sum(
                [p.area for p in peaks if p.center_time < s1_sorted[0].center_time])
            s1_0_recpos = s1_sorted[0].reconstructed_positions
            for rp in s1_0_recpos:
                if (rp.algorithm == 'PosRecTopPatternFit'):
                    s1_0_recpos_pf = rp
                    result['s1_0_x'] = s1_0_recpos_pf.x
                    result['s1_0_y'] = s1_0_recpos_pf.y
                    result['s1_0_posrec_goodness_of_fit'] = s1_0_recpos_pf.goodness_of_fit

            result['s1_0_area'] = s1_sorted[0].area
            result['s1_0_center_time'] = s1_sorted[0].center_time
            result['s1_0_aft'] = s1_sorted[0].area_fraction_top
            result['s1_0_50p_width'] = s1_sorted[0].range_area_decile[5]
            result['s1_0_90p_width'] = s1_sorted[0].range_area_decile[9]
            result['s1_0_rise_time'] = -s1_sorted[0].area_decile_from_midpoint[1]
            result['s1_0_largest_hit_area'] = s1_sorted[0].largest_hit_area

        if len(s2_sorted) > 0:
            result['area_before_largest_s2'] = np.sum(p.area for p in peaks if p.center_time < s2_sorted[0].center_time)
            s2_0_recpos = s2_sorted[0].reconstructed_positions
            for rp in s2_0_recpos:
                if (rp.algorithm == 'PosRecTopPatternFit'):
                    s2_0_recpos_pf = rp
                    result['s2_0_x'] = s2_0_recpos_pf.x
                    result['s2_0_y'] = s2_0_recpos_pf.y
                    result['s2_0_posrec_goodness_of_fit'] = s2_0_recpos_pf.goodness_of_fit

                if (rp.algorithm == 'PosRecNeuralNet'):
                    s2_0_recpos_pf = rp
                    result['s2_0_x_nn'] = s2_0_recpos_pf.x
                    result['s2_0_y_nn'] = s2_0_recpos_pf.y
                    result['s2_0_posrec_goodness_of_fit_nn'] = s2_0_recpos_pf.goodness_of_fit

            s2_0_xy_tensorflow = self.tfnn_posrec(list(s2_sorted[0].area_per_channel), self.run_number)
            result['s2_0_x_nn_tf'] = s2_0_xy_tensorflow[0, 0] / 10.
            result['s2_0_y_nn_tf'] = s2_0_xy_tensorflow[0, 1] / 10.
            
            result['s2_0_area'] = s2_sorted[0].area
            result['s2_0_center_time'] = s2_sorted[0].center_time
            result['s2_0_left'] = s2_sorted[0].center_time - s2_sorted[0].left
            result['s2_0_aft'] = s2_sorted[0].area_fraction_top
            result['s2_0_50p_width'] = s2_sorted[0].range_area_decile[5]
            result['s2_0_rise_time'] = -s2_sorted[0].area_decile_from_midpoint[1]
            result['s2_0_largest_hit_area'] = s2_sorted[0].largest_hit_area

            # S2 corrections based on X,Y maps, for new s2 AFT cut
            cvals = [result['s2_0_x_nn_tf'], result['s2_0_y_nn_tf']]
            result['s2_0_xy_correction_tot'] = (1.0 / self.corrections_handler.get_correction_from_map(
                "s2_xy_map", self.run_number, cvals))
            result['s2_0_xy_correction_top'] = (1.0 / self.corrections_handler.get_correction_from_map(
                "s2_xy_map", self.run_number, cvals, map_name='map_top'))
            result['s2_0_xy_correction_bottom'] = (1.0 / self.corrections_handler.get_correction_from_map(
                "s2_xy_map", self.run_number, cvals, map_name='map_bottom'))
            result['cs2_0_tot'] = result['s2_0_area'] * result['s2_0_xy_correction_tot']
            result['cs2_0_top'] = result['s2_0_area'] * result['s2_0_xy_correction_top'] * result['s2_0_aft']
            result['cs2_0_bottom'] = \
                result['s2_0_area'] * result['s2_0_xy_correction_bottom'] * (1 - result['s2_0_aft'])
            result['cs2_0_aft'] = result['cs2_0_top'] / (result['cs2_0_top'] + result['cs2_0_bottom'])

        if len(s2_sorted) > 1:
            s2_1_recpos = s2_sorted[1].reconstructed_positions
            for rp in s2_1_recpos:
                if (rp.algorithm == 'PosRecTopPatternFit'):
                    s2_1_recpos_pf = rp
                    result['s2_1_x'] = s2_1_recpos_pf.x
                    result['s2_1_y'] = s2_1_recpos_pf.y
                    result['s2_1_posrec_goodness_of_fit'] = s2_1_recpos_pf.goodness_of_fit

                if (rp.algorithm == 'PosRecNeuralNet'):
                    s2_1_recpos_pf = rp
                    result['s2_1_x_nn'] = s2_1_recpos_pf.x
                    result['s2_1_y_nn'] = s2_1_recpos_pf.y
                    result['s2_1_posrec_goodness_of_fit_nn'] = s2_1_recpos_pf.goodness_of_fit

            s2_1_xy_tensorflow = self.tfnn_posrec(list(s2_sorted[1].area_per_channel), self.run_number)
            result['s2_1_x_nn_tf'] = s2_1_xy_tensorflow[0, 0] / 10.
            result['s2_1_y_nn_tf'] = s2_1_xy_tensorflow[0, 1] / 10.

            result['s2_1_area'] = s2_sorted[1].area
            result['s2_1_center_time'] = s2_sorted[1].center_time
            result['s2_1_aft'] = s2_sorted[1].area_fraction_top
            result['s2_1_50p_width'] = s2_sorted[1].range_area_decile[5]
            result['s2_1_rise_time'] = -s2_sorted[1].area_decile_from_midpoint[1]
            result['s2_1_largest_hit_area'] = s2_sorted[1].largest_hit_area
            
            # S2 corrections based on X,Y maps, for new s2 AFT cut
            cvals = [result['s2_1_x_nn_tf'], result['s2_1_y_nn_tf']]
            result['s2_1_xy_correction_tot'] = (1.0 / self.corrections_handler.get_correction_from_map(
                "s2_xy_map", self.run_number, cvals))
            result['s2_1_xy_correction_top'] = (1.0 / self.corrections_handler.get_correction_from_map(
                "s2_xy_map", self.run_number, cvals, map_name='map_top'))
            result['s2_1_xy_correction_bottom'] = (1.0 / self.corrections_handler.get_correction_from_map(
                "s2_xy_map", self.run_number, cvals, map_name='map_bottom'))
            result['cs2_1_tot'] = result['s2_1_area'] * result['s2_1_xy_correction_tot']
            result['cs2_1_top'] = result['s2_1_area'] * result['s2_1_xy_correction_top'] * result['s2_1_aft']
            result['cs2_1_bottom'] = \
                result['s2_1_area'] * result['s2_1_xy_correction_bottom'] * (1 - result['s2_1_aft'])
            result['cs2_1_aft'] = result['cs2_1_top'] / (result['cs2_1_top'] + result['cs2_1_bottom'])

        return result


import numba
class PeaksBeforeTrigger(hax.minitrees.TreeMaker):
    """Missing variables to investigate AC BG
    Provides:
    """
    # never_store = True
    __version__ = '0.4'
    branch_selection = [
        'peaks.*',
        'interactions.*',
    ]
    cache_size = 20000
    noisy_channel = np.array([21, 45, 58, 96, 110, 128, 159, 179, 207])
    
    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def n_noise_hit(hpc, noisy_channel):
        counts = 0
        for i in range(len(noisy_channel)):
            if hpc[noisy_channel[i]] > 0: counts += 1
        return counts

    def extract_data(self, event):
        cache = {'s1':[], 's2':[], 's1_i':[], 's2_i':[], 'unknown':[], 'lone_hit':[]}
        for peak_i, peak in enumerate(event.peaks):
            if peak.type in ['s1', 's2', 'unknown', 'lone_hit'] and peak.left < 1e5 - 400: 
                cache[peak.type].append(peak.area)
                if peak.type in ['s1', 's2']:
                    cache[peak.type+'_i'].append(peak_i)

        result = dict(n_s1_before_trigger=len(cache['s1']),
                      n_s2_before_trigger=len(cache['s2']),
                      n_unknown_before_trigger=len(cache['unknown']),
                      n_lone_hit_before_trigger=len(cache['lone_hit']),
                     )
        result['largest_s2_before_trigger'] = 0
        result['total_s2_before_trigger'] = 0
        result['largest_unknown_before_trigger'] = 0
        result['total_unknown_before_trigger'] = 0
        result['total_lone_hit_before_trigger'] = 0
        
        result['largest_s1_before_trigger'] = 0
        result['total_s1_before_trigger'] = 0
        result['noisy_hits_in_other_s1_before_trigger'] = 0
        result['s1_n_hits'] = 0
        result['s1_hit_aft'] = 0      
        result['noisy_hits_in_s1'] = 0
        result['s2_n_hits'] = 0
        result['s2_hit_aft'] = 0      


        if len(cache['s2']):
            result['largest_s2_before_trigger'] = np.max(cache['s2'])
            result['total_s2_before_trigger'] = np.sum(cache['s2'])

        if len(cache['unknown']):
            result['largest_unknown_before_trigger'] = np.max(cache['unknown'])
            result['total_unknown_before_trigger'] = np.sum(cache['unknown'])

        if len(cache['lone_hit']):
            result['total_lone_hit_before_trigger'] = np.sum(cache['lone_hit'])

        result['noisy_hits_in_other_s1_before_trigger'] = 0
        if len(cache['s1']):
            i_max = np.argmax(cache['s1'])
            result['largest_s1_before_trigger'] = cache['s1'][i_max]
            result['total_s1_before_trigger'] = np.sum(cache['s1'])
            result['noisy_hits_in_other_s1_before_trigger'] = self.n_noise_hit(
                event.peaks[cache['s1_i'][i_max]].hits_per_channel, self.noisy_channel)

        if len(event.interactions) != 0:
            s1 = event.peaks[event.interactions[0].s1]
            result['s1_n_hits'] = s1.n_hits
            result['s1_hit_aft'] = s1.hits_fraction_top        
            result['noisy_hits_in_s1'] = self.n_noise_hit(s1.hits_per_channel, self.noisy_channel)

            s2 = event.peaks[event.interactions[0].s2]
            result['s2_n_hits'] = s2.n_hits
            result['s2_hit_aft'] = s2.hits_fraction_top        
      
        return result


class NoiseRejection(hax.minitrees.TreeMaker):
    '''lowER group
    '''
    __version__ = '0.3'

    extra_branches = ['n_hits_rejected*', 'noise_pulses_in*']

    def extract_data(self, event):
        result = {}

        n_hits_rejected = np.array([i for i in event.n_hits_rejected])
        noise_pulses_in = np.array([i for i in event.noise_pulses_in])
        total_channels = len(n_hits_rejected)
        rejected_channels = len(n_hits_rejected[n_hits_rejected > 0])
        result['fraction_channels_rejected'] = rejected_channels / total_channels
        result['num_channels_rejected'] = rejected_channels
        result['num_hits_rejected'] = np.sum(n_hits_rejected)
        result['num_noise_pulses'] = np.sum(noise_pulses_in)
        result['fraction_channels_with_noise'] = len(noise_pulses_in[noise_pulses_in > 0])/len(noise_pulses_in)
        return result



