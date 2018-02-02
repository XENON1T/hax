import hax
from hax.minitrees import TreeMaker
import numpy as np
from hax.corrections_handler import CorrectionsHandler


# Lone signal in pre_s1 window
class LoneSignalsPreS1(TreeMaker):
    __version__ = '0.2'
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
            result['s1_1_center_time'] = s1_sorted[1].center_time
            result['s1_1_aft'] = s1_sorted[1].area_fraction_top
            result['s1_1_n_hits'] = s1_sorted[1].n_hits
            result['s1_1_hit_aft'] = s1_sorted[1].hits_fraction_top
            result['s1_1_50p_width'] = s1_sorted[1].range_area_decile[5]
            result['s1_1_90p_width'] = s1_sorted[1].range_area_decile[9]
            result['s1_1_rise_time'] = -s1_sorted[1].area_decile_from_midpoint[1]
            result['s1_1_largest_hit_area'] = s1_sorted[1].largest_hit_area
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
class LoneSignals(TreeMaker):
    __version__ = '0.2'
    extra_branches = ['peaks.*']
    extra_metadata = hax.config['corrections_definitions']
    corrections_handler = CorrectionsHandler()

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

            result['s2_0_area'] = s2_sorted[0].area
            result['s2_0_center_time'] = s2_sorted[0].center_time
            result['s2_0_left'] = s2_sorted[0].center_time - s2_sorted[0].left
            result['s2_0_aft'] = s2_sorted[0].area_fraction_top
            result['s2_0_50p_width'] = s2_sorted[0].range_area_decile[5]
            result['s2_0_rise_time'] = -s2_sorted[0].area_decile_from_midpoint[1]
            result['s2_0_largest_hit_area'] = s2_sorted[0].largest_hit_area

            # S2 corrections based on X,Y maps, for new s2 AFT cut
            cvals = [result['s2_0_x_nn'], result['s2_0_y_nn']]
            result['s2_0_xy_correction_tot'] = (1.0 / self.corrections_handler.get_correction_from_map(
                "s2_xy_map", self.run_number, cvals))
            result['s2_0_xy_correction_top'] = (1.0 / self.corrections_handler.get_correction_from_map(
                "s2_xy_map", self.run_number, cvals, map_name='map_top'))
            result['s2_0_xy_correction_bottom'] = (1.0 / self.corrections_handler.get_correction_from_map(
                "s2_xy_map", self.run_number, cvals, map_name='map_bottom'))
            result['cs2_0_tot'] = result['s2_0_area'] * result['s2_0_xy_correction_tot']
            result['cs2_0_top'] = result['s2_0_area'] * result['s2_0_xy_correction_top'] * result['s2_0_aft']
            result['cs2_0_bottom'] = result['s2_0_area'] * result['s2_0_xy_correction_bottom'] * (1 - result['s2_0_aft'])
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

            result['s2_1_area'] = s2_sorted[1].area
            result['s2_1_center_time'] = s2_sorted[1].center_time
            result['s2_1_aft'] = s2_sorted[1].area_fraction_top
            result['s2_1_50p_width'] = s2_sorted[1].range_area_decile[5]
            result['s2_1_rise_time'] = -s2_sorted[1].area_decile_from_midpoint[1]
            result['s2_1_largest_hit_area'] = s2_sorted[1].largest_hit_area

        return result
