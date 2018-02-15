"""Tree makers for studying peaks on their own

Treemakers used for analyses such as the single-electron shape in time
and stability.
"""
import hax
from hax.minitrees import MultipleRowExtractor
import numpy as np
from pax import units


class PeakExtractor(MultipleRowExtractor):
    """Base class for reading peak data in minitrees. For more information, check out example 10 in hax/examples.
    """

    # Default branch selection is EVERYTHING in peaks, overwrite for speed increase
    # Don't forget to include branches used in cuts
    extra_branches = ['peaks.*']
    peak_fields = ['area']
    event_cut_list = []
    peak_cut_list = []
    event_cut_string = 'True'
    peak_cut_string = 'True'
    stop_after = np.inf

    # Hacks for want of string support :'(
    peaktypes = dict(lone_hit=0, s1=1, s2=2, unknown=3)
    detectors = dict(tpc=0, veto=1, sum_wv=2, busy_on=3, busy_off=4)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_cut_string = self.build_cut_string(
            self.event_cut_list, 'event')
        self.peak_cut_string = self.build_cut_string(
            self.peak_cut_list, 'peak')

    def build_cut_string(self, cut_list, obj):
        '''
        Build a string of cuts that can be applied using eval() function.
        '''
        # If no cut is specified, always pass cut
        if len(cut_list) == 0:
            return 'True'
        # Check if user entered range_50p_area, since this won't work
        cut_list = [
            cut.replace(
                'range_50p_area',
                'range_area_decile[5]') for cut in cut_list]

        cut_string = '('
        for cut in cut_list[:-1]:
            cut_string += obj + '.' + cut + ') & ('
        cut_string += obj + '.' + cut_list[-1] + ')'
        return cut_string

    def extract_data(self, event):
        if event.event_number == self.stop_after:
            raise hax.paxroot.StopEventLoop()

        peak_data = []
        # Check if event passes cut
        if eval(self.build_cut_string(self.event_cut_list, 'event')):
            # Loop over peaks and check if peak passes cut
            for peak in event.peaks:
                if eval(self.peak_cut_string):
                    # Loop over properties and add them to _current_peak one by one
                    _current_peak = {}
                    for field in self.peak_fields:
                        # Deal with special cases
                        if field == 'range_50p_area':
                            _x = list(peak.range_area_decile)[5]
                        elif field == 'rise_time':
                            _x = -peak.area_decile_from_midpoint[1]
                        elif field in ('x', 'y'):
                            # In case of x and y need to get position from reconstructed_positions
                            for rp in peak.reconstructed_positions:
                                if rp.algorithm == 'PosRecTopPatternFit':
                                    _x = getattr(rp, field)
                                    break
                            else:
                                _x = float('nan')
                            # Change field name!
                            field = field + '_peak'
                        elif field == 'type':
                            _x = self.peaktypes.get(peak.type, -1)
                        elif field == 'detector':
                            _x = self.detectors.get(peak.detector, -1)
                        else:
                            _x = getattr(peak, field)

                        _current_peak[field] = _x
                    # All properties added, now finish this peak
                    # The event number is necessary to join to event properties
                    _current_peak['event_number'] = event.event_number
                    peak_data.append(_current_peak)

            return peak_data
        else:
            # If event does not pass cut return empty list
            return []


class IsolatedPeaks(MultipleRowExtractor):  # pylint: disable=unused-variable
    """Returns one row per peak isolated in time

    Specifically returns properties of each individual peak.
    """
    __version__ = '0.1.2'
    extra_branches = ['peaks.left', 'peaks.right', 'peaks.n_hits',
                      'peaks.n_contributing_channels',
                      'peaks.reconstructed_positions*', 'peaks.area_decile_from_midpoint*']

    nhits_bounds = (0, float('inf'))
    width_bounds = (0, float('inf'))

    def extract_data(self, event):
        results = []
        for peak, time_to_nearest_peak in self.yield_peak(event, self.nhits_bounds, self.width_bounds):
            result = dict({x: getattr(peak, x)
                           for x in ['area', 'area_fraction_top', 'n_hits']})
            result['time_to_nearest_peak'] = time_to_nearest_peak
            result['range_50p_area'] = peak.range_area_decile[5]
            result['n_contributing_channels'] = peak.n_contributing_channels
            result['rise_time'] = - peak.area_decile_from_midpoint[1]
            result['range_90p_area'] = peak.range_area_decile[9]
            for rp in peak.reconstructed_positions:
                if rp.algorithm == 'PosRecTopPatternFit':
                    result['x'] = rp.x
                    result['y'] = rp.y
                    result['xy_gof'] = rp.goodness_of_fit
                    break
            results.append(result)

        return results

    @staticmethod
    def yield_peak(event, nhits_bounds, width_bounds):
        """Extracts a row per peak

        The peak type can be single electron and have some selection.  This is
        a generator, and yields (peak, time_to_nearest).
        """
        # Get all non-lone-hit peaks in the TPC
        peaks = [p for p in event.peaks if p.detector == 'tpc' and not p.type == 'lone_hit']
        peaks = sorted(peaks, key=lambda p: p.left)

        if not len(peaks):
            return []

        # For each of these, find the smallest gap
        lefts, rights = np.array([(p.left, p.right) for p in peaks]).T * 10 * units.ns
        gap_on_left = np.concatenate(([0], lefts[1:] - rights[:-1]))
        gap_on_right = np.concatenate((lefts[1:] - rights[:-1], [0]))
        smallest_gap = np.clip(gap_on_left, 0, gap_on_right)

        for i, peak in enumerate(peaks):
            time_to_nearest_peak = smallest_gap[i]
            if time_to_nearest_peak < 10 * units.us:
                continue
            if not (nhits_bounds[0] <= peak.n_hits < nhits_bounds[1]):
                continue
            width = peak.range_area_decile[5]
            if not (width_bounds[0] <= width < width_bounds[1]):
                continue
            yield peak, time_to_nearest_peak

class SingleElectrons(IsolatedPeaks):
    nhits_bounds = (15, 26.01)    # 26 is in
    width_bounds = (50, 450)
