"""
Treemakers used in the s2-only analysis
see https://github.com/XENON1T/s2only for more information
"""

import hax
import numpy as np

try:
    from s2only.common import cathode_z_range
except ImportError:
    # Don't have the s2only framework itself
    # Let's hope this value doesn't change or we remember to change it
    cathode_z_range = np.array([-100, -95])


def get_small_events(data):
    """Return rows from data corresponding to events
    that have S2 < 500 and S1 < 70 (or lack S1s altogether).
    """
    d = data[data['s2_area'] < 500]

    # Remove a very large population of events with large S1s
    # where the S2 is probably just fake nonsense after the S1 (AP, PI SE)
    d = d[
        np.isnan(d['s1_area'])
        | (d['s1_area'] < 70)]

    return d


class BigBeforeSmall(hax.minitrees.MultipleRowExtractor):
    """Return very basic info about all large S2s closely preceding a small-S2 event
    "Large" means: S2 area > max(300 PE, 0.1 * largest_s2_area)
    "Small" is defined by get_small_events
    """
    __version__ = '0.0.7'
    extra_branches = ['peaks.reconstructed_positions*',
                      'peaks.left', 'peaks.center_time']

    def get_data(self, dataset, event_list=None):
        if event_list is not None:
            raise RuntimeError("BigBeforeSmall computes its own event list")

        # Notice we use bypass_blinding here: we want *all* large events
        # before small ones, even those that are candidates for funny processes
        # like DEC and DBD.
        # (looking back at this, this is probably not necessary, if I recall correctly;
        # blinding only gets activated if you load the correction minitree
        # which we do not)
        data = hax.minitrees.load_single_dataset(
            dataset,
            ['Fundamentals', 'LargestPeakProperties'],
            bypass_blinding=True)[0]

        small_events = get_small_events(data)

        # Find all events for which a small event follows in 1s
        # This is about 10% of all events in a normal background dataset
        dts = (small_events['event_time'].values.reshape(1, -1) -
               data['event_time'].values.reshape(-1, 1))
        close_mask = np.any((0 < dts) & (dts < int(1e9)), axis=1)
        event_list = np.where(close_mask)[0]

        return hax.minitrees.MultipleRowExtractor.get_data(self, dataset, event_list=event_list)

    def extract_data(self, event):
        # Get the largest S2s (up to five) satisfying
        #   area > max(300 PE, 0.1 * largest_s2_area)
        s2s = [p
               for p in event.peaks
               if p.type == 's2']
        if not len(s2s):
            # Hm, that's somewhat interesting...
            # Maybe have a look at these sometime.
            return []

        s2_areas = np.array([p.area for p in s2s])
        order = np.argsort(s2_areas)[::-1]
        s2s = [s2s[j] for j in order]

        largest_s2_area = s2_areas.max()
        s2s = [p
               for p in s2s
               if p.area > max(300, largest_s2_area * 0.1)][:5]

        result = []
        for i, p in enumerate(s2s):
            for rp in p.reconstructed_positions:
                if rp.algorithm == 'PosRecTopPatternFit':
                    x, y = rp.x, rp.y
                    break
            else:
                x, y = float('nan'), float('nan')

            if len(event.interactions):
                main_s1_time = (
                    int(event.start_time) +
                    int(event.peaks[event.interactions[0].s1].center_time))
            else:
                main_s1_time = -1

            result.append(dict(
                time=int(event.start_time) + int(p.center_time),
                area=p.area,
                x=x,
                y=y,
                range_50p_area=p.range_area_decile[5],
                nth_largest_s2_in_event=i,
                largest_s2_area_in_event=s2s[0].area,
                main_s1_time=main_s1_time,
            ))

        return result


class LoneS2Info(hax.minitrees.TreeMaker):
    """Treemaker for S2-only analysis: returns info about the largest and second largest S2 in the event.
    High-energy (S2 > 5000 PE) events are automatically skipped to speed up treemaking.

    This treemaker implements the s2-only blinding protocol.
     - Version 0.2.0 (November 2018): unblind events ending in 0, 3, 7; as well as
       events whose z indicates a cathode interaction.
     - Version 0.1.0 (Early August 2018): unblind events whose number ends in 0.
    """
    extra_branches = ['peaks.reconstructed_positions*',
                      'peaks.area_decile_from_midpoint[11]',
                      'peaks.left', 'peaks.center_time']

    __version__ = '0.2.1'

    peak_properties_to_get = 'area area_fraction_top left n_hits center_time'.split()

    def get_properties(self, peak, prefix=''):
        """Return dictionary with peak properties, keys prefixed with prefix
        if peak is None, will return nans for all values
        """
        result = {field: getattr(peak, field) for field in self.peak_properties_to_get}
        result['range_50p_area'] = peak.range_area_decile[5]
        result['range_90p_area'] = peak.range_area_decile[9]
        result['rise_time'] = -peak.area_decile_from_midpoint[1]
        for rp in peak.reconstructed_positions:
            if rp.algorithm == 'PosRecTopPatternFit':
                result['x'] = rp.x
                result['y'] = rp.y
                result['pattern_fit'] = rp.goodness_of_fit
            elif rp.algorithm == 'PosRecNeuralNet':
                result['x_nn'] = rp.x
                result['y_nn'] = rp.y
                result['pattern_fit_nn'] = rp.goodness_of_fit
        return {prefix + k: v for k, v in result.items()}

    def get_data(self, dataset, event_list=None):
        try:
            run_number = hax.runs.get_run_number(dataset)
            run_data = hax.runs.datasets.query('number == %d' % run_number).iloc[0]
        except Exception as e:
            self.log.warning("Exception while trying to find run %s: %s.\n"
                             "I assume this is an MC dataset: NOT blinding!" % (dataset, str(e)))
            self.blind = False
        else:
            self.blind = run_data.reader__ini__name.startswith('background')

        return super().get_data(dataset, event_list)

    def extract_data(self, event):
        if self.blind:
            # Consider blinding for this dataset (it's background)
            if event.event_number % 10 not in (0, 3, 7):
                # NOT in the training set. Probably have to blind this.
                if not len(event.interactions):
                    # No valid S1-S2 interactions: certainly blind this
                    return dict()
                else:
                    # There is an interaction. Is it a cathode event?
                    z = event.interactions[0].z
                    if not (cathode_z_range[0] < z < cathode_z_range[1]):
                        # NO, so blind this
                        return dict()

        # Throw out big events quickly. Not sure if it helps much...
        if len(event.interactions):
            main_s2 = event.peaks[event.interactions[0].s2]
            if main_s2.area > 5000:
                return {}

        # Get S2s sorted descending by area
        s2s = [p for p in event.peaks if p.type == 's2']
        s2_areas = np.array([p.area for p in s2s])
        order = np.argsort(s2_areas)[::-1]
        s2s = [s2s[j] for j in order]
        s2_areas = s2_areas[order]

        if len(s2s) == 0:
            return {}

        # Restrict to low-energy events to save computation time and disk space
        if s2_areas[0] > 5000:
            return {}

        result = self.get_properties(s2s[0], prefix='s2_')

        # Get area before s20
        # This was used in SR0; we keep it around anyway
        area_before_main_s2 = [p.area
                               for p in event.peaks
                               if p.detector == 'tpc' and p.left < s2s[0].left]
        if len(area_before_main_s2):
            result['area_before_largest_s2'] = sum(area_before_main_s2)
        else:
            result['area_before_largest_s2'] = 0

        # Pretrigger cut quantities
        pretrigger_s2s = [p for p in s2s if p.center_time < 1e6]
        result['n_pretrigger_s2s'] = len(pretrigger_s2s)
        if len(pretrigger_s2s):
            result['max_pretrigger_s2_area'] = max([p.area
                                                    for p in pretrigger_s2s])

        # Properties of the second largest S2
        # Currently not used used
        if len(s2s) > 1:
            result.update(self.get_properties(s2s[1], prefix='s2_1_'))

        # Find out if there is any S1 pointing to a cathode interaction
        # Note we use ints rather than True/False, apparently something
        # in the minitree chain doesn't like python bools..
        result['cathode_tag'] = 0
        for ia in event.interactions:
            if cathode_z_range[0] < ia.z < cathode_z_range[1]:
                result['cathode_tag'] = 1
                break

        return result
