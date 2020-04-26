import numpy as np
import pandas as pd
from pax.datastructure import TriggerSignal

import hax
from hax.minitrees import TreeMaker
from hax.trigger_data import get_aqm_pulses


class LargestTriggeringSignal(TreeMaker):
    """Information on the largest trigger signal with the trigger flag set in the event

    Provides:
     - trigger_*, where * is any of the attributes of datastructure.TriggerSignal
    """
    branch_selection = ['trigger_signals*', 'event_number']
    pax_version_independent = True
    __version__ = '0.0.3'

    def extract_data(self, event):
        tss = [t for t in event.trigger_signals if t.trigger]
        if not len(tss):
            # No trigger signal! This indicates an error in the trigger's signal grouping,
            # See https://github.com/XENON1T/pax/issues/344
            return dict()
        ts = tss[int(np.argmax([t.n_pulses for t in tss]))]
        return {"trigger_" + k: getattr(ts, k) for k in [a[0] for a in TriggerSignal.get_fields_data(TriggerSignal())]}


class Proximity(hax.minitrees.TreeMaker):
    """Information on the proximity of other events and acquisition monitor signals (e.g. busy and muon veto trigger)
        Provides:
         - previous_x: Time (in ns) between the time center of the event and the previous x (see below for various x).
           This also considers any x inside the event.
         - next_x: same, for time to next x
         - nearest_x: Time to the nearest x. NB: If the nearest x is in the past, this time is negative!
            x denotes the object of interest, and could be either:
         - muon_veto_trigger.
         - busy_x: a busy-on or busy-off signal
         - hev_x: a high-energy veto on or -off signal
         - event: any event (excluding itself :-)
         - 1e5_pe_event: any event with total peak area > 1e5 pe (excluding itself)
         - 3e5_pe_event: "" 3e5 pe "
         - 1e6_pe_event: "" 1e6 pe "
         - s2_area: Area of main s2 in event
         - All the information about the muon_veto_trigger are calculated with respect to the TPC trigger and
         - not to the middle of the event.
    """
    __version__ = '0.1.0'
    pax_version_independent = False          # Now that we include S2 area it's not

    aqm_labels = [
        'muon_veto_trigger',
        'busy_on',
        'hev_on',
        'busy_off',
        'hev_off',
        'busy',
        'hev']

    def bad_mv_triggers(self, aqm_pulses, min_time=500):
        """
        get me the indices of the sync signal in the MV_trigger sent to the TPC
        """
        indices_mv = list()

        for tsync in aqm_pulses['mv_sync']:
            # This dict will have the info of the unwanted MV_triggers sent to TPC
            # maybe there is a better way to save those!!!
            list_times = dict()
            for j, tmvt in enumerate(aqm_pulses['muon_veto_trigger']):
                list_times[j] = np.abs(tsync - tmvt)
            # these are the indices of the MV_trigger that needs to be taken away
            indices_mv.extend([key for key, value in list_times.items() if value < min_time])
        return indices_mv

    def select_physical_pulses(self, aqm_pulses, ap_time=20000):
        """
        get rid of afterpulses from the MV data of course!!!!!
        """
        # lets get rid of the trigger that comes from the mv_sync signal
        mask = np.ones(aqm_pulses["muon_veto_trigger"].shape, dtype=bool)
        mask[self.bad_mv_triggers(aqm_pulses)] = False
        list_indicies = list()
        for l, times in enumerate(np.diff(aqm_pulses["muon_veto_trigger"][mask])):
            if times < ap_time:  # get rid of pulses that comes within 20mu sec.
                list_indicies.append(l+1)

        aqm_pulses["muon_veto_trigger"] = np.delete(aqm_pulses["muon_veto_trigger"][mask], list_indicies)

        return aqm_pulses

    def get_data(self, dataset, event_list=None):
        aqm_pulses = self.select_physical_pulses(get_aqm_pulses(dataset))

        # Load the fundamentals and totalproperties minitree
        # Yes, minitrees loading other minitrees, the fun has begun :-)
        event_data = hax.minitrees.load_single_dataset(
            dataset, ['Fundamentals', 'TotalProperties', 'LargestPeakProperties'])[0]
        # Note integer division here, not optional: float arithmetic is too inprecise
        # (fortuately our digitizer sampling resolution is an even number of nanoseconds...)
        event_data['center_time'] = event_data.event_time + event_data.event_duration // 2

        # Build the various lists of 2-tuples (label, times) to search through
        self.search_these = ([(x, aqm_pulses[x]) for x in self.aqm_labels] +
                             [(boundary + 'pe_event', event_data[event_data.total_peak_area >
                                                                 eval(boundary)].center_time.values)
                              for boundary in ['1e5', '3e5', '1e6']] +
                             [('event', event_data.center_time.values)]
                             )
        self.s2s = event_data.s2_area.values

        # super() does not play nice with dask computations, for some reason
        return hax.minitrees.TreeMaker.get_data(self, dataset, event_list)

    def extract_data(self, event):
        # Again, integer division is not optional here!
        t = (event.start_time + event.stop_time) // 2
        result = dict()

        for label, x in self.search_these:
            # we want to use for the MV the time with respect to the trigger in the TPC
            if label == "muon_veto_trigger":
                t_corr = np.int64(event.stop_time - event.start_time) // 2 - np.int64(10**6)
            else:
                t_corr = np.int64(0)

            prev = 'previous_%s' % label
            nxt = 'next_%s' % label

            # Find the first object (at or) after t
            if label == 'event':
                i = event.event_number
            else:
                # Index in x of the first value >= t
                i = np.searchsorted(x, t)

            if i == 0:
                # ~- int(np.inf).... 100th Birthday
                result[prev] = 368395560000000000
                if label == 'event':
                    result['previous_s2_area'] = 368395560000000000
            else:
                result[prev] = t - x[i - 1]
                if label == 'event':
                    result['previous_s2_area'] = self.s2s[i - 1]

            # Check if the sought-after object is exactly at t
            # This is always true if label == 'event', only very rarely for aqm signals.
            # (but then we don't want to advance the 'next' index, it's important that the signal
            #  is right in the center!)
            if i != len(x) and x[i] == t and label not in self.aqm_labels:
                # The real 'next' is one further:
                i += 1
            else:
                assert label != 'event'

            if i == len(x):
                result[nxt] = 368395560000000000
                if label == 'event':
                    result['next_s2_area'] = 368395560000000000
            else:
                result[nxt] = x[i] - t
                if label == 'event':
                    result['next_s2_area'] = self.s2s[i]

            # Include the 'nearest' variable. This is negative if the nearest sought-after object
            # is in the past.
            tnr = 'nearest_' + label
            if result[nxt] > result[prev]:
                result[tnr] = - result[prev] + t_corr
            else:
                result[tnr] = result[nxt] + t_corr

        # Need special logic for nearest s2 area
        if result['nearest_event'] == result['previous_event']:
            result['nearest_s2_area'] = result['previous_s2_area']
        else:
            result['nearest_s2_area'] = result['next_s2_area']

        return result


# new minitrees
class TailCut(hax.minitrees.TreeMaker):
    __version__ = '0.3'
    never_store = True
    def get_data(self, dataset, event_list=None):
        look_back = 50
        # Load Fundamentals and LargestPeakProperties
        # Using load_single_dataset instead of load will ensure no blindding cut is applies
        data, _ = hax.minitrees.load_single_dataset(dataset, ['Fundamentals', LoneSignals])#'LargestPeakProperties'])
        if data.empty:
            return pd.DataFrame({})
        # Get largest S2 in the event (or 0, if no S2 was found)
        s2 = data['s2_0_area'].values
        #s2 = data['total_peak_area'].values
        s2_x = data['s2_0_x_nn_tf'].values
        s2_y = data['s2_0_y_nn_tf'].values

        s2_other = data['s2_1_area'].values
        s2_x_other = data['s2_1_x_nn_tf'].values
        s2_y_other = data['s2_1_y_nn_tf'].values
        
        s2[np.isnan(s2)] = 0
        s2_other[np.isnan(s2_other)] = 0
        # Get the center time of the event
        t = data['event_time'].values # + data['event_duration'].values/2  #why doesn't it work????
        tdiff_min = np.zeros(len(t))
        tdiff_min[0] = 1e18
        for i in range(1, len(t)):
            tdiff_min[i] = t[i] - t[i-1]
        # Compute S2/tdif for each value of lookback (from 1 to look_back, inclusive)
        # Allocates n_events * look_back floats; should not be a problem unless look_back is insane number.
        s2_over_tdiff_lookback = np.zeros((len(t), look_back + 1))
        s2_area_lookback = np.zeros((len(t), look_back + 1))
        se_probabal_coor = np.zeros((len(t), 2, look_back + 1))
        
        s2_other_over_tdiff_lookback = np.zeros((len(t), look_back + 1))
        s2_other_area_lookback = np.zeros((len(t), look_back + 1))
        se_probabal_coor_other = np.zeros((len(t), 2, look_back + 1))
        
        for i in range(1, look_back + 1):
            s2_over_tdiff_lookback[i:, i] = s2[:-i]/(t[i:] - t[:-i])
            s2_area_lookback[i:, i] = s2[:-i]
            se_probabal_coor[i:, 0, i] = s2_x[i:] - s2_x[:-i]
            se_probabal_coor[i:, 1, i] = s2_y[i:] - s2_y[:-i]
            s2_other_over_tdiff_lookback[i:, i] = s2_other[:-i]/(t[i:] - t[:-i])
            s2_other_area_lookback[i:, i] = s2_other[:-i]
            se_probabal_coor_other[i:, 0, i] = s2_x_other[i:] - s2_x[:-i]
            se_probabal_coor_other[i:, 1, i] = s2_y_other[i:] - s2_y[:-i]

            
        # Which event
        tailcut_set_by = np.argmax(s2_over_tdiff_lookback, axis=1)
        result = s2_over_tdiff_lookback.max(axis=1)
        # Which event for se
        distance_cut = np.sqrt(np.sum(se_probabal_coor**2, axis=1))
        distance_cut = list(map(lambda d, take: d[take], distance_cut, tailcut_set_by))
        # s2_over_tdiff_lookback[distance_cut>10] = 0
        tailcut_other_set_by = np.argmax(s2_other_over_tdiff_lookback, axis=1)
        result_other = s2_other_over_tdiff_lookback.max(axis=1)

        distance_cut_other = np.sqrt(np.sum(se_probabal_coor_other**2, axis=1))
        distance_cut_other = list(map(lambda d, take: d[take], distance_cut_other, tailcut_other_set_by))

        return pd.DataFrame(dict(event_number=data['event_number'],
            run_number=data['run_number'],
            s2_over_tdiff=result,
            tdiff_min=tdiff_min,
            tailcut_set_by=tailcut_set_by,
            distance_cut = distance_cut,
            s2_other_over_tdiff=result_other,
            tailcut_other_set_by=tailcut_other_set_by,
            distance_cut_other = distance_cut_other,
                                            ))