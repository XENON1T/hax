import numpy as np
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
        return {"trigger_" + k: getattr(ts, k)
                for k in [a[0] for a in TriggerSignal.get_fields_data(TriggerSignal())]}


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
    """
    __version__ = '0.0.12'
    pax_version_independent = False          # Now that we include S2 area it's not

    aqm_labels = ['muon_veto_trigger', 'busy_on', 'hev_on', 'busy_off', 'hev_off', 'busy', 'hev']

    def get_data(self, dataset, event_list=None):
        aqm_pulses = get_aqm_pulses(dataset)

        # Load the fundamentals and totalproperties minitree
        # Yes, minitrees loading other minitrees, the fun has begun :-)
        event_data = hax.minitrees.load_single_dataset(dataset, ['Fundamentals', 'TotalProperties', 'LargestPeakProperties'])[0]
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
            prev = 'previous_%s' % label
            nxt = 'next_%s' % label

            # Find the first object (at or) after t
            if label == 'event':
                i = event.event_number
            else:
                i = np.searchsorted(x, t)   # Index in x of the first value >= t

            if i == 0:
                result[prev] = 368395560000000000 # ~- int(np.inf).... 100th Birthday
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
                try:
                    assert label != 'event'
                except:
                    print(x[i])
                    print(t)
                    raise
                    
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
                result[tnr] = - result[prev]
            else:
                result[tnr] = result[nxt]

        # Need special logic for nearest s2 area
        if result['nearest_event'] == result['previous_event']:
            result['nearest_s2_area'] = result['previous_s2_area']
        else:
            result['nearest_s2_area'] = result['next_s2_area']

        return result

class TailCut(hax.minitrees.TreeMaker):

    __version__ = '0.0.2'

    def get_data(self, dataset, event_list=None):

        self.event_data = hax.minitrees.load_single_dataset(
            dataset, ['Fundamentals', 'TotalProperties', 'LargestPeakProperties'])[0]
        self.event_data['center_time'] = self.event_data.event_time + self.event_data.event_duration // 2
        self.center_time = self.event_data.center_time.values
        self.s2_area = self.event_data.s2_area.values
        self.look_back=50
        return hax.minitrees.TreeMaker.get_data(self, dataset, event_list)

    def extract_data(self, event):

        i = event.event_number
        tnow = (event.start_time + event.stop_time) // 2

        ct = self.center_time[i-self.look_back:i]
        ls2 = self.s2_area[i-self.look_back:i]
        try:
            mp = max([ls2[i]/(tnow-ct[i]) for i in range(len(ct))])
        except:
            mp = None
        return {"s2_over_tdiff": mp}
