from hax.minitrees import TreeMaker


class TimeDifferences(TreeMaker):
    """Compute time differences between S1s and S2s

    The convention is that these times are how much later the
    second largest signal is from the main one.  For example,
    this could be the time of the second largest S2 minus the
    time of the largest S2.  Therefore, you'll have a positive
    number if the second largest S2 came after the main S2.

    Provides:
     - dt_s1s: Time difference between two largest S1s
     - dt_s2s: Time difference between two largest S2s

    Notes:

    * Positive times means the second largest signal is later.
    * The code is intentionally not general for clarity.
    * This code does not use the interaction concept in any way!
    * 'Largest' refers to uncorrected area

    """

    extra_branches = ['peaks.hit_time_mean']
    __version__ = '0.0.1'

    def extract_data(self, event):
        # If there are no interactions at all, we can't extract anything...
        event_data = dict()

        if len(event.s1s) >= 2:
            times = [event.peaks[i].hit_time_mean for i in event.s1s]
            event_data['dt_s1s'] = times[1] - times[0]

        if len(event.s2s) >= 2:
            times = [event.peaks[i].hit_time_mean for i in event.s2s]
            event_data['dt_s2s'] = times[1] - times[0]

        return event_data
