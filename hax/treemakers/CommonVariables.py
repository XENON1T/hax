"""Standard variables for most analyses
"""

from hax.minitrees import TreeMaker
from collections import defaultdict


class Basics(TreeMaker):
    """Basic minitree containing variables needed in almost every basic analysis.

    Provides:
     - event_number: Event number within the dataset
     - dataset_number: Numerical representation of dataset id, e.g. xe100_120402_2000 -> 1204022000
     - s1: The uncorrected area in pe of the main interaction's S1
     - s2: The uncorrected area in pe of the main interaction's S2
     - cs1: The corrected area in pe of the main interaction's S1
     - cs2: The corrected area in pe of the main interaction's S2
     - x: The x-position of the main interaction (primary algorithm chosen by pax, currently TopPatternFit)
     - y: The y-position of the main interaction
     - z: The z-position of the main interaction (computed by pax using configured drift velocity)
     - drift_time: The drift time in ns (pax units) of the main interaction
     - s1_area_fraction_top: The fraction of uncorrected area in the main interaction's S1 seen by the top array
     - s2_area_fraction_top: The fraction of uncorrected area in the main interaction's S2 seen by the top array
     - largest_other_s1: The uncorrected area in pe of the largest S1 in the TPC not in the main interaction
     - largest_other_s2: The uncorrected area in pe of the largest S2 in the TPC not in the main interaction
     - largest_veto: The uncorrected area in pe of the largest non-lone_hit peak in the veto
     - largest_unknown: The largest TPC peak of type 'unknown'
     - largest_coincidence: The largest TPC peak of type 'coincidence'

    Notes:
     * 'largest' refers to uncorrected area.
     * 'uncorrected' refers to the area in pe without applying ANY position- or saturation corrections.
     * 'corrected' refers to applying ALL position- and/or saturation corrections (depending on pax configuration used)
     * 'main interaction' is event.interactions[0], which is determined by pax
                          (currently just the largest S1 + largest S2 after it)

    """
    __version__ = '0.0.2'
    extra_branches = ['dataset_name']

    def extract_data(self, event):
        # Convert from XENON100 dataset name (like xe100_120402_2000_000000.xed) to number
        dsetname = event.dataset_name
        if dsetname.endswith('.xed'):
            filename = dsetname.split("/")[-1]
            _, date, time, _ = filename.split('_')
            dataset_number = int(date) * 1e4 + int(time)

        event_data = dict(event_number=event.event_number,
                          event_time=event.start_time,
                          dataset_number=dataset_number)

        # Detect events without at least one S1 + S2 pair immediatly
        # We cannot even fill the basic variables for these
        if len(event.interactions) != 0:

            # Extract basic data: useful in any analysis
            interaction = event.interactions[0]
            s1 = event.peaks[interaction.s1]
            s2 = event.peaks[interaction.s2]
            event_data.update(dict(s1=s1.area,
                                   s2=s2.area,
                                   s1_area_fraction_top=s1.area_fraction_top,
                                   s2_area_fraction_top=s2.area_fraction_top,
                                   cs1=s1.area * interaction.s1_area_correction,
                                   cs2=s2.area * interaction.s2_area_correction,
                                   x=interaction.x,
                                   y=interaction.y,
                                   z=interaction.z,
                                   drift_time=interaction.drift_time))

            exclude_peaks_from_double_sc = [interaction.s1, interaction.s2]
        else:
            exclude_peaks_from_double_sc = []

        # Add double scatter cut data: area of the second largest peak of each peak type.
        # Note we explicitly differentiate non-tpc peaks here; in interaction objects this is already taken care of.
        largest_area_of_type = defaultdict(float)
        for i, p in enumerate(event.peaks):
            if i in exclude_peaks_from_double_sc:
                continue
            if p.detector == 'tpc':
                peak_type = p.type
            else:
                # Lump all non-lone-hit veto peaks together as 'veto'
                if p.type == 'lone_hit':
                    peak_type = 'lone_hit_%s' % p.detector    # Will not be saved
                else:
                    peak_type = p.detector
            largest_area_of_type[peak_type] = max(p.area, largest_area_of_type[peak_type])

        event_data.update(dict(largest_other_s1=largest_area_of_type['s1'],
                               largest_other_s2=largest_area_of_type['s2'],
                               largest_veto=largest_area_of_type['veto'],
                               largest_unknown=largest_area_of_type['unknown'],
                               largest_coincidence=largest_area_of_type['coincidence']))

        return event_data

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

    """

    extra_branches = ['peaks.hit_time_mean']
    __version__ = '0.0.1'


    def extract_data(self, event):
        # If there are no interactions at all, we can't extract anything...
        event_data = dict()

        # Loop over S1s
        if len(event.s1s) >= 2:
            times = [event.peaks[i].hit_time_mean for i in event.s1s]
            dt = times[1] - times[0]
            event_data.update(dict(dt_s1s=dt))

        # Loop over S2s
        if len(event.s2s) >= 2:
            times = [event.peaks[i].hit_time_mean for i in event.s2s]
            dt = times[1] - times[0]
            event_data.update(dict(dt_s2s=dt))

        return event_data

class Widths(TreeMaker):
    """Compute width metrics for main S1 and main S2

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

    """

    extra_branches = ['peaks.range_area_decile']
    __version__ = '0.0.1'


    def extract_data(self, event):
        # If there are no interactions at all, we can't extract anything...
        event_data = dict()

        # Loop over S1s
        if len(event.s1s) >= 2:
            times = [event.peaks[i].hit_time_mean for i in event.s1s]
            dt = times[1] - times[0]
            event_data.update(dict(dt_s1s=dt))

        # Loop over S2s
        if len(event.s2s) >= 2:
            times = [event.peaks[i].hit_time_mean for i in event.s2s]
            dt = times[1] - times[0]
            event_data.update(dict(dt_s2s=dt))

        return event_data


        if len(event.interactions) != 0:

            # Extract basic data: useful in any analysis
            interaction = event.interactions[0]
            s1 = event.peaks[interaction.s1]
            s2 = event.peaks[interaction.s2]

class EnergyCut(TreeMaker):
    """S1 and S2 size cut booleans

    Require that the S1 and S2 be large enough.

    Provides:
     - pass_s1_area_cut: S1 bigger than 1 pe
     - pass_s2_area_cut: S2 bigger than 150 pe

    Notes:

    * This only cuts signals that are too small.

    """

    __version__ = '0.0.1'


    def extract_data(self, event):
        # If there are no interactions at all, we can't extract anything...
        event_data = dict()

        good_s1 = False
        good_s2 = False

        if len(event.interactions) != 0:
            # Extract basic data: useful in any analysis
            interaction = event.interactions[0]

            s1 = event.peaks[interaction.s1]
            s2 = event.peaks[interaction.s2]

            if s1.area > 1:
                good_s1 = True
            if s2.area > 150:
                good_s2 = True

        return dict(pass_s1_area_cut=good_s1,
                    pass_s2_area_cut=good_s2)

