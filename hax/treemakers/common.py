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
     * 'corrected' refers to applying all available position- and/or saturation corrections
       (see https://github.com/XENON1T/pax/blob/master/pax/plugins/interaction_processing/BuildInteractions.py#L105)
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
        else:
            # TODO: XENON1T support
            dataset_number = 0

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


class PeakProperties(TreeMaker):
    """Largest peak properties for each type and for all peaks.

    """
    extra_branches = ['peaks.area_fraction_top',
                      'peaks.bottom_hitpattern_spread',
                      'peaks.hit_time_std',
                      'peaks.n_hits',
                      'peaks.area',
                      'peaks.n_saturated_channels']
    __version__ = '0.0.1'

    def extract_data(self, event):  # This runs on each event
        # 'values' is returned once filled and each field defaults to zero.
        values = defaultdict(float)

        # Store the start time of the event
        values['time'] = event.start_time

        peaks_values = {}  # type name -> index

        # If no peaks, just continue
        if event.peaks.size() == 0:
            return values

        for i, peak in enumerate(event.peaks):
            if peak.type not in peaks_values:
                peaks_values[peak.type] = i

            type_key = 'largest_%s' % peak.type
            biggest_peak_this_type = event.peaks[peaks_values[type_key]]

            if peak.area > biggest_peak_this_type.area:
                peaks_values[type_key] = i

            if 'largest_peak' not in peaks_values:
                peaks_values['largest_peak'] = i
            if peak.area > event.peaks[peaks_values['largest_peak']].area:
                peaks_values['largest_peak'] = i

        for name, index in peaks_values.items():
            peak = event.peaks[index]
            # The store each peak field we want in 'values'
            for field in self.extra_branches:
                field = field[6:]
                field_name = '%s_%s' % (name,
                                        field)
                values[field_name] = getattr(peak,
                                             field)

        return values