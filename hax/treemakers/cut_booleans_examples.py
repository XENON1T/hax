from hax.minitrees import TreeMaker


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
