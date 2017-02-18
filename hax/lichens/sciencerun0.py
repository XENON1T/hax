import numpy as np
from hax.lichen import Lichen, RangeLichen, ManyLichen

class AllCutsSR0(ManyLichen):
    def __init__(self):
        self.lichen_list = [S2Threshold(),
                            cS2Threshold(),
                            S1Threshold(),
                            Fiducial(),
                            S2AreaFractionTop(),
                            InteractionPeaksBiggest(),
                            DoubleScatterS2(),
                            Width()]



class S2Threshold(RangeLichen):
    allowed_range = (100, np.inf)
    variable = 's2'


class cS2Threshold(S2Threshold):
    variable = 'cs2'


class S1Threshold(RangeLichen):
    allowed_range = (3, 70)
    variable = 'cs1'


class S1Threshold(RangeLichen):
    allowed_range = (3, 70)
    variable = 'cs1'


class Fiducial(ManyLichen):
    def __init__(self):
        self.lichen_list = [self.Z(),
                            self.R()]

    class Z(RangeLichen):
        allowed_range = (-92, -10)
        variable = 'z'

    class R(RangeLichen):
        variable = 'temp'
        allowed_range = (0, 30)


class S2AreaFractionTop(RangeLichen):
    allowed_range = (0.6, 0.72)
    variable = 's2_area_fraction_top'


class InteractionPeaksBiggest(ManyLichen):
    def __init__(self):
        self.lichen_list = [self.S1(),
                            self.S2()]

    class S1(Lichen):
        def _process(self, df):
            return df.s1 > df.largest_other_s1

    class S2(Lichen):
        def _process(self, df):
            return df.s2 > df.largest_other_s2


class DoubleScatterS2(Lichen):
    allowed_range = (0, np.inf)
    variable = 'temp'

    def other_s2_bound(self, s2):
        return np.clip((2 * s2) ** 0.5, 70, float('inf'))

    def _process(self, df):
        return df.largest_other_s2 < self.other_s2_bound(df.s2)

class Width(ManyLichen):
    scale = 1.349
    scaleratio = 1.349 / 3.32
    sigma_0 = 0.23 / scaleratio

    def __init__(self):
        self.lichen_list = [self.WidthHigh(),
                            self.WidthLow()]

    class WidthHigh(RangeLichen):
        variable = 'temp'
        allowed_range = (-10, 0)

        def s2_width_up(self, dt):
            return (0.8 + 0.5 / 500 * dt + np.sqrt(
                (self.scale * self.sigma_0) ** 2 + self.coefficient2 * dt)) * self.scaleratio

        def pre(self, df):
            df['temp'] = df.s2_range_50p_area / 1000. - self.s2_width_up(df.drift_time / 1000.)
            return df

    class WidthLow(RangeLichen):
        variable = 'temp'
        allowed_range = (0, 10)

        def s2_width_low(self, dt):
            return (-0.4 - 0.5 / 500 * dt + np.sqrt(
                (self.scale * self.sigma_0) ** 2 + self.coefficient2 * dt)) * self.scaleratio

        def pre(self, df):
            df['temp'] = df.s2_range_50p_area / 1000. - self.s2_width_low(df.drift_time / 1000.)
            return df
