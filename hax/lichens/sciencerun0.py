import numpy as np
from pax import units, configuration
PAX_CONFIG = configuration.load_configuration('XENON1T')
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
    allowed_range = (100, np.inf)
    variable = 'cs2'


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

        def pre(self, df):
            df[self.variable] = np.sqrt(df['x']*df['x'] + df['y']*df['y'])
            return df

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
            df[self.__class__.__name__] = df.s1 > df.largest_other_s1
            return df

    class S2(Lichen):
        def _process(self, df):
            df[self.__class__.__name__] = df.s2 > df.largest_other_s2
            return df


class DoubleScatterS2(Lichen):
    allowed_range = (0, np.inf)
    variable = 'temp'

    def other_s2_bound(self, s2):
        return np.clip((2 * s2) ** 0.5, 70, float('inf'))

    def _process(self, df):
        df[self.__class__.__name__] = df.largest_other_s2 < self.other_s2_bound(df.s2)
        return df

class Width(ManyLichen):

    
    def __init__(self):
        self.lichen_list = [self.WidthHigh(),
                            self.WidthLow()]

    def s2_width_model(self, z):
        v_drift = PAX_CONFIG['DEFAULT']['drift_velocity_liquid']
        diffusion_constant = PAX_CONFIG['WaveformSimulator']['diffusion_constant_liquid']
        w0 = 304 * units.ns
        return np.sqrt(w0**2 - 3.6395 * diffusion_constant * z / v_drift**3)


    def subpre(self, df):
        # relative_s2_width
        df['temp'] = df['s2_range_50p_area'] / Width.s2_width_model(self, df['z'])
        return df

    def relative_s2_width_bounds(s2, kind='high'):
        x = 0.3 * np.log10(np.clip(s2, 150, 7000))
        if kind == 'high':
            return 2.3 - x
        elif kind == 'low':
            return -0.3 + x
        raise ValueError("kind must be high or low")
    

    class WidthHigh(Lichen):
        def pre(self, df):
            return Width.subpre(self, df)

        def _process(self, df):
            df[self.__class__.__name__] = (df.temp <= Width.relative_s2_width_bounds(df.s2,
                                                                                     kind='high'))
            return df

    class WidthLow(RangeLichen):
        def pre(self, df):
            return Width.subpre(self, df)

        def _process(self, df):
            df[self.__class__.__name__] = (Width.relative_s2_width_bounds(df.s2,
                                                                           kind='low') <= df.temp)
            return df
