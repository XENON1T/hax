"""Lichens grows on trees

Extend the Minitree produced DataFrames with derivative values.
"""
class Lichen(object):
    def pre(self, df):
        return df

    def process(self, df):
        df = self.pre(df)
        df = self._process(df)
        df = self.post(df)
        return df

    def _process(self, df):
        raise NotImplementedError()

    def post(self, df):
        if 'temp' in df.columns:
            return df.drop('temp', 1)
        return df


class RangeLichen(Lichen):
    allowed_range = None  # tuple of min then max
    variable = None  # variable name in DataFrame

    def get_allowed_range(self):
        if self.allowed_range is None:
            raise NotImplemented()

    def get_min(self):
        if self.variable is None:
            raise NotImplemented()
        return self.allowed_range[0]

    def get_max(self):
        if self.variable is None:
            raise NotImplemented()
        return self.allowed_range[0]

    def _process(self, df):
        df[self.__class__.__name__] = (df[self.variable] > self.allowed_range[0]) & (df[self.variable] < self.allowed_range[1])
        return df


class ManyLichen(Lichen):
    lichen_list = []

    def process(self, df):
        all_cuts_bool = None
        for lichen in self.lichen_list:
            df = lichen.process(df)

            if all_cuts_bool is None:
                all_cuts_bool = df[lichen.__class__.__name__]
            else:
                all_cuts_bool = all_cuts_bool & df[lichen.__class__.__name__]

        df[self.__class__.__name__] = all_cuts_bool
        return df
