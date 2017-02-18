"""Lichens grows on trees

Extend the Minitree produced DataFrames with derivative values.
"""
class Lichen(object):
    pass


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

    def pre(self, df):
        return df

    def process(self, df):
        self.pre(df)
        df = self._process(df)
        self.post(df)

    def _process(self, df):
        df[self.__class__.__name__] = (df[self.variable] > self.allowed_range[0]) & (df[self.variable] < self.allowed_range[1])
        return df

    def post(self, df):
        return df.drop('temp', 1)


class ManyLichen(Lichen):
    lichen_list = []

    def process(self, df):
        for lichen in self.lichen_list:
            df = lichen.process(df)
        return df
