import numpy as np
import hax


class PreviousEventBasics(hax.minitrees.TreeMaker):
    """Basic information about the previous event.

    This minitree provides all columns in Basics, with a 'previous_' prefix.

    The first event of each dataset has all values NaN (since the previous event is in another dataset, or more
    likely during a time when the DAQ was off).

    This treemaker never produces any minitree files, since it only has to load the Basics file and shift it.
    (well, it makes Basics if you haven't made it yet).
    """
    __version__ = '0.0.1'
    never_store = True

    def get_data(self, dataset, event_list=None):
        # Load Basics for this dataset and shift it by 1
        data = hax.minitrees.load_single_minitree(dataset, 'Basics')
        df = data.shift(1)

        # Add previous_ prefix to all columns
        df = df.rename(columns=lambda x: 'previous_' + x)

        # Add (unshifted) event number and run number, to support merging
        df['event_number'] = data['event_number']
        df['run_number'] = data['run_number']

        # Support for event list (lame)
        if event_list is not None:
            df = df[np.in1d(df['event_number'].values, event_list)]

        return df
