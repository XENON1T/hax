"""Helper functions for doing cuts/selections on dataframes,
while printing out the passthrough info.
"""
import pandas as pd
import numpy as np
import dask
import dask.dataframe

# Unlike most hax modules, this doesn't require init()
import hax

import logging
log = logging.getLogger('hax.cuts')

UNNAMED_DESCRIPTION = 'Unnamed'

##
# Cut history tracking
##


def history(d):
    """Return pandas dataframe describing cuts history on dataframe."""
    if not hasattr(d, 'cut_history'):
        raise ValueError("Cut history for this data not available.")
    hist = pd.DataFrame(d.cut_history, columns=['selection_desc', 'n_before', 'n_after'])
    hist['n_removed'] = hist.n_before - hist.n_after
    hist['fraction_passed'] = hist.n_after / hist.n_before
    hist['cumulative_fraction_left'] = hist.n_after / hist.iloc[0].n_before
    return hist


def _get_history(d):
    if not hasattr(d, 'cut_history'):
        return []
    return d.cut_history


def record_combined_histories(d, partial_histories, quiet=None):
    """Record history for dataframe d by combining list of dictionaries partial_histories"""
    if quiet is None:
        quiet = not hax.config.get('print_passthrough_info', False)

    new_history = []
    # Loop over cuts
    for cut_i, cut in enumerate(partial_histories[0]):
        q = dict(selection_desc=cut['selection_desc'],
                 n_before=sum([q[cut_i]['n_before'] for q in partial_histories]),
                 n_after=sum([q[cut_i]['n_after'] for q in partial_histories]))
        if not quiet:
            print(passthrough_message(q))
        new_history.append(q)
    d.cut_history = new_history

##
# Cut helper functions
##


def passthrough_message(passthrough_dict):
    """Prints passthrough info given dictionary with selection_desc, n_before, n_after"""
    desc = passthrough_dict['selection_desc']
    n_before = passthrough_dict['n_before']
    n_after = passthrough_dict['n_after']
    if n_before == 0:
        return "%s selection: nothing done since dataframe is already empty" % desc
    return "%s selection: %d rows removed (%0.2f%% passed)" % (
        desc, n_before - n_after, n_after / n_before * 100)


def selection(d, bools, desc=UNNAMED_DESCRIPTION, return_passthrough_info=False, quiet=None, _invert=False,
              force_repeat=False):
    """Returns d[bools], print out passthrough info.
     - data on which to perform the selection (pandas dataframe)
     - bools: boolean array of same length as d. If True, row will be in selection returned.
     - return_passthrough_info: if True (default False), return d[bools], len_d_before, len_d_now instead
     - quiet: prints passthrough info if False, not if True.
              The default is controlled by the hax init option 'print_passthrough_info'
     - _invert: inverts bools before applying them
     - force_repeat: do the selection even if a cut with an identical description has already been performed.
    """
    if quiet is None:
        quiet = not hax.config.get('print_passthrough_info', False)

    # The last part of the function has two entry points, so we need to call this instead of return:
    def get_return_value():
        if return_passthrough_info:
            return d, n_before, n_now
        return d

    if isinstance(d, dask.dataframe.DataFrame):
        # Cuts history tracking for delayed computations not yet implemented
        n_before = float('nan')
        n_now = float('nan')
        d = d[bools]
        if not quiet:
            print("%s selection readied for delayed evaluation" % desc)
        return get_return_value()

    prev_cuts = _get_history(d)
    n_before = n_now = len(d)

    if desc != UNNAMED_DESCRIPTION and not force_repeat:
        # Check if this cut has already been done
        for c in prev_cuts:
            if c['selection_desc'] == desc:
                log.debug("%s selection already performed on this data; cut skipped. Use force_repeat=True to repeat."
                          "Showing historical passthrough info." % desc)
                if not quiet:
                    print(passthrough_message(c))
                return get_return_value()

    # Invert if needed
    if _invert:
        bools = True ^ bools

    # Apply the selection
    d = d[bools]

    # Print and track the passthrough infos
    n_now = len(d)
    passthrough_dict = dict(selection_desc=desc, n_before=n_before, n_after=n_now)
    if not quiet:
        print(passthrough_message(passthrough_dict))
    d.cut_history = prev_cuts + [passthrough_dict]

    return get_return_value()


def cut(d, bools, **kwargs):
    """Same as do_selection, with bools inverted. That is, specify which rows you do NOT want to select."""
    return selection(d, True ^ bools, **kwargs)


def notnan(d, axis, **kwargs):
    """Require that d[axis] is not NaN. See selection for options and return value."""
    kwargs.setdefault('desc', '%s not NaN' % axis)
    return selection(d, d[axis].notnull(), **kwargs)


def isfinite(d, axis, **kwargs):
    """Require d[axis] finite. See selection for options and return value."""
    kwargs.setdefault('desc', 'Finite %s' % axis)
    if isinstance(d, dask.dataframe.DataFrame):
        raise NotImplementedError(
            "isfinite not yet implemented for delayed computations. "
            "Maybe cuts.notnan suffices?")
    return selection(d, np.isfinite(d[axis]), **kwargs)


def above(d, axis, threshold, **kwargs):
    """Require d[axis] > threshold. See selection for options and return value."""
    kwargs.setdefault('desc', '%s above %s' % (axis, threshold))
    return selection(d, d[axis] > threshold, **kwargs)


def below(d, axis, threshold, **kwargs):
    """Require d[axis] < threshold. See selection for options and return value."""
    kwargs.setdefault('desc', '%s below %s' % (axis, threshold))
    return selection(d, d[axis] < threshold, **kwargs)


##
# Range selection helpers
##
def range_selection(d, axis, bounds, **kwargs):
    """Select elements from d for which bounds[0] <= d[axis] < bounds[1].
    kwargs are same as 'selection' method.
    """
    if kwargs.get('_invert', False):
        kwargs.setdefault('desc', '%s NOT in [%s, %s)' % (axis, bounds[0], bounds[1]))
    else:
        kwargs.setdefault('desc', '%s in [%s, %s)' % (axis, bounds[0], bounds[1]))
    return selection(d, (d[axis] >= bounds[0]) & (d[axis] < bounds[1]), **kwargs)


def range_cut(*args, **kwargs):
    """Cut elements in a range from the data; see range_selection docstring."""
    kwargs['_invert'] = True
    return range_selection(*args, **kwargs)


def range_selections(d, *selection_tuples, **kwargs):
    """Do selections based on one or more (axis, bounds) tuples.
     - data on which to perform the selection (pandas dataframe)
     - selection_tuples: one or more tuples like (axis, (low_bound, high_bound)). See range_selection.
     - kwargs are same as 'selection' method.
    """
    for stuff in selection_tuples:
        if 'desc' in kwargs:
            # Make sure each cut has a unique description.
            _kwargs = kwargs.copy()
            _kwargs['desc'] += ' (%s)' % (stuff[0])
        else:
            _kwargs = kwargs
        d = range_selection(d, *stuff, **_kwargs)
    return d


def range_cuts(*args, **kwargs):
    """Do cuts based on one or more (axis, bounds) tuples. See range_selections docstring."""
    kwargs['_invert'] = True
    range_selections(*args, **kwargs)


def apply_lichen(data, lichen_names, lichen_file='sciencerun1', **kwargs):
    """Apply cuts defined by the lax lichen(s) lichen_names from the lichen_file to data.
    """
    # Support for single lichen
    if isinstance(lichen_names, str):
        lichen_names = [lichen_names]

    try:
        import lax
    except ImportError:
        print("You don't seem to have lax. A wise man once said software works better after you install it.")
        raise

    for lichen_name in lichen_names:
        lichen = getattr(getattr(lax.lichens, lichen_file), lichen_name)

        # .copy() to prevent pandas warning and pollution with new columns
        d = lichen().process(data.copy())

        desc = lichen_name
        if hasattr(lichen, 'version'):
            desc += ' v' + str(lichen.version)
        else:
            desc += ' (lax v%s)' % lax.__version__

        data = selection(data, getattr(d, 'Cut' + lichen_name), desc=desc, **kwargs)

    return data

##
# pandas.DataFrame.eval selections
##

def eval_selection(d, eval_string, **kwargs):
    """Apply a selection specified by a pandas.DataFrame.eval string that returns the boolean array.
    If no description is provided, the eval string itself is used as the description.
    """
    kwargs.setdefault('desc', eval_string)
    return selection(d, d.eval(eval_string), **kwargs)


def eval_cut(d, eval_string, **kwargs):
    kwargs['_invert'] = True
    return eval_selection(d, eval_string, **kwargs)
