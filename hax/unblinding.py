import hax
import logging

log = logging.getLogger('hax.unblinding')

blind_from_run = 3936

# Unblinding Selection
# String selections below are prepended to any preselection defined by the user.
# The first field should denote the physics channel of interest.
# All selections in the same physics channel will be concatenated with an OR "|".
# Then each group of physics channel selections will be concatenated with an AND "&".

# Full selection to be constructed out of 'unblind' dictionary below
unblinding_selection = ''

# 2D dictionary first index denoting physics channel, second index set of selections
unblind = {}

# Unblinded regions for CEVNS search
unblind['cevns'] = {}
unblind['cevns']['below_threshold_s2'] = '(s2 < 120)'
unblind['cevns']['three_fold'] = '(s1_tight_coincidence > 2)'

# 0vbb blinded from 2.3-2.6 MeV (see #161)
unblind['0vbb'] = {}
unblind['0vbb']['e_range'] = \
    '((0.0137*(cs1/(z_3d_nn*0.000092 + 0.14628) + cs2_bottom/(-0.017*z_3d_nn + 10.628)) < 2457.83*(1 - 4*0.01))' \
    ' | (0.0137*(cs1/(z_3d_nn*0.000092 + 0.14628) + cs2_bottom/(-0.017*z_3d_nn + 10.628)) > 2457.83*(1 + 4*0.01)))'

# HE multiple scatter
unblind['0vbb']['multiple_scatter'] = '(largest_other_s2 > 10000) & (largest_other_s2 > 0.2 * s2)'

def make_unblinding_selection():
    """Generate full unblinding selection string

    Second field of 'unblind' dictionary above will be joined by OR "|".
    Then each set of cuts will be joined by AND "&" for each first field.
    """
    global unblinding_selection
    unblinding_selection = ''

    # Loop over all physics channels
    for channel in unblind:

        # Join all selections of a given set with OR
        selection_string = ' | '.join(['%s' % (value) for (key, value) in unblind[channel].items()])

        # Create full unblinding selection, joining all sets with AND
        unblinding_selection += '(' + selection_string + ') & '

    # Remove extraneous AND
    unblinding_selection = unblinding_selection[:-3]


def is_blind(run_id):
    """Determine if a dataset should be blinded based on the runDB

    :param run_id: name or number of the run to check

    :returns : True if the blinding cut should be applied, False if not
    """
    if hax.config['experiment'] != 'XENON1T':
        return False

    try:
        # Do not blind MC
        if hax.runs.is_mc(run_id)[0]:
            return False

        run_number = hax.runs.get_run_number(run_id)
        run_data = hax.runs.datasets.query('number == %d' % run_number).iloc[0]

    except Exception:
        # Couldn't find in runDB, so blind by default
        log.warning("Exception while trying to find or identify run %s: blinding by default" % run_id)
        return True

    tag_names = [tag for tag in run_data.tags.split(',')]
    number = run_data['number']

    # Blind runs with explicit blinding tag, unblind runs with explicit unblinding tag.
    # (underscore means that it is a protected tag)
    if 'blinded' in tag_names or '_blinded' in tag_names:
        return True
    if '_unblinded' in tag_names:
        return False

    # Blind runs past a configured run number
    if number > blind_from_run and \
            run_data.reader__ini__name.startswith('background'):
        return True

    # Everything else is not blinded
    return False
