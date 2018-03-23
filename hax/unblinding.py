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

# Unblinded regions for WIMP SI search (see #168)
unblind['wimp'] = {}

# Full low-E unblinding for SR0+SR1 (see #216)
unblind['wimp']['unblind_lowe'] = '(cs1<80)'

# ER band above -2*RMS at low-E
unblind['wimp']['above_er_cs1_lt250'] = '((cs1<252) & (log(cs2_bottom/cs1)/log(10) > 0.466119*exp(-cs1/47.9903) + 1.31033 -0.000314047*cs1 + 1.33977/cs1))'

# Kr line above -3*RMS
unblind['wimp']['above_er_cs1_lt375'] = '((250<cs1) & (cs1<375) & (log(cs2_bottom/cs1)/log(10) > 0.822161*exp(-(cs1-207.702)/343.275) + 0.515139))'

# ER band (constant line at high-E)
unblind['wimp']['above_er_cs1_gt375'] = '((cs1>=375) & (log(cs2_bottom/cs1)/log(10) > 1.02015))'

# Above cs1 = 3000 pe
unblind['wimp']['cs1_gt3000'] = '(cs1>3000)'

# Below S2 threshold (for AC modeling)
unblind['wimp']['s2_threshold'] = '(s2<200)'

# Multiple scatters
unblind['wimp']['multiple_scatter'] = '(largest_other_s2 > 200)'

# Reconstructed position outside TPC (for wall leakage modeling; see #169)
unblind['wimp']['outside_tpc'] = '((cs1<500) & (r_3d_nn>47.9))'

# Below NR band (constant line at low-E for wall+AC modeling; see #199)
unblind['wimp']['below_nr_cs1_lt20'] = '((cs1<20) & (log(cs2_bottom/cs1)/log(10) < 1.08159))'

# Below NR band -4.5sigma (see #199)
unblind['wimp']['below_nr_cs1_gt20'] = '((20<=cs1) & (log(cs2_bottom/cs1)/log(10) < 1.21239 + -0.0016025*cs1 + -1.97495/cs1))'

# 2 e- capture (DEC) blinded from 50-80 keV (see #161)
unblind['dec'] = {}
unblind['dec']['e_range'] = '((0.0137*(cs1/.1429 + cs2_bottom/11.36) < 50.) | (0.0137*(cs1/.1429 + cs2_bottom/11.36) > 80.))'

# Unblinding of 14 days post-AmBe data for I-125 removal assessment (see #215)
unblind['dec']['i125_after_ambe'] = '((run_number>=8340) & (run_number<=8728))'

# Unblinding of post-NG data after SR1 for I-125 removal assessment (see #220; compare to #215)
unblind['dec']['i125_after_NG'] = '((run_number>17580) & (run_number<=17820))'

# 0vbb blinded from 2.3-2.6 MeV (see #161)
unblind['0vbb'] = {}
unblind['0vbb']['e_range'] = '((0.0137*(cs1/.1429 + cs2_bottom/11.36) < 2300.) | (0.0137*(cs1/.1429 + cs2_bottom/11.36) > 2600.))'


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

    # Do not blind MC
    if hax.runs.is_mc(run_id)[0]:
        return False

    try:
        run_number = hax.runs.get_run_number(run_id)
        run_data = hax.runs.datasets.query('number == %d' % run_number).iloc[0]

    except Exception:
        # Couldn't find in runDB, so blind by default
        log.warning("Exception while trying to find run %s: blinding by default" % run_id)
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
