import hax
import logging

log = logging.getLogger('hax.unblinding')

##
# Unblinding selection
# We're blinding NR band (1st term; see #168), 2 e- capture from 50-80 keV (2nd term; see #161), and 0nbb 2.3-2.6 MeV
# (3rd term) Details of NR band blinding (open regions): (LowE above ER -2*RMS) | (above Kr83m -3*RMS)
#                                            | (HighE above ER const line) | (HighE below NR -4.5sigma)
#                                            | (sideband outside TPC radius; see #169)
##
unblinding_selection = '(((log(cs2_bottom/cs1)/log(10) > 0.466119*exp(-cs1/47.9903) + 1.31033 -0.000314047*cs1 + 1.33977/cs1)&(cs1<252)) | ((250<cs1)&(cs1<375)&(log(cs2_bottom/cs1)/log(10) > 0.822161*exp(-(cs1-207.702)/343.275) + 0.515139)) | ((cs1>375)&(log(cs2_bottom/cs1)/log(10) > 1.02015)) | (cs1>200)&(log(cs2_bottom/cs1)/log(10) < 1.21239 + -0.0016025*cs1 + -1.97495/cs1) | ((cs1<500)&(r_3d_nn>47.9)) | (cs1>3000) | (s2<200) | (largest_other_s2>200)) & ((0.0137*(cs1/.1429 + cs2_bottom/11.36) < 50.) | (0.0137*(cs1/.1429 + cs2_bottom/11.36) > 80.)) & ((0.0137*(cs1/.1429 + cs2_bottom/11.36) < 2300.) | (0.0137*(cs1/.1429 + cs2_bottom/11.36) > 2600.))'
blind_from_run = 3936


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
