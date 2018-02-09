import logging
import os
import inspect
from configparser import ConfigParser
import socket
import numba  # flake8: noqa: F401
from . import misc, minitrees, paxroot, pmt_plot, raw_data, runs, utils, treemakers, data_extractor, slow_control, \
    trigger_data, ipython, recorrect, unblinding  # flake8: noqa: F401
__version__ = '2.4.0'


# Stitch the package together
# I am surprised this works (even if we do 'from hax' instead of 'from .')
# as some of the modules do 'import hax' or 'from hax.foo import bar'.. shouldn't we get circular imports??
# I need to read up on python packaging more...

# Store the directory of hax (i.e. this file's directory) as HAX_DIR
hax_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# This stores the hax configuration. Will be filled by hax.init().
# DO NOT import this directly (from hax import config), it will not behave like you expect!
# (e.g. if you do hax.init() afterwards, your config will still seem to be the empty dictionary.
# Just do import hax, then hax.init(), then access config has hax.config.
config = {}

log = logging.getLogger('hax.__init__')


def init(filename=None, **kwargs):
    """Loads hax configuration from hax.ini file filename. You should always call this before starting up hax.
    You can call it again to reload the hax config.
    Any keyword arguments passed will override settings from the configuration.
    """

    if filename is None:
        filename = os.path.join(hax_dir, 'hax.ini')

    global config
    configp = ConfigParser(inline_comment_prefixes='#', strict=True)
    configp.read(filename)

    # Pick the correct section for this host
    section_to_use = 'DEFAULT'
    full_domain_name = socket.getfqdn()
    for section_name in configp.sections():
        if section_name in full_domain_name:
            section_to_use = section_name
            break

    # Evaluate the values in the ini file
    config = {}
    for key, value in configp[section_to_use].items():
        config[key] = eval(value, {'hax_dir': hax_dir, 'os': os})

    # Set the logging level of the root logger (and therefore all loggers that don't have a special loglevel specified)
    # This is better than basicConfig, which only works once in a python
    # session (and pax already uses it :-()
    logging.getLogger().setLevel(getattr(logging, config.get('log_level', 'INFO')))
    log.debug("Read in hax configuration file %s and set up logging" % filename)

    # Override with kwargs
    config.update(kwargs)

    # Convert potential 'raw_data_local_path' entry for backwards compatibility
    if "raw_data_local_path" in config and isinstance(config["raw_data_local_path"], str):
        config["raw_data_local_path"] = [config["raw_data_local_path"]]

    # Call some inits of the submodules
    runs.load_corrections()
    runs.update_datasets()
    minitrees.update_treemakers()
    slow_control.init_sc_interface()

    if not config['cax_key'] or config['cax_key'] == 'sorry_I_dont_have_one':
        log.warning("You're not at a XENON analysis facility, or hax can't detect at which analysis facility you are.")
        if config['pax_version_policy'] != 'loose':
            raise ValueError(
                "Outside an analysis facility you must explicitly set pax_version_policy = 'loose', "
                "to acknowledge you are not getting any version consistency checks."
            )

    # Setup unblinding selection
    unblinding.make_unblinding_selection()
