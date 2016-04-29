import logging
import os
import inspect
from configparser import ConfigParser
import socket

__version__ = 0.2

# Stitch the package together
# I am surprised this works (even if we do 'from hax' instead of 'from .')
# as some of the modules do 'import hax' or 'from hax.foo import bar'.. shouldn't we get circular imports??
# I need to read up on python packaging more...
from . import ipython, minitrees, paxroot, pmt_plot, raw_data, runs, utils, treemakers, data_extractor

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

    # Do NOT move import to top of file, will crash docs building
    global config
    configp = ConfigParser(inline_comment_prefixes='#', strict=True)
    configp.read(filename)
    log.debug("Read in hax configuration file %s" % filename)

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

    # Override with kwargs
    config.update(kwargs)

    # This import can't be at the top, would be circular
    from hax.runs import update_datasets
    update_datasets()

    from hax.minitrees import update_treemakers
    update_treemakers()