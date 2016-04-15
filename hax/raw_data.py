"""Functions for working with raw data.
"""
import atexit
import itertools
import os
import tempfile

import pandas as pd
from urllib.request import HTTPSHandler, build_opener
import http.client

from pax import core

import hax


def inspect_events_from_minitree(events, *args, **kwargs):
    """Show the pax event display for events, where events is a (slice of) a dataframe loaded from a minitree
    Any additional arguments will be passed to inspect_events, see its docstring for details
    """
    if isinstance(events, pd.Series):
        events = pd.DataFrame([events])
    for dataset_number, evts in events.groupby('dataset_number'):
        event_numbers = evts.event_number.values
        inspect_events(dataset_number, event_numbers, *args, **kwargs)


def inspect_events(dataset_number, event_numbers, focus='all', save_to_dir=None):
    """Show the pax event display for the events in dataset_number,

    The dataframe must at least contain 'Basics'; currently only supports XENON100 run 10.
    focus can be 'all' (default) which shows the entire event, 'largest', 'first', 'main_s1', or 'main_s2'
    """
    # Config to let pax fo plotting
    config_dict = {'pax': {'output': ['Plotting.PlotEventSummary' if focus == 'all' else 'Plotting.PeakViewer'],
                           'encoder_plugin': None,
                           'pre_output': [],
                           'output_name': 'SCREEN' if save_to_dir is None else save_to_dir}}
    # You need to set block_view = True to make it NOT block the view
    # TODO: Fix this in pax
    if focus != 'all':
        config_dict['Plotting.PeakViewer'] = {'starting_peak': focus,
                                              'block_view': True}
    else:
        config_dict['Plotting.PlotEventSummary'] = {'block_view': True}

    # After we configure pax to do the plotting, we just have to iterate over the events and do nothing
    # Could do list(...) as well, but that would save all the events and return then in a big list at the end
    for _ in process_events(dataset_number, event_numbers, config_override=config_dict):
        pass


def raw_events(dataset_number, event_numbers=None, config_override=None):
    """Yields raw event(s) numbered event_numbers from dataset numbered dataset_number
    config_override is a dictionary with extra pax options
    """
    if config_override is None:
        config_override = {}

    # Combine the users config_override with the options necessary to get pax to spit out raw events
    config_override.setdefault('pax', {})
    pax_config_dict = {'plugin_group_names': ['input', 'preprocessing', 'output'],
                       'preprocessing':      ['CheckPulses.SortPulses',
                                              'CheckPulses.ConcatenateAdjacentPulses',],
                       'output':             'Dummy.DummyOutput',
                       'encoder_plugin':     None}
    for k, v in pax_config_dict.items():
        config_override['pax'].setdefault(k, v)

    for event in process_events(dataset_number, event_numbers, config_override):
        yield event


def process_events(dataset_number, event_numbers=None, config_override=None):
    """Yields processed event(s) numbered event_numbers from dataset numbered dataset_number
    config_override is a dictionary with extra pax options
    """
    if config_override is None:
        config_override = {}
    if isinstance(event_numbers, int):
        # Support passing a single event number
        event_numbers = [event_numbers]
    config = hax.config

    # Get the dataset information
    dataset_info = hax.runs.datasets[hax.runs.datasets['number'] == dataset_number].iloc[0]

    # Set the events to process in config_override
    if event_numbers is not None:
        config_override.setdefault('pax', {})
        config_override['pax'].setdefault('events_to_process', event_numbers)

    if config['raw_data_access_mode'] == 'local':
        # HURRAY HURRAY we have the raw data locally (either really or through sshfs)
        # We can let pax deal with jumping from file to file, selecting events, etc.
        if not dataset_info.raw_data_found:
            raise ValueError("Raw data for dataset number %d (%s) not found." % (dataset_number,
                                                                                 dataset_info['name']))
        dirname = os.path.join(config['raw_data_local_path'],
                               dataset_info.raw_data_subfolder,
                               dataset_info['name'])
        mypax = raw_data_processor(dirname, config_override)
        for event in mypax.get_events():
            yield mypax.process_event(event)

    elif config['raw_data_access_mode'] == 'grid':
        # OH NO we have to get the raw data from GRID (pam pam pam pompadam pompadam)
        # We only know how to access single files from grid, so we need to predict the file name foreach event,
        # switch files manually and all sorts of other fun stuff.
        global temporary_data_files

        # If event_numbers wasn't specified, just iterate over events until we crash / user had enough
        if event_numbers is None:
            event_numbers = itertools.count()

        if hax.config['experiment'] != 'XENON100':
            raise ValueError("Can't get raw data from GRID for %s!" % hax.config['experiment'])

        currently_open_file_name = None
        for event_number in event_numbers:
            # Which XED file does this event belong to?
            data_file_name = 'xe100_%06d_%04d_%06d.xed' % (int(dataset_number / 1e4),
                                                           dataset_number % 1e4,
                                                           int(event_number / 1e3))

            if data_file_name != currently_open_file_name:
                # Has the required file already been downloaded in this session? Then return its location.
                cache_key = (dataset_number, data_file_name)
                if cache_key in temporary_data_files:
                    path_to_file = temporary_data_files[cache_key]

                else:
                    # We have to download a new file
                    file_path_tail = os.path.join(dataset_info.raw_data_subfolder,
                                                  dataset_info['name'],
                                                  data_file_name)
                    path_to_file = download_from_grid(file_path_tail)
                    temporary_data_files[cache_key] = path_to_file

                currently_open_file_name = os.path.basename(path_to_file)

                # Start a new pax to process the events from this file
                mypax = raw_data_processor(path_to_file, config_override)

            event = mypax.get_single_event(event_number)
            yield mypax.process_event(event)

    else:
        raise ValueError("Unknown raw data access mode %s, must be local or grid." % config['raw_data_access_mode'])


def raw_data_processor(input_file_or_directory, config_override=None):
    """Return a raw data processor which reads events from input_file_or_directory
    config_override can be used to set additional pax options
    """
    if config_override is None:
        config_override = {}

    # Add the input name to the config_override
    # Apply the user overrides, section by section
    config_override.setdefault('pax', {})
    config_override['pax']['input_name'] = input_file_or_directory

    return core.Processor(config_names=hax.config['experiment'], config_dict=config_override)


##
# Grid stuff
##

# Holds paths to temporarily downloaded data files
# dictionary: (dataset, event): path
# will be deleted at exit.
temporary_data_files = {}


def cleanup_temporary_data_files():
    """Removes all temporarily downloaded raw data files.
    Run automatically for you when your program quits
    """
    for tempfile_path in temporary_data_files.values():
        os.remove(tempfile_path)

atexit.register(cleanup_temporary_data_files)


def download_from_grid(file_path_tail):
    """Downloads file_path_tail from grid, returns filename of temporary file
    """
    config = hax.config
    # Check if we have the grid key & certificate
    grid_key_path = os.path.expanduser(config['grid_key'])
    grid_cert_path = os.path.expanduser(config['grid_certificate'])
    if not os.path.exists(grid_key_path):
        raise ValueError("Cannot download from grid: grid key does not exist at %s" % grid_key_path)
    if not os.path.exists(grid_key_path):
        raise ValueError("Cannot download from grid: grid certificate does not exist at %s" % grid_key_path)

    # Make the grid URL
    grid_url = config['raw_data_grid_url'] + ''
    if not grid_url.endswith('/'):
        grid_url += '/'     # Remember strings are immutable, so don't worry
    grid_url += file_path_tail

    # Download the file from GRID
    opener = build_opener(HTTPSClientAuthHandler(grid_key_path, grid_cert_path))
    response = opener.open(grid_url)
    block_sz = 8192
    f = tempfile.NamedTemporaryFile(delete=False)
    while True:
        buffer = response.read(block_sz)
        if not buffer:
            break
        f.write(buffer)
    f.close()

    return f.name


class HTTPSClientAuthHandler(HTTPSHandler):
    """Used for accessing GRID data and handling authentication"""
    def __init__(self, key, cert):
        HTTPSHandler.__init__(self)
        self.key = key
        self.cert = cert

    def https_open(self, req):
        return self.do_open(self.getConnection, req)

    def getConnection(self, host, timeout):
        # TODO: timout is not used, but is passed, can't delete it or error
        return http.client.HTTPSConnection(host, key_file=self.key, cert_file=self.cert)
