import os
import pandas as pd
import tempfile
from urllib.request import HTTPSHandler, build_opener
import http.client

from pax import core

from hax.config import CONFIG


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


def inspect_events(events, focus='all'):
    """Show the waveforms of all events in the dataframe/series events
    The dataframe must at least contain 'Basics'; currently only supports XENON100 run 10.
    focus can be:
        'all': (default): show the entire event
        'largest', 'first', 'main_s1', 'main_s2': show a specific peak
    """
    if isinstance(events, pd.Series):
        events = pd.DataFrame([events])

    for _, event in events.iterrows():
        # Construct the XED filename from the dataset number and event number
        xed_filename = 'xe100_%06d_%04d_%06d.xed' % (int(event.dataset_number / 1e4),
                                                     event.dataset_number % 1e4,
                                                     int(event.event_number / 1e3))
        _inspect_event_xenon100(event.event_number, xed_filename, focus=focus)


def _inspect_event_xenon100(event_number, xed_filename, focus='all'):
    """Show the waveform of event_number from xed_filename, where xed_filename belongs to a XENON100 run 10 dataset
    Will probably be deprecated soon
    """
    dataset_name = xed_filename[:-11]

    with tempfile.NamedTemporaryFile(delete=False) as f:

        if CONFIG['raw_data_access_mode'] == 'local':
            filename = CONFIG['raw_data_local_path'].format(dataset_name=dataset_name, xed_filename=xed_filename)

        else:
            grid_url = CONFIG['raw_data_grid_url'].format(dataset_name=dataset_name, xed_filename=xed_filename)

            opener = build_opener(HTTPSClientAuthHandler(os.path.expanduser(CONFIG['grid_key']),
                                                         os.path.expanduser(CONFIG['grid_certificate'])))
            response = opener.open(grid_url)
            block_sz = 8192
            while True:
                buffer = response.read(block_sz)
                if not buffer:
                    break
                f.write(buffer)
            f.close()

            filename = f.name

        config_dict = {'pax': {'output': ['Plotting.PlotEventSummary' if focus == 'all' else 'Plotting.PeakViewer'],
                               'encoder_plugin': None,
                               'pre_output': [],
                               'input_name':  filename,
                               'events_to_process': [event_number],
                               'output_name': 'SCREEN'}}
        if focus != 'all':
            config_dict['Plotting.PeakViewer'] = {'starting_peak': focus}

        mypax = core.Processor(config_names='XENON100', config_dict=config_dict)
        mypax.run()
        os.unlink(f.name)