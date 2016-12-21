import pickle
import zipfile
import zlib
import os
from collections import defaultdict

import pytz
import numpy as np
import bson
import pandas as pd

from pax.configuration import load_configuration
from pax import datastructure, units
import hax

# Custom data types used (others are just np.int)
data_types = {
    'trigger_signals': datastructure.TriggerSignal.get_dtype(),
    'trigger_signals_histogram': np.float,
}
matrix_fields = ['trigger_signals_histogram', 'count_of_2pmt_coincidences']


def get_trigger_data(run_id, select_data_types='all', format_version=2):
    """Return dictionary with the trigger data from run_id
    select_data_types can be 'all', a trigger data type name, or a list of trigger data type names.
    If you want to find out which data types exists, use 'all' and look at the keys of the dictionary.
    """
    if isinstance(select_data_types, str) and select_data_types != 'all':
        select_data_types = [select_data_types]

    data = defaultdict(list)
    run_name = hax.runs.get_run_name(run_id)
    f = zipfile.ZipFile(os.path.join(hax.config['raw_data_local_path'],
                                     run_name, 'trigger_monitor_data.zip'))
    for doc_name in f.namelist():
        data_type, index = doc_name.split('=')
        if select_data_types != 'all':
            if data_type not in select_data_types:
                continue
        with f.open(doc_name) as doc_file:
            d = doc_file.read()
            d = zlib.decompress(d)
            d = bson.BSON.decode(d)
            if 'data' in d:
                if format_version >= 2:
                    # Numpy arrays stored as lists
                    d = np.array(d['data'], dtype=data_types.get(data_type, np.int))
                else:
                    # Numpy arrays stored as strings
                    d = np.fromstring(d['data'], dtype=data_types.get(data_type, np.int))
            data[data_type].append(d)

    # Flatten / post-process the data
    for k in data.keys():
        if not len(data[k]):
            continue

        if isinstance(data[k][0], dict):
            # Dictionaries describing data
            data[k] = pd.DataFrame(data[k])

        elif k == 'trigger_signals':
            data[k] = np.concatenate(data[k])

        else:
            if k in matrix_fields:
                if format_version <= 1:
                    data[k] = np.vstack(data[k])
                    # Arrays were flattened when they are converted to strings
                    n = np.sqrt(data[k].shape[1]).astype('int')
                    data[k] = data[k].reshape((-1, n, n))
                else:
                    # Not flattened, just need to stack
                    data[k] = np.stack(data[k])
            else:
                data[k] = np.vstack(data[k])

    if select_data_types != 'all' and len(select_data_types) == 1:
        return data[select_data_types[0]]
    return data


def get_aqm_pulses(run_id):
    """Return a dictionary of acquisition monitor pulse times in the run run_id.
    keys are channel labels (e.g. muon_veto_trigger).
    Under the keys 'busy' and 'hev', you'll get the sorted combination of all busy/hev _on and _off signals.
    """
    basename = 'acquisition_monitor_data.pickles'
    filename = os.path.join(hax.config['raw_data_local_path'],
                            hax.runs.get_run_name(run_id),
                            basename)

    # Get the run start time in ns since the unix epoch. This isn't known with such accuracy,
    # but we use the exact same determination here as in the event builder.
    # Hence we can compare the times to the event times.
    start_datetime = hax.runs.get_run_info(run_id, 'start').replace(tzinfo=pytz.utc).timestamp()
    time_of_run_start = int(start_datetime * units.s)

    # Get the pax configuration
    pax_config = load_configuration(hax.config['experiment'])
    dt = pax_config['DEFAULT']['sample_duration']

    # Make the (module, channel) -> Acquisition monitor channel label map
    pmt_map = pax_config['DEFAULT']['pmts']
    aqm_channel = {(pmt_map[v[0]]['digitizer']['module'], pmt_map[v[0]]['digitizer']['channel']): k
                   for k, v in pax_config['DEFAULT']['channels_in_detector'].items() if len(v) == 1}

    if not os.path.exists(filename):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            raise ValueError("Raw data directory for %s not found -- "
                             "required to get acquisition monitor pulses" % run_id)
        raise ValueError("Can't load acquisition monitor pulses, no file named %s in %s" % (basename, dirname))

    aqm_signals = {k: [] for k in aqm_channel.values()}
    with open(filename, 'rb') as infile:
        while True:
            try:
                doc = pickle.load(infile)
                aqm_signals[aqm_channel[doc['module'], doc['channel']]].append(doc['time'] * dt + time_of_run_start)
            except EOFError:
                break

    # Combine busy_on and hev_on signals for convenience
    for x in ('busy', 'hev'):
        aqm_signals[x] = np.sort(aqm_signals[x + '_on'] + aqm_signals[x + '_off'])

    return {k: np.array(v) for k, v in aqm_signals.items()}
