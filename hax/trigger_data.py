import hax
import zipfile
import zlib
import bson
from collections import defaultdict
from pax import datastructure

# Custom data types used (others are just np.int)
data_types = {
    'trigger_signals': datastructure.TriggerSignal.get_dtype(),
    'trigger_signals_histogram': np.float,
}
matrix_fields = ['trigger_signals_histogram', 'count_of_2pmt_coincidences']


def get_trigger_data(run_id, select_data_types='all'):
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
                d = np.fromstring(d['data'], dtype=data_types.get(data_type, np.int))
            data[data_type].append(d)

    # Flatten / post-process the data
    for k in data.keys():
        if not len(data[k]):
            continue

        if isinstance(data[k][0], dict):
            # Dictionaries describing data
            data[k] = pd.DataFrame(data['batch_info'])

        elif k == 'trigger_signals':
            data[k] = np.concatenate(data[k])

        else:
            data[k] = np.vstack(data[k])

            if k in matrix_fields:
                n = np.sqrt(data[k].shape[1]).astype('int')
                data[k] = data[k].reshape((-1, n, n))

    if select_data_types != 'all' and len(select_data_types) == 1:
        return data[select_data_types[0]]
    return data
