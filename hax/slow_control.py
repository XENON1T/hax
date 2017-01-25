import os
from datetime import datetime, timedelta
import requests
import logging

import numpy as np
import pandas as pd

import hax
from hax.utils import human_to_utc_datetime, utc_timestamp

log = logging.getLogger('hax.slow_control')
sc_variables = None


def get_sc_api_key():
    """Return the slow control API key, if we know it"""
    if 'sc_api_key' in hax.config:
        return hax.config['sc_api_key']
    elif 'SC_API_KEY' in os.environ:
        return os.environ['SC_API_KEY']
    else:
        raise ValueError('Please set the SC_API_KEY environment variable or the hax.sc_api_key option '
                         'to access the slow control web API.')


def init_sc_interface():
    """Initialize the slow control interface access and list of variables"""
    global sc_variables

    sc_variables = pd.read_csv(hax.config['sc_variable_list'])

    # lowercase all the descriptions, so queries by description are case insensitive
    sc_variables['Description'] = [x.lower() if not isinstance(x, float) else ''
                                   for x in sc_variables['Description'].values]


class UnknownSlowControlMonikerException(Exception):
    pass


class AmbiguousSlowControlMonikerException(Exception):
    pass


def get_sc_name(name):
    """Return slow control historian name of name.  You can pass
    a historian name, sc name, pid identifier, or description. For a full table, see hax.
    """
    # Find out what variable we need to query. Try all possible slow control abbreviations/codes/etc
    for key in ['Historian_name', 'SC_Name', 'Pid_identifier', 'Description']:
        if key == 'Description':
            # For descriptions, we do an even fuzzier matching: look for descriptions which contain the passed string
            # We lowered all descriptions to become case-insensitive
            mask = np.array([name.lower() in x for x in sc_variables[key]])
            q = sc_variables[mask]
        else:
            q = sc_variables[sc_variables[key] == name]
        if len(q) == 1:
            return q.iloc[0].Historian_name
        elif len(q) > 1:
            raise AmbiguousSlowControlMonikerException("'%s' has multiple mathching %ss: %s" %
                                                       (name, key, str(q[key].values)))
    raise UnknownSlowControlMonikerException("Don't known any slow control moniker matching %s" % name)


def get_sc_data(names, run=None, start=None, end=None):
    """
    Retrieve the data from the historian database (hax.slow_control.get is just a synonym of this function)

    :param names: name or list of names of slow control variables; see get_historian_name.

    :param run: run number/name to return data for. If passed, start/end is ignored.

    :param start: String indicating start of time range, in arbitrary format (thanks to parsedatetime)

    :param end: String indicating end of time range, in arbitrary format

    :return: pandas Series of the values, with index the time in UTC. If you requested multiple names, pandas DataFrame
    """
    c = hax.config

    if isinstance(names, (list, tuple)):
        # Get multiple values, return in a single dataframe. I hope the variables all have the same time resolution,
        # otherwise you get NaNs...
        df = pd.DataFrame([get_sc_data(name, run=run, start=start, end=end) for name in names]).T
        df.columns = names
        return df
    name = names

    try:
        name = get_sc_name(name)
    except UnknownSlowControlMonikerException:
        log.warning("Slow control moniker %s not known, trying to query the API anyway..." % name)

    # Find out the start and end time
    if run is not None:
        q = hax.runs.datasets.query('number == %d' % hax.runs.get_run_number(run)).iloc[0]
        start = q.start
        end = q.end
    else:
        start = human_to_utc_datetime(start)
        end = human_to_utc_datetime(end)

    params = {
        "name": name,
        "QueryType": "lab",
        "StartDateUnix": int(utc_timestamp(start)),
        "EndDateUnix": int(utc_timestamp(end)),
        "username": c['sc_api_username'],
        "api_key": get_sc_api_key(),
    }
    r = requests.get(c['sc_api_url'], params=params)

    dates = []
    values = []
    for entry in r.json():
        dates.append(datetime.utcfromtimestamp(entry['timestampseconds']))
        values.append(entry['value'])

    return pd.Series(values, index=dates)


# Alias for convenience
get = get_sc_data
