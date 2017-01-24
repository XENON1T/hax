from datetime import datetime, timedelta
import requests
import logging
import os

from pytz import timezone
import parsedatetime
import numpy as np
import pandas as pd
from Crypto.Cipher import DES

import hax

log = logging.getLogger('hax.slow_control')
sc_variables = None


def get_utc_datetime(x, return_type='timestamp'):
    """Return UTC timestamp or a python UTC-localized datetime object corresponding to the human-readable date/time x
    :param x: string with a human-readable date/time indication (e.g. "now"). If you specify something absolute, it will
              be taken as UTC.
    :param return_type: if 'timestamp' (default), return the time as number of seconds since the epoch.
                        Otherwise, return a datetime.datetime UTC-localized object.
    """
    cal = parsedatetime.Calendar()
    f = cal.parseDT(datetimeString=x,
                    sourceTime=datetime.utcnow(),
                    tzinfo=timezone("UTC"))[0]
    if return_type == 'timestamp':
        # To convert to a UTC epoch timestamp, further trickery is necessary
        # see http://stackoverflow.com/questions/8777753/converting-datetime-date-to-utc-timestamp-in-python
        # Datetimes in python really are a mess
        return (d - datetime(1970, 1, 1, tzinfo=timezone('UTC'))) / timedelta(seconds=1)
    return d



def init_sc_interface():
    """Initialize the slow control interface access and list of variables"""
    global sc_variables

    # Decrypt the slow control API key with the runs db password
    # (we can't store the plaintext API key in an open-source program)
    des = DES.new(hax.runs.get_rundb_password()[:8], DES.MODE_ECB)
    hax.config['sc_api_key'] = des.decrypt(hax.config['sc_api_key_encrypted']).decode('utf-8')

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


def get_sc_data(name, run=None, start=None, end=None):
    """
    Retrieve the data from the historian database (hax.slow_control.get is just a synonym of this function)

    :param name: name of a slow control variable; see get_historian_name.

    :param run: run number/name to return data for. If passed, start/end is ignored.

    :param start: String indicating start of time range, in arbitrary format (thanks to parsedatetime)

    :param end: String indicating end of time range, in arbitrary format

    :return: pandas Series of the values, with index the time in UTC.
    """
    c = hax.config

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
        start = get_utc_datetime(start)
        end = get_utc_datetime(end)

    params = {
        "name": name,
        "QueryType": "lab",
        "StartDateUnix": start,
        "EndDateUnix": end,
        "username": c['sc_api_username'],
        "api_key": c['sc_api_key'],
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