"""Utilities for use INSIDE hax (and perhaps random weird use outside hax)
If you have a nice function that doesn't fit anywhere, misc.py is where you want to go
"""
import collections
import os
import platform
import pickle
import itertools
import gzip

import pytz
from pytz import timezone
from datetime import datetime, timedelta
import parsedatetime


def human_to_utc_datetime(x):
    """Return a python UTC-localized datetime object corresponding to the human-readable date/time x
    :param x: string with a human-readable date/time indication (e.g. "now"). If you specify something absolute, it will
              be taken as UTC.
    """

    return parsedatetime.Calendar().parseDT(datetimeString=x,
                                            sourceTime=datetime.utcnow(),
                                            tzinfo=timezone("UTC"))[0]


def utc_timestamp(d):
    """Convert a UTC datetime object d to (float) seconds in the UTC since the UTC unix epoch.
    If you pass a timezone-naive datetime object, it will be treated as UTC.
    """
    if isinstance(d, pd.tslib.Timestamp):
        # Pandas datetime timestamp whatever thing
        if d.tz != pytz.utc:
            d = d.tz_localize('UTC')
    else:
        # Normal datetime object
        if d.tzinfo != pytz.utc:
            d = pytz.utc.localize(d)
    # Yes, you have to write it out like this, there is no convenient method.
    # See http://stackoverflow.com/questions/8777753/converting-datetime-date-to-utc-timestamp-in-python
    # Datetimes in python really are a mess...
    return (d - datetime(1970, 1, 1, tzinfo=timezone('UTC'))) / timedelta(seconds=1)


def find_file_in_folders(filename, folders):
    """Searches for filename in folders, then return full path or raise FileNotFoundError
    Does not recurse into subdirectories
    """
    for folder in folders:
        folder = os.path.expanduser(folder)
        full_path = os.path.join(folder, filename)
        if os.path.exists(full_path):
            return full_path
    raise FileNotFoundError("Did not find file %s!" % filename)


def combine_pax_configs(config, overrides):
    """Combines configuration dictionaries config and overrides.
    overrides has higher priotity.
    each config must be a dictionary containing only string->dict pairs (like the pax/ConfigParser configs)
    """
    # TODO: we should soon be able to get this from pax, but let's wait a while to prevent incompatibilties
    for section_name, stuff in overrides.items():
        config.setdefault(section_name, {})
        config[section_name].update(stuff)
    return config


def get_user_id():
    """:return: string identifying the currently active system user as name@node
    :note: user can be set with the 'USER' environment variable, usually set on windows
    :note: on unix based systems you can use the password database to get the login name of the effective process user
    """
    if os.name == "posix":
        import pwd   # Don't put this at the top of the file, it's not available on windows
        username = pwd.getpwuid(os.geteuid()).pw_name
    else:
        ukn = 'UNKNOWN'
        username = os.environ.get('USER', os.environ.get('USERNAME', ukn))
        if username == ukn and hasattr(os, 'getlogin'):
            username = os.getlogin()

    return "%s@%s" % (username, platform.node())


def get_xenon100_dataset_number(dsetname):
    """Converts a XENON100 dataset name to a number"""
    _, date, time = dsetname.split('_')
    return int(date) * 10000 + int(time)


def flatten_dict(d, separator=':', _parent_key=''):
    """Flatten nested dictionaries into a single dictionary, indicating levels by separator
    Don't set _parent_key argument, this is used for recursive calls.
    Stolen from http://stackoverflow.com/questions/6027558
    """
    items = []
    for k, v in d.items():
        new_key = _parent_key + separator + k if _parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, separator=separator, _parent_key=new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_pickles(filename, load_first=None):
    """Returns list of pickles stored in filename.
    :param load_first: number of pickles to read. Otherwise reads until file is exhausted
    """
    if load_first is None:
        counter = itertools.count()
    else:
        counter = range(load_first)
    with gzip.open(filename, mode='rb') as infile:
        result = []
        for _ in counter:
            try:
                result.append(pickle.load(infile))
            except EOFError:
                if load_first is not None:
                    raise
                break
    return result


def save_pickles(filename, *args):
    """Compresses and pickles *args to filename.
    The pickles are stacked: load them with load_pickles"""
    with gzip.open(filename, 'wb') as outfile:
        for thing in args:
            pickle.dump(thing, outfile)
