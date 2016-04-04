import collections
import os
import platform
import pwd
import sys


def find_file_in_folders(filename, folders):
    """Searches for filename in folders, then return full path or raise FileNotFoundError
    Does not recurse into subdirectories
    """
    for folder in folders:
        full_path = os.path.join(folder, filename)
        if os.path.exists(full_path):
            return full_path
    raise FileNotFoundError("Did not find file %s!" % filename)


def get_user_id():
    """:return: string identifying the currently active system user as name@node
    :note: user can be set with the 'USER' environment variable, usually set on windows
    :note: on unix based systems you can use the password database to get the login name of the effective process user
    """
    if os.name == "posix":
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


def flatten_dict(d, sep=':', _parent_key=''):
    """Flatten nested dictionaries into a single dictionary, indicating levels by sep
    Leave _parent_key argument alone, used for recursion.
    Stolen from http://stackoverflow.com/questions/6027558
    """
    items = []
    for k, v in d.items():
        new_key = _parent_key + sep + k if _parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, sep=sep, _parent_key=new_key, ).items())
        else:
            items.append((new_key, v))
    return dict(items)