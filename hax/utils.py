import os
import inspect
import platform
import pwd

# Store the directory of hax (i.e. this file's directory) as PAX_DIR
HAX_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def find_file_in_folders(filename, folders):
    """Searches for filename in folders, then return full path or raise FileNotFoundError"""
    for folder in folders:
        full_path = os.path.join(folder, filename)
        if os.path.exists(full_path):
            return full_path
    raise FileNotFoundError("Did not find file %s!" % filename)


def get_user_id():
    """:return: string identifying the currently active system user as name@node                                                
    :note: user can be set with the 'USER' environment variable, usually set on windows                                         
    :note: on unix based systems you can use the password database                                                              
    to get the login name of the effective process user"""
    if os.name == "posix":
        username = pwd.getpwuid(os.geteuid()).pw_name
    else:
        ukn = 'UNKNOWN'
        username = os.environ.get('USER', os.environ.get('USERNAME', ukn))
        if username == ukn and hasattr(os, 'getlogin'):
            username = os.getlogin()

    return "%s@%s" % (username, platform.node())
