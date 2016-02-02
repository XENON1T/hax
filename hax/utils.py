import os
import inspect

# Store the directory of hax (i.e. this file's directory) as PAX_DIR
HAX_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def find_file_in_folders(filename, folders):
    """Searches for filename in folders, then return full path or raise FileNotFoundError"""
    for folder in folders:
        full_path = os.path.join(folder, filename)
        if os.path.exists(full_path):
            return full_path
    raise FileNotFoundError("Did not find file %s!" % filename)
