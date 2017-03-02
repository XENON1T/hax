"""Functions to redo late-stage pax corrections with new maps on existing minitree dataframes

These functions will be slow, since the pax interpolating map was never designed to be quick (vectorized),
other processing plugins dominate the run time of pax.
"""
import numpy as np

from pax import configuration
from tqdm import tqdm
from pax.utils import data_file_name
from pax.InterpolatingMap import InterpolatingMap
pax_config = configuration.load_configuration('XENON1T')      # TODO: use hax.config['experiment'], do this after init


def add_uncorrected_position(data):
    """Adds r, theta, u_r, u_x, u_y, u_z to data. If u_x already exists, does nothing.
    Returns no value. Modifies data in place.
    """
    if '_u_x' in data.columns:
        return
    data['r'] = np.sqrt(data['x']**2 + data['y']**2)
    data['theta'] = np.arctan2(data.y, data.x)

    data['_u_r'] = data['r'] - data['r_pos_correction']
    data['_u_z'] = data['z'] - data['z_pos_correction']

    data['_u_x'] = data['_u_r'] * np.cos(data.theta)
    data['_u_y'] = data['_u_r'] * np.sin(data.theta)


def recorrect_s2xy(data,
                   old_map_file='s2_xy_XENON1T_17Feb2017.json',
                   new_map_file=pax_config['WaveformSimulator']['s2_light_yield_map']):
    """Recompute the (x,y) correction for a different map
    :param data: dataframe (Basics and Extended minitrees required)
    :param old_map_file: Map filename that was used to process the dataframe. Defaults to the map used for 6.4.2
    :param new_map_file: Map filename that you want to use for the correction. Defaults to the pax config default.
    :return: dataframe with altered value in cS2 (and few added columns for uncorrected position)
    """
    data = data.copy()
    add_uncorrected_position(data)

    old_map = InterpolatingMap(data_file_name(old_map_file))
    new_map = InterpolatingMap(data_file_name(new_map_file))

    # Correction is a *division* factor (map contains light yield), so to un-correct we first multiply
    recorrection = np.zeros(len(data))
    x = data._u_x.values
    y = data._u_y.values
    for i in tqdm(range(len(data))):
        recorrection[i] = old_map.get_value(x[i], y[i]) / new_map.get_value(x[i], y[i])

    data['cs2'] *= recorrection

    return data


def recorrect_rz(data, new_map_file=pax_config['WaveformSimulator']['rz_position_distortion_map']):
    """Recompute the (r,z)(r,z) field distortion correction
    Be sure to redo the S1(x,y,z) correction after this as well, whether or not the S1(x,y,z) map changed!

    :param data: input dataframe
    :param new_map_file: file with (r,z)(r,z) correction map to use. Defaults to map currently in pax config.
    :return: dataframe with altered values in x, y, z (and few added columns for uncorrected position)
    """
    data = data.copy()
    add_uncorrected_position(data)

    # Compute correction for new map
    new_map = InterpolatingMap(data_file_name(new_map_file))

    r_corr = np.zeros(len(data))
    z_corr = np.zeros(len(data))
    _u_r = data._u_r.values
    _u_z = data._u_z.values
    for i in tqdm(range(len(data)), desc="Redoing (r,z) correction"):
        r_corr[i] = new_map.get_value(_u_r[i], _u_z[i], map_name='to_true_r')
        z_corr[i] = new_map.get_value(_u_r[i], _u_z[i], map_name='to_true_z')

    data['r'] = data._u_r + r_corr
    data['x'] = data.r * np.cos(data.theta)
    data['y'] = data.r * np.sin(data.theta)
    data['z'] = data._u_z + z_corr
    return data


def recorrect_s1xyz(data,
                     old_map_file='s1_xyz_XENON1T_kr83m_sep29_doublez.json',
                     new_map_file=pax_config['WaveformSimulator']['s1_light_yield_map']):
    """Recompute the S1(x,y,z) light yield correction.
    If you want to redo (r,z)(r,z), do it before doing this!

    :param data: Dataframe. Only Basics minitree required.
    :param old_map_name: Filename of map used to process the data. Defaults to map used in pax v6.4.2
    :param new_map_name: Filename of map you want to use for the correction.
    :return: Dataframe with changed values in cs1 column
    """

    old_map = InterpolatingMap(data_file_name(old_map_file))
    new_map = InterpolatingMap(data_file_name(new_map_file))

    # Correction is a *division* factor (map contains light yield), so to un-correct we first multiply
    x = data.x.values
    y = data.y.values
    z = data.z.values
    recorrection = np.zeros(len(data))
    for i in tqdm(range(len(data)), desc='Redoing S1(x,y,z) correction'):
        recorrection[i] *= old_map.get_value(x[i], y[i], z[i]) / new_map.get_value(x[i], y[i], z[i])

    data['cs1'] *= recorrection

    return data
