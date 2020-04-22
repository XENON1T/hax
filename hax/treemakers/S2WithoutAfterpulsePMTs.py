import gzip
import hax
import json
import logging
from hax.minitrees import TreeMaker
from pax.configuration import load_configuration
from pax.utils import data_file_name
import re
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
import numpy as np


class InterpolateAndExtrapolate(object):
    """Linearly interpolate- and extrapolate using inverse-distance
    weighted averaging between nearby points.
    Note: Taken from code by Tianyu Zhu
    """

    def __init__(self, points, values, neighbours_to_use=None):
        """
        :param points: array (n_points, n_dims) of coordinates
        :param values: array (n_points) of values
        :param neighbours_to_use: Number of neighbouring points to use for
        averaging. Default is 2 * dimensions of points.
        """
        self.kdtree = cKDTree(points)
        self.values = values
        if neighbours_to_use is None:
            neighbours_to_use = points.shape[1] * 2
        self.neighbours_to_use = neighbours_to_use

    def __call__(self, points):
        distances, indices = self.kdtree.query(points, self.neighbours_to_use)
        # If one of the coordinates is NaN, the neighbour-query fails.
        # If we don't filter these out, it would result in an IndexError
        # as the kdtree returns an invalid index if it can't find neighbours.
        result = np.ones(len(points)) * float('nan')
        valid = (distances < float('inf')).max(axis=-1)
        result[valid] = np.average(
            self.values[indices[valid]],
            weights=1/np.clip(distances[valid], 1e-6, float('inf')),
            axis=-1)
        return result


class InterpolateAndExtrapolateArray(InterpolateAndExtrapolate):
    """Note: Taken from code by Tianyu Zhu"""
    def __call__(self, points):
        distances, indices = self.kdtree.query(points, self.neighbours_to_use)
        result = np.ones((len(points), self.values.shape[-1])) * float('nan')
        valid = (distances < float('inf')).max(axis=-1)

        values = self.values[indices[valid]]
        weights = np.repeat(1/np.clip(distances[valid], 1e-6, float('inf')), values.shape[-1]).reshape(values.shape)

        result[valid] = np.average(values, weights=weights, axis=-2)
        return result


class InterpolatingMapInverseDist(object):
    """Correction map that computes values using inverse-weighted distance
    interpolation.
    Note: Taken from code by Tianyu Zhu

    The map must be specified as a json translating to a dictionary like this:
        'coordinate_system' :   [[x1, y1], [x2, y2], [x3, y3], [x4, y4], ...],
        'map' :                 [value1, value2, value3, value4, ...]
        'another_map' :         idem
        'name':                 'Nice file with maps',
        'description':          'Say what the maps are, who you are, etc',
        'timestamp':            unix epoch seconds timestamp

    with the straightforward generalization to 1d and 3d.

    The default map name is 'map', I'd recommend you use that.

    For a 0d placeholder map, use
        'points': [],
        'map': 42,
        etc

    """
    data_field_names = ['timestamp', 'description', 'coordinate_system',
                        'name', 'irregular']

    def __init__(self, data):
        self.log = logging.getLogger('InterpolatingMapInverseDist')

        if data.endswith('.gz'):
            data_file = gzip.open(data).read()
            self.data = json.loads(data_file.decode())
        else:
            with open(data) as data_file:
                self.data = json.load(data_file)

        self.coordinate_system = cs = self.data['coordinate_system']
        if not len(cs):
            self.dimensions = 0
        elif isinstance(cs[0], list):
            if isinstance(cs[0][0], str):
                grid = [np.linspace(left, right, points) for axis, (left, right, points) in cs]
                cs = np.array(np.meshgrid(*grid))
                cs = np.transpose(cs, np.roll(np.arange(len(grid)+1), -1))
                cs = np.array(cs).reshape((-1, len(grid)))
                self.dimensions =len(grid)
            else:
                self.dimensions = len(cs[0])
        else:
            self.dimensions = 1

        self.interpolators = {}
        self.map_names = sorted([k for k in self.data.keys()
                                 if k not in self.data_field_names])
        self.log.debug('Map name: %s' % self.data['name'])
        self.log.debug('Map description:\n    ' +
                       re.sub(r'\n', r'\n    ', self.data['description']))
        self.log.debug("Map names found: %s" % self.map_names)

        for map_name in self.map_names:
            map_data = np.array(self.data[map_name])
            if self.dimensions == 0:
                # 0 D -- placeholder maps which take no arguments
                # and always return a single value
                def itp_fun(positions):
                    return map_data * np.ones_like(positions)
            elif len(map_data.shape) == self.dimensions + 1:
                map_data = map_data.reshape((-1, map_data.shape[-1]))
                itp_fun = InterpolateAndExtrapolateArray(points=np.array(cs),
                                                    values=np.array(map_data))
            else:
                itp_fun = InterpolateAndExtrapolate(points=np.array(cs),
                                                    values=np.array(map_data))

            self.interpolators[map_name] = itp_fun

    def __call__(self, positions, map_name='map'):
        """Returns the value of the map at the position given by coordinates
        :param positions: array (n_dim) or (n_points, n_dim) of positions
        :param map_name: Name of the map to use. Default is 'map'.
        """
        return self.interpolators[map_name](positions)


class S2WithoutAfterpulsePMTs(TreeMaker):
    """Calculate main interaction S2 parameters related to contributions from high afterpulse PMTs removed

    Provides:
     - s2_from_top_ap_pmts: Contribution of high afterpulse PMTs in top array to main S2
     - s2_from_bottom_ap_pmts: Contribution of high afterpulse PMTs in bottom array to main S2
     - s2_xy_correction_no_ap_pmts_top: S2 XY correction factor for top array contribution when high afterpulse PMTs are removed from analysis
     - s2_xy_correction_no_ap_pmts_bottom: S2 XY correction factor for bottom array contribution when high afterpulse PMTs are removed from analysis
     - s2_t_correction_no_ap_pmts_top: S2 time-based correction factor for top array contribution when high afterpulse PMTs are removed from analysis
     - s2_t_correction_no_ap_pmts_bottom: S2 time-based correction factor for bottom array contribution when high afterpulse PMTs are removed from analysis
     - s2_no_ap_pmts: S2 area without high afterpulse PMTs
     - s2_top_no_ap_pmts: S2 area top array contribution without high afterpulse PMTs
     - s2_bottom_no_ap_pmts: S2 area bottom array contribution without high afterpulse PMTs
     - cxyts2_top_no_ap_pmts: S2 area top array contribution without high afterpulse PMTs (corrected for XY and event time dependence)
     - cxyts2_bottom_no_ap_pmts: S2 area bottom array contribution without high afterpulse PMTs (corrected for XY and event time dependence)
     - cs2_aft_no_ap_pmts: S2 area fraction top (corrected for XY dependence)
    """

    __version__ = '1.0'

    extra_branches = ['start_time',
                      'peaks.area',
                      'peaks.area_fraction_top',
                      'peaks.reconstructed_positions*',
                      'peaks.area_per_channel*']

    def __init__(self):
        hax.minitrees.TreeMaker.__init__(self)
        pax_config = load_configuration('XENON1T')
        self.apc = pax_config['DesaturatePulses.DesaturatePulses']['large_after_pulsing_channels']
        self.apc_top = list(set(self.apc).intersection(pax_config['DEFAULT']['channels_top']))
        self.apc_bottom = list(set(self.apc).intersection(pax_config['DEFAULT']['channels_bottom']))
        self.config = hax.config['special_minitree_options']['S2WithoutAfterpulsePMTs']

        self.lce_s2_xy = InterpolatingMapInverseDist(data_file_name(self.config['xy_map']))

        with open(data_file_name(self.config['t_map'])) as data_file:
            data = json.load(data_file)
            self.lce_s2_t_top = interp1d(data['coordinate_system'],
                                         data['map_top_reduced_ap'],
                                         bounds_error=False,
                                         fill_value='extrapolate')
            self.lce_s2_t_bottom = interp1d(data['coordinate_system'],
                                            data['map_bottom_reduced_ap'],
                                            bounds_error=False,
                                            fill_value='extrapolate')

    def extract_data(self, event):
        result = dict()

        # return empty dict if there is no interaction
        if len(event.interactions) == 0:
            return result

        # get main interaction and determine
        interaction = event.interactions[0]
        s2 = event.peaks[interaction.s2]

        result['s2_from_top_ap_pmts'] = np.sum(
            np.array(list(s2.area_per_channel))[self.apc_top])
        result['s2_from_bottom_ap_pmts'] = np.sum(
            np.array(list(s2.area_per_channel))[self.apc_bottom])

        # calculate XY and time-based correction factors
        # use X/Y observed from TPF
        x_obs = None
        y_obs = None

        for rp in s2.reconstructed_positions:
            if rp.algorithm == 'PosRecTopPatternFit':
                x_obs = rp.x
                y_obs = rp.y
                break

        if x_obs is None or y_obs is None:
            result['s2_xy_correction_no_ap_pmts_top'] = float('nan')
            result['s2_xy_correction_no_ap_pmts_bottom'] = float('nan')
        else:
            result['s2_xy_correction_no_ap_pmts_top'] = (1. / self.lce_s2_xy(np.array([x_obs, y_obs]).reshape(-1, 2), 'map_top_reduced_ap_nn_tf'))[0]
            result['s2_xy_correction_no_ap_pmts_bottom'] = (1. / self.lce_s2_xy(np.array([x_obs, y_obs]).reshape(-1, 2), 'map_bottom_reduced_ap_nn_tf'))[0]

        result['s2_t_correction_no_ap_pmts_top'] = 1. / self.lce_s2_t_top(event.start_time)
        result['s2_t_correction_no_ap_pmts_bottom'] = 1. / self.lce_s2_t_bottom(event.start_time)

        # calculate S2 size parameters for top and bottom array contributions
        result['s2_no_ap_pmts'] = s2.area - result['s2_from_top_ap_pmts'] - result['s2_from_bottom_ap_pmts']
        result['s2_top_no_ap_pmts'] = s2.area * s2.area_fraction_top - result['s2_from_top_ap_pmts']
        result['s2_bottom_no_ap_pmts'] = s2.area * (1. - s2.area_fraction_top)- result['s2_from_bottom_ap_pmts']
        result['cxyts2_top_no_ap_pmts'] = result['s2_top_no_ap_pmts'] * result['s2_xy_correction_no_ap_pmts_top'] * result['s2_t_correction_no_ap_pmts_top']
        result['cxyts2_bottom_no_ap_pmts'] = result['s2_bottom_no_ap_pmts'] * result['s2_xy_correction_no_ap_pmts_bottom'] * result['s2_t_correction_no_ap_pmts_bottom']
        result['cs2_aft_no_ap_pmts'] = result['cxyts2_top_no_ap_pmts'] / (result['cxyts2_top_no_ap_pmts'] + result['cxyts2_bottom_no_ap_pmts'])

        return result
