##
## Plot some data on the PMT arrays
##
## Jelle, February 2016
## May go into hax soon...
##
## Known issue's I'm to lazy to fix right now:
##  - on physical layout, color and/or size probably not on same scale in two subplots 
##    unless vmin&vmax are specified explicitly.
##  - Have categorical labels event if _channel present. Make digitizer obey _channel suffix convention.
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from pax import units
from pax.configuration import load_configuration

from hax.utils import flatten_dict

##
# Load the PMT data from the pax configuration
##

# Convert PMT/channel map to record array
pax_config = load_configuration('XENON1T')      # TODO: depends on experiment, should do after init
pmt_data = pd.DataFrame([flatten_dict(info, separator=':')
                         for info in pax_config['DEFAULT']['pmts']
                         if 'array' in info])
pmt_numbering_start = pmt_data['pmt_position'].min()


##
# Plotting functions
##

def _pad_to_length_of(a, b):
    """Pads a with zeros until it has the length of b"""
    lendiff = len(b) - len(a)
    if lendiff < 0:
        raise ValueError("Cannot pad a negative number of zeros!")
    elif lendiff > 0:
        return np.concatenate((a, np.zeros(lendiff)))
    else:
        return a


def _plot_pmts(ax, xkey, color, size,
               pmt_selection=None,
               ykey=None,
               tight_limits=False,
               **kwargs):
    """Makes a scatter plot of pmts on ax, with xkey (index in pmt_data) on x axis.
     - color, size: control PMT marker properties
     - ykey: index in pmt_data. If given, is put on y axis. If not,  make y axis a meaningless int to distringuish pmts.
     - pmt_selection: legal index array to pmt_data, selects pmts to plot
     - tight_limits: if True, clip limits to min and max of xkey and ykey
    Any kwargs will be passed to ax.scatter

    Returns the return value of ax.scatter (useful to define a colorbar).
    """
    # Pad color and size with zeros, to support just TPC PMTs, for example
    color = _pad_to_length_of(color, pmt_data)
    size = _pad_to_length_of(size, pmt_data)

    if pmt_selection is None:
        # Select all PMTs
        pmt_selection = np.ones(len(pmt_data), dtype=np.bool_)

    xlabel = xkey

    if ykey is None:
        # No y key given, make y a duplication counter
        ykey = 'pmt_position'
        ylabel = 'Meaningless integer'

        x, pmt_numbers = pmt_data[pmt_selection][[xkey, 'pmt_position']].as_matrix().T

        n_occs = defaultdict(float)
        y = []
        for i, q in enumerate(x):
            y.append(n_occs[q])
            n_occs[q] += 1
        y = np.array(y)

    else:
        ylabel = ykey
        x, y, pmt_numbers = pmt_data[pmt_selection][[xkey, ykey, 'pmt_position']].as_matrix().T

    # For xkey and ykey, if integer values, change to a categorical axis
    tick_labels = {}
    for q, qname in ((x, 'x'), (y, 'y')):
        if isinstance(q[0], (int, np.int, np.int32, np.int64)):
            labels = sorted(list(set(q)))
            for i, w in enumerate(q):
                q[i] = labels.index(w)
            tick_labels[qname] = labels
            locals()[qname] = q

    # Plot the PMT as circles with specified colors and sizes
    sc = ax.scatter(x, y, 
                    c=color[pmt_selection], s=size[pmt_selection], 
                    **kwargs)
    
    # Show the PMT id texts
    for i in range(len(x)):
        ax.text(x[i], y[i], int(pmt_numbers[i]),
                fontsize=8, va='center', ha='center')
                   
    # Set limits and labels
    lim_scale = 1.3
    if tight_limits:
        ax.set_xlim(x.min() * lim_scale, x.max() * lim_scale)
        ax.set_ylim(y.min() * lim_scale, y.max() * lim_scale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set categorical ticks
    for qname, labels in tick_labels.items():
        getattr(plt, qname + 'ticks')(np.arange(len(labels)),
                                      labels,
                                      rotation='vertical' if qname == 'x' else 'horizontal')

    return sc


def plot_on_pmt_arrays(color=None, size=None,
                       geometry='physical',
                       title=None,
                       scatter_kwargs=None, colorbar_kwargs=None):
    """Plot a scatter plot of PMTs in a specified geometry, with a specified color and size of the markers.
        Color or size must be per-PMT array that is indexable by another array, i.e. must be np.array and not list.
        scatter_kwargs will be passed to plt.scatter
        colorbar_kwargs will be passed to plt.colorbar
        geometry can be 'physical', a key from pmt_data, or a 2-tuple of keys from pmt_data.
    """
    if scatter_kwargs is None:
        scatter_kwargs = dict()
    if colorbar_kwargs is None:
        colorbar_kwargs = dict()

    if size is None:
        if color is None:
            raise ValueError("Give me at least size or color")
        size = 1000 * np.array(color)/np.nanmean(color)
    if color is None:
        color = 0 * np.ones(len(size))

    # Geometry shortcuts
    geometry = dict(digitizer=('digitizer:module', 'digitizer:channel'),
                    amplifier=('amplifier:serial', 'amplifier:plug'),
                    high_voltage=('high_voltage:connector', 'high_voltage:channel'),
                    signal=('signal:connector', 'signal:channel'),
                    ).get(geometry, geometry)
        
    if geometry == 'physical':
        # For the physical geometry, plot the top and bottom arrays side-by-side.
        _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 7))
        
        if title is not None:
            plt.suptitle(title, fontsize=24, x=0.435, y=1.0)

        # We must extract vmin and max explicitly, since we want two plots with the same scale
        scatter_kwargs.setdefault('vmin', np.nanmin(color))
        scatter_kwargs.setdefault('vmax', np.nanmax(color))
        for array_name, ax in (('top', ax1), ('bottom', ax2)):
            sc = _plot_pmts(ax=ax,
                            xkey='position:x', ykey='position:y',
                            color=color, size=size,
                            pmt_selection=(pmt_data['array'] == array_name).as_matrix(),
                            tight_limits=True,
                            **scatter_kwargs)
            ax.set_title('%s array' % array_name.capitalize())
                   
        cax, _ = matplotlib.colorbar.make_axes([ax1, ax2])
        plt.colorbar(sc, cax=cax, **colorbar_kwargs)
            
    else:
        plt.figure(figsize=(20, 8))
        if isinstance(geometry, tuple) and len(geometry) == 2:
            xkey, ykey = geometry
        else:
            xkey, ykey = geometry, None
        sc = _plot_pmts(ax=plt.gca(),
                        xkey=xkey, ykey=ykey,
                        color=color, size=size,
                        **scatter_kwargs)
        plt.colorbar(sc, **colorbar_kwargs)
