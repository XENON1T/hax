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
config = load_configuration('XENON1T')

pmt_data = pd.DataFrame([flatten_dict(info, sep='_') for info in config['DEFAULT']['pmts']])

pmt_data['PMT'] = pmt_data['pmt_position']
del pmt_data['pmt_position']

# Split connector + channel information into separate fields
# Note to anyone reading this: always make separate fields for separate information
for keyname in ('high_voltage_connector', 'signal_connector'):
    main = [int(q.split('.')[0]) for q in pmt_data[keyname]]
    channel = [int(q.split('.')[1]) for q in pmt_data[keyname]]
    pmt_data[keyname] = main
    pmt_data[keyname + '_channel'] = channel
        
pmt_numbering_start = pmt_data['PMT'].min()

##
# Plotting functions
##

def _plot_pmts(ax, pmt_selection, 
               xkey, ykey=None, 
               color=None, size=None, tight_limits=False, 
               **kwargs):
    """Makes a scatter plot of pmts on ax.
    Plot only pmts for which pmt_selection is True. xkey and ykey are used for x, y.
    Returns the return value of plt.scatter (useful to define a colorbar).
    """
    if ykey is not None:
        x, y, pmt_numbers = pmt_data[pmt_selection][[xkey, ykey, 'PMT']].as_matrix().T
        xlabel = xkey
        ylabel = ykey
    else:
        # No ykey is given: just stack pmts with same xkey in y.
        plot_keys, pmt_numbers = pmt_data[pmt_selection][[xkey, 'PMT']].as_matrix().T
        
        labels = sorted(list(set(plot_keys)))
        n_occs = defaultdict(float)
        y = []
        x = []
        for i, k in enumerate(plot_keys):
            x.append(labels.index(k))
            y.append(n_occs[k])
            n_occs[k] += 1
        xlabel = xkey
        ylabel = ''
    
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
    if ykey is None:
        plt.xticks(np.arange(len(labels)), labels, rotation='vertical')
        #ax.set_xticks(x)
        #ax.set_xticklabels(labels, rotation='vertical')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)   
                   
    return sc


def plot_on_pmt_arrays(color=None, size=None,
                       geometry='physical',
                       fontsize=8,title=None,
                       scatter_kwargs=None, colorbar_kwargs=None):
    """Plot a scatter plot of PMTs in a specified geometry, with a specified color and size of the markers.
        Color or size must be per-PMT array that is indexable by another array, i.e. must be np.array and not list.
        scatter_kwargs will be passed to plt.scatter
        colorbar_kwargs will be passed to plt.colorbar
        geometry can be 'physical', or any key from pmt_data
    """
    if scatter_kwargs is None:
        scatter_kwargs = dict()
    if colorbar_kwargs is None:
        colorbar_kwargs = dict()
    if size is None:
        if color is None:
            raise ValueError("Give me at least size or color")
        size = 1000 * color/np.nanmean(color)
    if color is None:
        color = 'blue'
        
    if geometry == 'physical':
        # For the physical geometry, plot the top and bottom arrays side-by-side.
        _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 7))
        
        if title is not None:
            plt.suptitle(title,fontsize=24,x=0.435,y=1.0)
                   
        for array_name, ax in (('top', ax1), ('bottom', ax2)):
            sc = _plot_pmts(ax=ax, 
                            pmt_selection=(pmt_data['array'] == array_name).as_matrix(), 
                            xkey='position_x', ykey='position_y', 
                            color=color, size=size, 
                            tight_limits=True, **scatter_kwargs)
            ax.set_title('%s array' % array_name.capitalize())
                   
        cax, _ = matplotlib.colorbar.make_axes([ax1, ax2])
        plt.colorbar(sc, cax=cax, **colorbar_kwargs)
            
    else:
        plt.figure(figsize=(20, 8))
        if geometry + '_channel' in pmt_data.columns:
            sc = _plot_pmts(ax=plt.gca(), 
                            pmt_selection=slice(None), 
                            xkey=geometry, ykey=geometry + '_channel',
                            color=color, size=size,
                            **scatter_kwargs)
            plt.colorbar(sc, **colorbar_kwargs)
            
        else:
            sc = _plot_pmts(ax=plt.gca(), 
                            pmt_selection=slice(None), 
                            xkey=geometry,
                            color=color, size=size,
                            **scatter_kwargs)
            plt.colorbar(sc, **colorbar_kwargs)
