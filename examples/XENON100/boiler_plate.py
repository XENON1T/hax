import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.mlab as mlab
import matplotlib   # Needed for font size spec, color map transformation function bla bla
import matplotlib.pyplot as plt
matplotlib.rc('font', size=16)
plt.rcParams['figure.figsize'] = (12.0, 10.0) # resize plots
plt.set_cmap('viridis')

from matplotlib.colors import LogNorm
from sklearn import mixture

import hax

def c(x):
    """ get center"""
    return 0.5*(x[1:]+x[:-1])