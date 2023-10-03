'''
plot_params.py

Parameters for matplotlib plotting.
'''

import matplotlib
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
colors = [pl.cm.tab10(i) for i in range(len(regions))]

font = {'family' : 'Arial',
        'size'   : 8}

#matplotlib.rc('font', **font)
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8