import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def draw_distribution(data, bnd_lst):
    """
    data shape must be one dimension,
    if one has (42, 165) shape, convert it to 42*165
    likewise, Thailand has (42, 165, 4, 4), then flatten it to 42*165*4*4
    """
    fig = plt.figure()
    ax = plt.subplot()
    ax.hist(data, bins=1000, alpha=.5, color='darkcyan')
    for i in bnd_lst:
        ax.axvline(i, umin=0, ymax=len(data), alpha=.8, color='salmon')
    plt.show()

