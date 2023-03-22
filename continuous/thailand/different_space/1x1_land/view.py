import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import Normalize

def acc_map(acc, lat_grid=20, lon_grid=20):
    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-90, -70, 5, 25) # location = (N5-25, E90-110)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(acc,
                     origin='upper',
                     extent=img_extent,
                     transform=projection,
                     vmin=0.82, vmax=0.91,
                     cmap='Pastel1')
    cbar = fig.colorbar(mat, ax=ax)
    plt.show(block=False)


def show_map(image, vmin=-1, vmax=1):
    cmap = plt.cm.get_cmap('BrBG')
    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-90, -70, 5, 25) # locatin = (N5-25, E90-110)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(image,
                     origin='upper',
                     extent=img_extent,
                     transform=projection,
                     norm=Normalize(vmin=vmin, vmax=vmax),
                     cmap = cmap)
    cbar = fig.colorbar(mat, ax=ax)
    plt.show(block=False)

def diff_bar(data, vmin=0, vmax=2):
    # grid毎にabs(実際のlabelデータ-予測結果)を400個棒グラフにして出力する
    fig = plt.figure()
    ax = plt.subplot()
    pixcel = np.arange(len(data))
    ax.bar(pixcel, data, color='magenta')
    ax.set_ylim(vmin, vmax)
    plt.show()

def draw_val(true_count, false_count):
    fig = plt.figure()
    ax = plt.subplot()
    print(true_count, false_count)
    ax.barh(1, true_count, height=0.5, color='darkslategray', align='center', label='True')
    ax.barh(1, false_count, left=true_count, height=0.5, color='orange', align='center', label='False')
    ax.set_ylim(0,2)
    ax.set_yticks([1.0], ['val'])
    plt.legend()
    plt.show()

