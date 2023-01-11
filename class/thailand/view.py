import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs

def draw_val(val_pred, val_label):
    fig = plt.figure()
    ax = plt.subplot()

    val_list = [1, 2, 3, 4, 5]
    width, linewidth, align = 0.5, 0.5, 'center'

    # count
    val_true = []
    val_false = []
    for i, j in zip(val_pred, val_label):
        if np.argmax(i) == np.argmax(j):
            val_true.append(val_list[np.argmax(j)])
        else:
            val_false.append(val_list[np.argmax(j)])

    # true
    val_tcount = [val_true.count(j) for j in val_list]
    ax.bar(val_list, val_tcount,
           color='darkslategray', width=width, linewidth=linewidth, align=align)

    # false
    val_fcount = [val_false.count(j) for j in val_list]
    ax.bar(val_list, val_fcount,
           color='orange', bottom=val_tcount, width=width, linewidth=linewidth, align=align, alpha=.8)

    plt.show()

def show_class5(image):
    cmap = plt.cm.get_cmap('BrBG', 5)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ticks = [0, 1, 2, 3, 4]

    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-90, -70, 5, 25) # location = (N5-25, E90-110)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(image,
                     origin='upper',
                     extent=img_extent,
                     transform=projection,
                     norm=norm,
                     cmap=cmap)
    cbar = fig.colorbar(mat,
                        ax=ax,
                        extend='both',
                        ticks=ticks,
                        spacing='proportional',
                        orientation='vertical')
    cbar.ax.set_yticklabels(['low', 'mid-low', 'normal', 'mid-high', 'high'])

    plt.show()

