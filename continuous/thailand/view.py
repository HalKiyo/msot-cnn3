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

def show_class(image, class_num=5, lat_grid=4, lon_grid=4):
    cmap = plt.cm.get_cmap('BrBG', class_num)
    bounds = [i - 0.5 for i in range(class_num + 1)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ticks = [i for i in range(class_num)]

    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-90, -70, 5, 25) # location=(N5-25, #90-110)
    dlt = 2
    txt_extent = (img_extent[0] + dlt, img_extent[1] - dlt,
                  img_extent[2] + dlt, img_extent[3] - dlt)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(image,
                     origin='uppur', 
                     extent=img_extent, 
                     transform=projection, 
                     norm=norm, 
                     cmap=cmap)
    cbar = fig.colorbar(mat, 
                        extend='both', 
                        ticks=ticks, 
                        spacing='proportional', 
                        orientation='vertical')

    if class_num == 5:
        cbar.ax.set_yticklabels(['low', 'mid-low', 'normal', 'mid-high', 'high'])
    else:
        lat_lst = np.linspace(txt_extent[3], txt_extent[2], lat_grid)
        lon_lst = np.linspace(txt_extent[0], txt_extent[1], lon_grid)
        for i, lat in enumerate(lat_lst):
            for j, lon in enumerate(lon_lst):
                ax.text(lon, lat, image[i,j],
                        ha="center", va="center", color="black", fontsize="15")
    plt.show()

