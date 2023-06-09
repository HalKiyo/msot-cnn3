import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import Normalize
import colormaps as clm

######################### CLASS ############################################
############################################################################
def show_class(image, class_num=5, lat_grid=4, lon_grid=4, txt_flag=False):
    cmap = plt.cm.get_cmap('BrBG', class_num)
    bounds = [i -0.5 for i in range(class_num+1)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ticks = [i for i in range(class_num)]

    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-90, -70, 5, 25) # location = (N5-25, E90-110)
    dlt = 2
    txt_extent = (img_extent[0] + dlt, img_extent[1] - dlt,
                  img_extent[2] + dlt, img_extent[3] - dlt)

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
    if txt_flag is True:
        if class_num == 5:
            cbar.ax.set_yticklabels(['low', 'mid-low', 'normal', 'mid-high', 'high'])
        else:
            lat_lst = np.linspace(txt_extent[3], txt_extent[2], lat_grid)
            lon_lst = np.linspace(txt_extent[0], txt_extent[1], lon_grid)
            for i, lat in enumerate(lat_lst):
                for j, lon in enumerate(lon_lst):
                    ax.text(lon, lat, image[i,j],
                            ha="center", va="center", color='black', fontsize='15')
    plt.rcParams["font.size"] = 18

    plt.show(block=False)

def true_false_bar(val_pred, val_label_onehot, class_num=5):
    plt.rcParams["font.size"] = 18
    fig = plt.figure()
    ax = plt.subplot()

    val_list = [i for i in range(class_num)]
    width, linewidth, align = 0.5, 0.5, 'center'

    # count
    val_true = []
    val_false = []
    for i, j in zip(val_pred, val_label_onehot):
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
           color='darkgoldenrod', bottom=val_tcount, width=width, linewidth=linewidth, align=align, alpha=.8)

    # count
    val_label_class =  [np.argmax(i) for i in val_label_onehot]
    u, counts = np.unique(val_label_class, return_counts=True)

    plt.show(block=False)
    return u, counts

def accuracy_map(acc, lat_grid=4, lon_grid=4):
    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-90, -70, 5, 25) # location = (N5-25, E90-110)
    dlt = 2
    txt_extent = (img_extent[0] + dlt, img_extent[1] - dlt,
                  img_extent[2] + dlt, img_extent[3] - dlt)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(acc,
                     origin='upper',
                     extent=img_extent,
                     transform=projection,
                     vmin=0, vmax=1,
                     cmap='Oranges')

    lat_lst = np.linspace(txt_extent[3], txt_extent[2], lat_grid)
    lon_lst = np.linspace(txt_extent[0], txt_extent[1], lon_grid)
    for i, lat in enumerate(lat_lst):
        for j, lon in enumerate(lon_lst):
            ax.text(lon, lat, acc[i,j],
                    ha="center", va="center", color='white', fontsize='15')
    cbar = fig.colorbar(mat, ax=ax)
    plt.rcParams["font.size"] = 18
    plt.show(block=False)

######################### CONTINUOUS #######################################
############################################################################
def show_continuous(image, vmin=-1, vmax=1):
    cmap = plt.cm.get_cmap('BrBG')
    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-90, -70, 5, 25) # location = (N5-25, E90-110)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(image,
                     origin='upper',
                     extent=img_extent,
                     transform=projection,
                     norm=Normalize(vmin=vmin, vmax=vmax),
                     cmap=cmap)
    fig.colorbar(mat, ax=ax)
    plt.show(block=False)

def ACC_map(ACC, lat_grid=20, lon_grid=20, vmin=0.75, vmax=1.00, discrete=5):
    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-90, -70, 5, 25) # location = (N5-25, E90-110)

    mpl.colormaps.unregister('ACC_map')
    mpl.colormaps.register(clm.temps, name='ACC_map')
    cm = plt.cm.get_cmap('ACC_map', discrete)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(ACC,
                     origin='upper',
                     extent=img_extent,
                     transform=projection,
                     vmin=vmin,
                     vmax=vmax,
                     cmap=cm)
    fig.colorbar(mat, ax=ax)
    plt.show(block=False)

######################### EVALUATION #######################################
############################################################################
def scatter_and_marginal_density(accuracy_lst,
                                 nrmse_lst,
                                 reliability_lst,
                                 true_accuracy_lst,
                                 true_nrmse_lst,
                                 true_reliability_lst,
                                 false_accuracy_lst,
                                 false_nrmse_lst,
                                 false_reliability_lst,
                                 else_accuracy_lst,
                                 else_nrmse_lst,
                                 else_reliability_lst):
    """
    seaborn acctepts only dataframe stracture
    """
    df = pd.DataFrame({
                       'acc':    accuracy_lst,
                       'nrmse':       nrmse_lst,
                       'rlbl': reliability_lst,
                       'true_acc': true_accuracy_lst,
                       'true_nrmse': true_nrmse_lst,
                       'true_rlbl': true_reliability_lst,
                       'false_acc': false_accuracy_lst,
                       'false_nrmse': false_nrmse_lst,
                       'false_rlbl': false_reliability_lst,
                       'else_acc': else_accuracy_lst,
                       'else_nrmse': else_nrmse_lst,
                       'else_rlbl': else_reliability_lst
                      })
    plt.rcParams["font.size"] = 20
    fig = plt.figure(figsize=[5, 5])
    ax = plt.subplot()
    ax.scatter(true_reliability_lst, true_nrmse_lst,
               c="#00AFBB",
               s=true_accuracy_lst,
               alpha=0.5,
               label=(f"True (number of sample = {len(true_reliability_lst)})")
               )
    ax.scatter(false_reliability_lst,
               false_nrmse_lst,
               c="#FC4E07",
               s=false_accuracy_lst,
               alpha=0.5,
               label=(f"False (number of sample={len(false_reliability_lst)})"))
    ax.scatter(else_reliability_lst,
               else_nrmse_lst,
               c="#E7B800",
               s=else_accuracy_lst,
               alpha=0.5,
               label=(f"Else (number of sample={len(else_reliability_lst)})"))
    plt.xlabel('Reliability (grids_mean)')
    plt.ylabel('NRMSE (grids_mean)')
    plt.legend()
    plt.show(block=False)

def cluster_scatter(accuracy_lst,
                   nrmse_lst,
                   reliability_lst):
    plt.rcParams["font.size"] = 16
    cm = colors.ListedColormap(["#FC4E07",
                                "#E7B800",
                                "#00AFBB"])
    fig = plt.figure(figsize=[6, 6])
    ax = plt.subplot()
    scat = ax.scatter(reliability_lst,
                      nrmse_lst,
                      c=np.array(accuracy_lst),
                      cmap=cm,
                      s=300,
                      alpha=0.5,
                      vmin=0,
                      vmax=400,
                     )
    ax.scatter([0.95],
               [0.03],
               c="#00AFBB",
               cmap=cm,
               s=300,
               alpha=0.5,
               label=(f"true (851 samples)")
               )
    ax.scatter([0.78],
               [0.5],
               c="#FC4E07",
               cmap=cm,
               s=300,
               alpha=0.5,
               label=(f"false (145 samples)")
               )
    ax.scatter([0.83],
               [0.47],
               c="#E7B800",
               cmap=cm,
               s=300,
               alpha=0.5,
               label=(f"else (4 samples)")
               )
    plt.xlabel('Reliability (grids_mean)')
    plt.ylabel('NRMSE (grids_mean)')
    clb = fig.colorbar(scat,
                       ax=ax,
                       orientation='horizontal')
    clb.set_label("number of true grids in 20x20 grids",
                  rotation=0)
    plt.legend(bbox_to_anchor=(1, 1),
               loc='upper right',
               borderaxespad=0,
               fontsize = 12)
    plt.show(block=False)
