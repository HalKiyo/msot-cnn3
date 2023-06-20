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
    plt.rcParams["font.size"] = 16
    cm = colors.ListedColormap(["#FC4E07",
                                "#E7B800",
                                "#00AFBB"])
    fig = plt.figure(figsize=[8, 8])
    gs = fig.add_gridspec(2,
                          2,
                          width_ratios=(4, 1),
                          height_ratios=(1, 4),
                          left=0.1,
                          right=0.9,
                          bottom=0.2,
                          top=0.95,
                          wspace=0.05,
                          hspace=0.05)
    ax = fig.add_subplot(gs[1 ,0])
    cax = fig.add_axes([0.12, 0.1, 0.58, 0.01])
    ax_x = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_x.tick_params(axis="x",
                     labelbottom=False)
    ax_y.tick_params(axis="y",
                     labelleft=False)

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

    bplot = ax_x.boxplot([false_reliability_lst, true_reliability_lst],
                         notch=True,
                         patch_artist=True,
                         showfliers=False,
                         vert=False,
                         labels=['false', 'true'])
    boxcolor = ['#FC4E07', '#00AFBB']
    for patch, color in zip(bplot['boxes'], boxcolor):
        patch.set_facecolor(color)

    ax_y.hist(false_nrmse_lst,
              bins=10,
              color="#FC4E07",
              orientation='horizontal')
    ax_y.hist(true_nrmse_lst,
              bins=10,
              color="#00AFBB",
              orientation='horizontal')

    ax.set_xlabel('Reliability (grids_mean)')
    ax.set_ylabel('NRMSE (grids_mean)')
    clb = fig.colorbar(scat,
                       cax=cax,
                       orientation='horizontal')
    clb.set_label("number of true grids in 20x20 grids",
                   rotation=0)
    ax.legend(bbox_to_anchor=(1, 1),
               loc='upper right',
               borderaxespad=0,
               fontsize = 12)

    plt.show(block=False)

def ensemble_kde(true_density,
                 false_density,
                 class_num=5,
                 key="0"):
    """
    it will take 30min
    for label "0"
    true: 73882, false: 9258
    and alpha should be adaptive
    """
    ticks = np.arange(class_num)

    fig, ax = plt.subplots()

    #false_length = len(false_density[key])
    false_length = 100
    for i in range(false_length):
        sample = false_density[key][i]
        thousand = [int(s*1e2) for s in sample]
        data = []
        for label in range(class_num):
            data += ([ticks[label]]*thousand[label])
        sns.kdeplot(np.array(data), bw=0.8, color='#FC4E07', alpha=0.01)

    #true_length = len(true_density[key])
    true_length = false_length
    for i in range(true_length):
        sample = true_density[key][i]
        thousand = [int(s*1e2) for s in sample]
        data = []
        for label in range(class_num):
            data += ([ticks[label]]*thousand[label])
        sns.kdeplot(np.array(data), bw=0.8, color='#00AFBB', alpha=0.01)

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 5)
    plt.show(block=False)

def ensemble_step(true_density,
                 false_density,
                 class_num=5,
                 key="0"):
    ticks = [i-0.5 for i in range(class_num+1)]

    fig, ax = plt.subplots()

    for  i in range(len(false_density[key])):
        sample = false_density[key][i]
        sample = np.insert(sample, 0, sample[0])
        ax.step(ticks, sample, where='pre', alpha=.01, color='#FC4E07')
    for  i in range(len(true_density[key])):
        sample = true_density[key][i]
        sample = np.insert(sample, 0, sample[0])
        ax.step(ticks, sample, where='pre', alpha=.01, color='#00AFBB')

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 1)
    plt.show(block=False)

def ensemble_violin(true_density,
                    false_density,
                    dict_for_df,
                    class_num=5,
                    key="1"):
    """
    key="0", true_index=100, false_index=1050
    key="1", true_index=100, false_index=1090
    key="2", true_index=200, false_index=1080
    key="3", true_index=100, false_index=1050
    key="4", true_index=200, false_index=1050
    """
    true_index = 100
    false_index = 1090

    dic = dict_for_df[key]
    print(f"number of items in dict: {len(dic['label'])}")
    df = pd.DataFrame(dic)
    ticks=[f"{i}" for i in range(class_num)]

    plt.rcParams["font.size"] = 18
    fig, ax = plt.subplots()

    sns.barplot(x=ticks,
                y=true_density[key][true_index],
                color='#00AFBB',
                alpha=0.1,
                width=0.5)
    sns.barplot(x=ticks,
                y=false_density[key][false_index],
                color='#FC4E07',
                alpha=0.1,
                width=1)
    sns.violinplot(data=df,
                   x="label",
                   y="prob",
                   hue="result",
                   #scale_hue=False,
                   split=True,
                   cut=0,
                   gridsize=100,
                   width=1.5,
                   inner=None,
                   palette = ["#00AFBB", "#FC4E07"],
                   linewidth=0.0001,
                   saturation=0.8,
                   bw=0.15
                   )
    plt.xlim(-1, 5)
    plt.ylim(0, 1)

    plt.show(block=False)

def gcm_bars(true_gcm,
             false_gcm,
             model_name,
             model_num=42):
    plt.rcParams["font.size"] = 12
    plt.rcParams["figure.subplot.bottom"] = 0.4
    width, linewidth, align = 0.8, 0.8, 'center'

    xticks = np.arange(model_num)
    true_count = list(true_gcm.values())
    false_count = list(false_gcm.values())

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.bar(xticks, true_count,
           color='#00AFBB', width=width, linewidth=linewidth, align=align,
           label='True')

    ax.bar(xticks, false_count,
           color='#FC4E07', bottom=true_count, width=width, linewidth=linewidth, align=align, alpha=.8,
           label='False')

    ax.set_xticks(xticks, model_name, rotation=90)
    plt.legend()
    plt.show(block=False)
