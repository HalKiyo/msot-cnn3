import numpy as np
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import colormaps as clm
from sklearn.metrics import auc
from scipy.stats import norm

def acc_map(acc, vmin=0.75, vmax=1.00, discrete=5):
    """
    cmap for scientific 
    colormap -> batlow
    diverging => roma 
    multi-sequential -> oleron10
    """# tab20c
    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-180, 180, -60, 60)  # location = (N5-25, E90-110)

    #mpl.colormaps.unregister('bat')
    #mpl.colormaps.register(clm.batlow, name='bat')
    #cm = plt.cm.get_cmap('bat', discrete)

    mpl.colormaps.unregister('tmp')
    mpl.colormaps.register(clm.temps, name='tmp')
    cm = plt.cm.get_cmap('tmp', discrete)

    #mpl.colormaps.unregister('tmp2')
    #mpl.colormaps.register(clm.cet_r_bgyr, name='tmp2')
    #cm = plt.cm.get_cmap('tmp2', discrete)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(acc,
                     origin='upper',
                     extent=img_extent,
                     transform=projection,
                     vmin=vmin, vmax=vmax,
                     cmap=cm)
    fig.colorbar(mat, ax=ax)
    plt.show(block=False)

def show_map(image, vmin=-1, vmax=1, discrete=5):
    cmap = plt.cm.get_cmap('BrBG', discrete)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-180, 180, -60, 60)  # location = (N5-25, E90-110)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(image,
                     origin='upper',
                     extent=img_extent,
                     transform=projection,
                     norm=norm,
                     cmap=cmap)
    fig.colorbar(mat, ax=ax)
    plt.show(block=False)

def AE_bar(data, vmin=0, vmax=2):
    # grid毎にabs(実際のlabelデータ-予測結果)を400個棒グラフにして出力する
    fig = plt.figure()
    ax = plt.subplot()
    pixcel = np.arange(len(data))
    ax.bar(pixcel, data, color='darkseagreen')
    ax.set_ylim(vmin, vmax)
    plt.show(block=False)

def bimodal_dist(data, gmm):
    fig, ax = plt.subplots()

    # histgram
    ax.hist(data, bins=1000, color='lightsteelblue', alpha=0.8)

    # gaussian mixture modelling
    ax2 = ax.twinx()
    x = np.linspace(0, 1, 1000)
    true = norm.pdf(x,
                    gmm.means_[0, -1],
                    np.sqrt(gmm.covariances_[0]))
    false = norm.pdf(x,
                     gmm.means_[1, -1],
                     np.sqrt(gmm.covariances_[1]))
    ax2.plot(x,
             np.squeeze(gmm.weights_[0]*true),
             color='darkslategray',
             label='true')
    ax2.plot(x,
             np.squeeze(gmm.weights_[1]*false),
             color='darkgoldenrod',
             label='false')
    ax2.legend()
    plt.show(block=False)

def TF_bar(true_count, false_count):
    fig = plt.figure()
    ax = plt.subplot()
    print(f"true: over_criteria_count{true_count}, false: under_criteria_count{false_count}")
    ax.barh(1,
            true_count,
            height=0.5,
            color='darkslategray',
            align='center',
            label=f"True({true_count})")
    ax.barh(1,
            false_count,
            left=true_count,
            height=0.5,
            color='darkgoldenrod',
            align='center',
            label=f"False({false_count})")
    ax.set_ylim(0, 2)
    ax.set_yticks([1.0], ['validation(1000samples)'])
    plt.legend()
    plt.show(block=False)

def draw_roc_curve(roc):
    # calculate auc
    fpr = roc[:, 1]
    tpr = roc[:, 0]
    AUC = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))

    # draw cnn_continuous line
    plt.plot(fpr,
             tpr,
             label=f"cnn_continuous ROC curve (AUC = {AUC})",
             color='olive',
             linestyle=":",
             linewidth=4)
    # plot cnn_continuous percentile results
    plt.scatter(fpr,
                tpr,
                s=100,
                color='darkseagreen')

    # plot auc=0.5 line
    plt.plot([0,1],
             [0,1],
             "k--",
             label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show(block=False)
