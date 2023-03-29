import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics import auc
from scipy.stats import norm

def acc_map(acc, lat_grid=20, lon_grid=20, vmin=0.75, vmax=1.00):
    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-180, 180, -60, 60)  # location = (N5-25, E90-110)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(acc,
                     origin='upper',
                     extent=img_extent,
                     transform=projection,
                     vmin=vmin, vmax=vmax,
                     cmap='tab20c')
    fig.colorbar(mat, ax=ax)
    plt.show(block=False)

def show_map(image, vmin=-1, vmax=1):
    cmap = plt.cm.get_cmap('BrBG')
    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-180, 180, -60, 60)  # location = (N5-25, E90-110)

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

def ae_bar(data, vmin=0, vmax=2):
    # grid毎にabs(実際のlabelデータ-予測結果)を400個棒グラフにして出力する
    fig = plt.figure()
    ax = plt.subplot()
    pixcel = np.arange(len(data))
    ax.bar(pixcel, data, color='magenta')
    ax.set_ylim(vmin, vmax)
    plt.show(block=False)

def bimodal_dist(data, gmm):
    fig, ax = plt.subplots()

    # histgram
    ax.hist(data, color='g', alpha=0.5)

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
             label='true')
    ax2.plot(x,
             np.squeeze(gmm.weights_[1]*false),
             label='false')
    ax2.legend()
    plt.show(block=False)

def draw_val(true_count, false_count):
    fig = plt.figure()
    ax = plt.subplot()
    print(true_count, false_count)
    ax.barh(1, true_count, height=0.5, color='darkslategray', align='center', label='True')
    ax.barh(1, false_count, left=true_count, height=0.5, color='orange', align='center', label='False')
    ax.set_ylim(0, 2)
    ax.set_yticks([1.0], ['val'])
    plt.legend()
    plt.show()

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
             color="deeppink",
             linestyle=":",
             linewidth=4)
    # plot cnn_continuous percentile results
    plt.scatter(fpr, tpr, s=100, color='red')

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
