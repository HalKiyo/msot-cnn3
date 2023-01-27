import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def draw_distribution(data, bnd):
    """
    data shape must be flattened into one dimension,
    if one has (42, 165) shape, convert it to 42*165/ use x.reshape(-1)
    likewise, Thailand has (42, 165, 4, 4), then flatten it to 42*165*4*4
    """
    fig = plt.figure()
    ax = plt.subplot()
    ax.hist(data, bins=1000, alpha=.5, color='darkcyan')
    for i in bnd:
        ax.axvline(i, ymin=0, ymax=len(data), alpha=.8, color='salmon')
    plt.show()

def draw_val(val_true, val_false, class_num=5):
    """
    np.argmax is not required
    since continuous prediction doesn't have one_hot_encoded label
    unlike class prediction model
    """
    val_list = [i for i in range(class_num)]

    true_count = [val_true.count(i) for i in val_list]
    false_count = [val_false.count(i) for i in val_list]

    # barplot
    fig = plt.figure()
    ax = plt.subplot()

    width, linewidth, align = 0.5, 0.5, 'center'
    ax.bar(val_list, true_count,
           color='darkslategray', width=width, linewidth=linewidth, align=align)
    ax.bar(val_list, false_count,
           color='darkslategray', width=width, linewidth=linewidth, align=align)
    plt.show()

