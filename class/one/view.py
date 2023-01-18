import numpy as np
import matplotlib.pyplot as plt

def draw_val(val_pred, val_label_one_hot, class_num=5):
    fig = plt.figure()
    ax = plt.subplot()

    val_list = [i for i in range(class_num)]
    width, linewidth, align = 0.5, 0.5, 'center'

    # count
    val_true = []
    val_false = []
    for i, j in zip(val_pred, val_label_one_hot):
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

    # count
    val_label_class = [np.argmax(i) for i in val_label_one_hot]
    u, counts = np.unique(val_label_class, return_counts=True)

    plt.show()
    return u, counts

def view_probability(val_pred, val_index=0):
    print(val_pred[val_index])

    fig = plt.figure()
    ax = plt.subplot()
    width, linewidth, align = 0.5, 0.5, 'center'

    output = val_pred[val_index]
    ticks = np.arange(len(output))
    ax.bar(ticks, output, color='darkslategray', width=width, linewidth=linewidth, align=align)
    plt.show()

