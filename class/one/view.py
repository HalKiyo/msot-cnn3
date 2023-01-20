import numpy as np
import pandas as pd
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

def box_crossentropy(val_pred, val_label_one_hot, class_num=5):
    #val_pred=(1000, class_num)
    true = {f"true{i}, false{i}": [] for i in range(class_num)}
    false = {f"true{i}, false{i}": [] for i in range(class_num)}

    for i in range(len(val_pred)):
        max_cross = np.max(val_pred[i])
        pred_class = np.argmax(val_pred[i])
        label_class = np.argmax(val_label_one_hot[i])
        if pred_class == label_class:
            true[f"true{int(pred_class)}, false{int(pred_class)}"].append(max_cross)
        else:
            false[f"true{int(pred_class)}, false{int(pred_class)}"].append(max_cross)

    # for key which doesn't have value in it
    for i in range(class_num):
        for key in [true, false]:
            if bool(key[f"true{i}, false{i}"]) is False:
                key[f"true{i}, false{i}"].append(0)

    label = np.arange(class_num)
    xs = {key:val for key, val in zip(true.keys(), label)}
    shift = 0.1

    fig, ax = plt.subplots()
    for key in true.keys():
        ax.boxplot(true[key], positions=[xs[key] - shift], boxprops=dict(color='b'))
        ax.boxplot(false[key], positions=[xs[key] + shift], boxprops=dict(color='r'))
    ax.set_xticks(range(class_num))
    ax.set_xticklabels(true.keys())
    plt.show()

