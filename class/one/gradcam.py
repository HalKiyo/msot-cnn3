import cv2
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers.core import Lambda
from keras.applications.vgg16 import preprocess_input
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def target_category_loss(x, category_index, class_num):
    return tf.multiply(x, K.one_hot([np.uint8(category_index)], class_num))

def target_category_loss_output_shape(input_shape):
    return input_shape

def image_preprocess(val, gradcam_index=0):
    img = val.copy()
    x = img[gradcam_index]
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    return x

def grad_cam(input_model, image, category_index, layer_name, lat=24, lon=72, class_num=5):
    #---1. claculate loss of predicted class
    target_layer = lambda x: target_category_loss(x, category_index, class_num)
    x = input_model.layers[-1].output
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(x)
    model = keras.models.Model(input_model.layers[0].input, x)
    loss = K.sum(model.layers[-1].output)
    conv_output = [l for l in model.layers if l.name == layer_name][0].output
    #---2. gradient from loss to last conv layer
    grads = normalize(K.gradients(loss, conv_output)[0])
    inp = input_model.layers[0].input
    output, grads_val = K.function([inp], [conv_output, grads])([image])
    output, grads_val = output[0,:], grads_val[0, :, :, :]
    #---3. channel average of gradient
    weights = np.mean(grads_val, axis=(0,1))
    #---4. multiply averaged channel weights to last output
    cam = np.dot(output, weights)
    cam = cv2.resize(cam, (lon, lat), cv2.INTER_LINEAR)#MODIFALABLE
    cam = np.maximum(cam, 0)
    if np.max(cam) == np.min(cam):
        heatmap = cam - np.min(cam)
        print("max(cam) = min(cam) = 0")
    else:
        heatmap = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    return heatmap

def show_heatmap(heatmap):
    proj = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-180, 180, -60, 60)

    fig = plt.figure()
    ax = plt.subplot(projection=proj)
    ax.coastlines(resolution='50m', lw=0.5)
    ax.gridlines(xlocs=mticker.MultipleLocator(90), ylocs=mticker.MultipleLocator(45),
                 linestyle='-', color = 'gray')
    mat = ax.matshow(heatmap, cmap='BuPu', norm=Normalize(vmin=0, vmax=1),
                     extent=img_extent, transform=proj)
    cbar = fig.colorbar(mat, ax=ax, orientation='horizontal')
    plt.show()

def box_gradcam(heatmap, pred_class, label_class, threshold=0.6, class_num=5):
    # heatmap(1000, 24, 72)
    # box true -> number of pixel which exceeds 0.6 color if prediction is correct

    true = {f"ture{i}, false{i}": [] for i in range(class_num)}
    false = {f"ture{i}, false{i}": [] for i in range(class_num)}

    for i in range(len(heatmap)):
        colored_pixel = np.count_nonzero(heatmap[i] >= threshold)
        prediction = np.argmax(pred_class[i])
        label = np.argmax(label_class[i])
        if prediction == label:
            true[f"ture{int(label)}, false{int(label)}"].append(colored_pixel)
        else:
            false[f"ture{int(label)}, false{int(label)}"].append(colored_pixel)

    label = np.arange(class_num)
    xs = {key:val for key, val in zip(true.keys(), label)}
    shift = 0.1

    fig, ax = plt.subplots()
    for key in true.keys():
        ax.boxplot(true[key], positions=[xs[key] - shift], boxprops=dict(color='b'),
                   showfliers=False)
        ax.boxplot(false[key], positions=[xs[key] + shift], boxprops=dict(color='r'),
                   showfliers=False)
    ax.set_xticks(range(class_num))
    ax.set_xticklabels(true.keys())
    plt.show()

