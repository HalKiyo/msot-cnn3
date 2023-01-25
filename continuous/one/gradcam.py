import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers.core import Lambda
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def image_preprocess(x_val, gradcam_index=0):
    # x_val is expected to be a image with depth (lat, lon, variables_num)
    img = x_val.copy()
    x = img[gradcam_index]
    x = np.expand_dims(x, axis=0)
    return x

def grad_cam(input_model, preprocessed_x_val, y_val, layer_name, lat=24, lon=72):
    # preprocessed_x_val is expected to be a image with depth (lat, lon, variables_num)
    # y_val is expected to be actual label (1,)
    pred_val = input_model.output[0]
    y_val = tf.convert_to_tensor(y_val.astype(np.float32))
    loss = K.mean(K.square(pred_val - y_val))
    conv_output = input_model.get_layer(layer_name).output
    #---2. gradient from loss to last conv layer
    grads = normalize(K.gradients(loss, conv_output)[0])
    inp = input_model.layers[0].input
    output, grads_val = K.function([inp], [conv_output, grads])([preprocessed_x_val])
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

def average_heatmap(x_val, input_model, y_val, layer_name, lat=24, lon=72, num=300):
    saliency = np.empty(x_val.shape[:3])[:num,:,:]
    for i in range(num):
        preprocessed_x_val = image_preprocess(x_val, gradcam_index=i)
        heatmap = grad_cam(input_model, preprocessed_x_val, y_val, layer_name, lat, lon)
        saliency[i,:,:] = heatmap
        if i%100 == 0:
            print(f"validation_sample_number: {i}")
    saliency = saliency.mean(axis=0)
    show_heatmap(saliency)

def box_gradcam(heatmap, pred_class, label_class, threshold=0.6, class_num=5):
    # heatmap(1000, 24, 72) 
    # box true -> number of pixel which exceeds 0.6 color if prediction is correct
    colored_pixel = np.count_nonzero(heatmap >= threshold)

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

