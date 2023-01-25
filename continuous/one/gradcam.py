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
    # preprocessed_x_val is expected to be a image with depth = (lat, lon, variables_num)
    # y_val is expected to be a actual label = (1,)
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

def box_gradcam(heatmap, prediction, threshold=0.7, criteria=0.4):
    # heatmap(1000, 24, 72) 
    # criteria: any positive value between min and max of prediction
    # criteria <groupA< max, min <groupB< -criteria, -criteria< groupC < criteria
    # box true -> number of pixel which exceeds 0.6 color if prediction is correct

    label_lst = ['low', 'middle', 'high']
    count_dct = {f"{i}": [] for i in label_lst}

    for i in range(len(heatmap)):
        colored_pixel = np.count_nonzero(heatmap[i] >= threshold)
        if prediction[i] < -criteria:
            count_dct['low'].append(colored_pixel)
        elif criteria < prediction[i]:
            count_dct['high'].append(colored_pixel)
        elif -criteria < prediction[i] < criteria:
            count_dct['middle'].append(colored_pixel)

    xs = {key:val for key, val in zip(count_dct.keys(), label_lst)}
    cmap = ['y', 'g', 'b']

    fig, ax = plt.subplots()
    for i, key in enumerate(count_dct.keys()):
        ax.boxplot(count_dct[key], 
                   positions=[xs[key]], 
                   boxprops=dict(color=cmap[i]),
                   showfliers=False)
    ax.set_xticks(range(len(label_lst)))
    ax.set_xticklabels(count_dct.keys())
    plt.show()

