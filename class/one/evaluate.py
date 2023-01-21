import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import tensorflow as tf

from model3 import build_model
from util import load, shuffle, mask
from view import draw_val, view_probability, box_crossentropy
from gradcam import grad_cam, show_heatmap, image_preprocess, box_gradcam

def main():
    #---0. file init
    class_num = 10 
    epochs = 200 
    descrete_mode = 'EFD' 
    batch_size = 256 
    vsample = 1000 
    seed = 1 
    var_num = 4 
    #---0.1 prob init
    prob_view_flag = False
    prob_flag = True #true prediction or false prediction
    prob_label = 9 
    false_index = 0 
    true_index = 0 
    #---0.11 box init
    box_flag = False
    #---0.2 gradcam init
    grad_view_flag = False
    grad_box_flag = True
    layer_name = 'conv2d_2' 
    #---0.3 validation init
    validation_view_flag = False
    #---0.4 model init
    lat, lon = 24, 72
    lr = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = tf.keras.metrics.CategoricalAccuracy()

    #---1. dataset
    tors = 'predictors_coarse_std_Apr_msot'
    tant = f"pr_1x1_std_MJJASO_one_{descrete_mode}_{class_num}"
    savefile = f"/docker/mnt/d/research/D2/cnn3/train_val/class/{tors}-{tant}.pickle"
    with open(savefile, 'rb') as f:
        data = pickle.load(f)
    x_val, y_val = data['x_val'], data['y_val']
    y_val_one_hot = tf.keras.utils.to_categorical(y_val, class_num)

    #---2. validation
    model = build_model((lat, lon, var_num), class_num)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    weights_dir = f"/docker/mnt/d/research/D2/cnn3/weights/class/{tors}-{tant}"
    weights_path = weights_dir + f"/class{class_num}_epoch{epochs}_batch{batch_size}.h5"
    model.load_weights(weights_path)
    loss, acc = model.evaluate(x_val, y_val_one_hot)
    print(f"CategoricalAccuracy: {acc}")

    #---2.1 individual validation
    pred_val = model.predict(x_val)
    for validation_label in range(class_num):
        print(f"validation_label={validation_label}")
        wrong = []
        for i, j in zip(pred_val, y_val_one_hot):
            if np.argmax(i) != np.argmax(j) and int(np.argmax(j)) == validation_label:
                wrong.append(np.argmax(i))
        print(wrong)

    #---2.2 probabilistic CategoricalCrossentropy
    if prob_view_flag is True:
        false_lst = []
        true_lst = []
        for ind, (pr, y) in enumerate(zip(pred_val, y_val_one_hot)):
            if np.argmax(pr) != np.argmax(y) and int(np.argmax(y)) == prob_label:
                false_lst.append(ind)
            elif np.argmax(pr) == np.argmax(y) and int(np.argmax(y)) == prob_label:
                true_lst.append(ind)
        if prob_flag is True:
            val_index = true_lst[true_index]
            print(f"true_index: {true_lst}")
        else:
            val_index = false_lst[false_index]
            print(f"false_index: {false_lst}")
        print(f"prediction output of label:{np.argmax(y_val_one_hot[val_index])}")
        view_probability(pred_val, val_index=val_index)

    #---2.3 boxprot of CategoricalCrossentropy
    if box_flag is True:
        box_crossentropy(pred_val, y_val_one_hot, class_num=class_num)

    #---3. true/false barplot
    if validation_view_flag is True:
        class_label, counts = draw_val(pred_val, y_val_one_hot, class_num=class_num)
        print(f"class_label: {class_label} \ncounts: {counts}")

    #---4. individual gradcam
    if grad_view_flag is True:
        if prob_flag is True:
            gradcam_index = true_lst[true_index]
        else:
            gradcam_index = false_lst[false_index]
        preprocessed_image = image_preprocess(x_val, gradcam_index=gradcam_index)
        heatmap = grad_cam(model, preprocessed_image, y_val[gradcam_index], layer_name,
                           lat=24, lon=72, class_num=class_num)
        show_heatmap(heatmap)

    #---4. boxplot of gradcam 
    if grad_box_flag is True:
        heatmap_arr = np.empty((vsample, lat, lon))
        for index, (pr, y) in enumerate(zip(pred_val, y_val_one_hot)):
            preprocessed_image = image_preprocess(x_val, gradcam_index=index)
            heatmap = grad_cam(model, preprocessed_image, y_val[index], layer_name,
                               lat=24, lon=72, class_num=class_num)
            heatmap_arr[index] = heatmap
            print(index)
        box_gradcam(heatmap_arr, pred_val, y_val_one_hot, class_num=class_num)


if __name__ == '__main__':
    main()

