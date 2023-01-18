import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import tensorflow as tf

from model3 import build_model
from view import draw_val, view_probability
from util import load, shuffle, mask
from gradcam import grad_cam, show_heatmap, image_preprocess

def main():
    #---0. initial setting
    class_num = 10#MODIFALABLE
    epochs = 200#MODIFALABLE
    descrete_mode = 'EWD' #MODIFALABLE
    batch_size = 256#MODIFALABLE
    vsample = 1000#MODIFALABLE
    seed = 1#MODIFALABLE
    lr = 0.0001#MODIFALABLE
    var_num = 4#MODIFALABLE
    prob_label = 0#MODIFALABLE
    false_index = 1#MODIFALABLE
    #gradcam_index = 100#MODIFALABLE
    layer_name = 'conv2d_2'#MODIFALABLE
    lat, lon = 24, 72
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

    #---2.5 individual validation
    pred_val = model.predict(x_val)
    for validation_label in range(class_num):
        print(f"validation_label={validation_label}")
        wrong = []
        for i, j in zip(pred_val, y_val_one_hot):
            if np.argmax(i) != np.argmax(j) and int(np.argmax(j)) == validation_label:
                wrong.append(np.argmax(i))
        print(wrong)

    #---2.6 probabilistic CategoricalCrossentropy
    false_lst = []
    true_lst = []
    for ind, (pr, y) in enumerate(zip(pred_val, y_val_one_hot)):
        if np.argmax(pr) != np.argmax(y) and int(np.argmax(y)) == prob_label:
            false_lst.append(ind)
        elif np.argmax(pr) == np.argmax(y) and int(np.argmax(y)) == prob_label:
            true_lst.append(ind)
    val_index = false_lst[false_index]
    print(f"prediction output of label:{np.argmax(y_val_one_hot[val_index])}")
    view_probability(pred_val, val_index=val_index)
    print(f"false_index: {false_lst}")
    print(f"true_index: {true_lst}")

    #---3. visualization
    class_label, counts = draw_val(pred_val, y_val_one_hot, class_num=class_num)
    print(f"class_label: {class_label} \ncounts: {counts}")

    #---4. gradcam
    gradcam_index = false_lst[false_index]
    preprocessed_image = image_preprocess(x_val, gradcam_index=gradcam_index)
    heatmap = grad_cam(model, preprocessed_image, y_val[gradcam_index], layer_name,
                       lat, lon, class_num)
    show_heatmap(heatmap)


if __name__ == '__main__':
    main()

