import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

from model3 import build_model
from view import draw_val
from util import load, shuffle, mask
from gradcam import grad_cam, show_heatmap, image_preprocess

disable_eager_execution()

def main():
    #---0. initial setting
    train_flag = True#MODIFALABLE
    vsample = 1000#MODIFALABLE
    seed = 1#MODIFALABLE
    class_num = 10#MODIFALABLE
    batch_size = 256#MODIFALABLE
    epochs = 200#MODIFALABLE
    lr = 0.0001#MODIFALABLE
    var_num = 4#MODIFALABLE
    gradcam_index = 100#MODIFALABLE
    layer_name = 'conv2d_2'#MODIFALABLE
    descrete_mode = 'EFD' #MODIFALABLE

    #---1. dataset
    tors = 'predictors_coarse_std_Apr_msot'
    tant = f"pr_1x1_std_MJJASO_one_{descrete_mode}_{class_num}"
    savefile = f"/docker/mnt/d/research/D2/cnn3/train_val/class/{tors}-{tant}.pickle"
    if os.path.exists(savefile) is True and train_flag is False:
        with open(savefile, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']
        y_val_one_hot = tf.keras.utils.to_categorical(y_val, class_num)
    else:
        predictors, predictant = load(tors, tant)
        x_train, y_train, x_val, y_val, train_dct, val_dct = shuffle(predictors, predictant, vsample, seed)
        x_train, x_val = mask(x_train), mask(x_val)
        x_train, x_val = x_train.transpose(0,2,3,1), x_val.transpose(0,2,3,1)
        y_train_one_hot = tf.keras.utils.to_categorical(y_train, class_num)
        y_val_one_hot = tf.keras.utils.to_categorical(y_val, class_num)

    #---2, training
    lat, lon = 24, 72
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = tf.keras.metrics.CategoricalAccuracy()
    model = build_model((lat, lon, var_num), class_num)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    weights_dir = f"/docker/mnt/d/research/D2/cnn3/weights/class/{tors}-{tant}"
    os.makedirs(weights_dir, exist_ok=True) # create weight directory
    weights_path = weights_dir + f"/class{class_num}_epoch{epochs}_batch{batch_size}.h5"
    if os.path.exists(weights_path) is True and train_flag is False:
        model.load_weights(weights_path)
    else:
        his = model.fit(x_train, y_train_one_hot, batch_size=batch_size, epochs=epochs)
        #model.summary()

    #---3. test
    results = model.evaluate(x_val, y_val_one_hot)
    print(f"CategoricalAccuracy: {results[1]}")
    pred_val = (model.predict(x_val))
    draw_val(pred_val, y_val_one_hot, class_num=class_num)

    #---3. gradcam
    preprocessed_image = image_preprocess(x_val, gradcam_index)
    heatmap = grad_cam(model, preprocessed_image, y_val[gradcam_index], layer_name,
                       lat, lon, class_num)
    show_heatmap(heatmap)

    #---4. save state
    if train_flag is True:
        model.save_weights(weights_path)
        dct = {'x_train': x_train, 'y_train': y_train,
               'x_val': x_val, 'y_val': y_val,
               'train_dct': train_dct, 'val_dct': val_dct}
        with open(savefile, 'wb') as f:
            pickle.dump(dct, f)
        print(f"{savefile} and weights are saved")
    else:
        print(f"train_flag is {train_flag}: not saved")


if __name__ == '__main__':
    main()

