import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocess import load, shuffle, mask
from model3 import build_model
from gradcam import image_preprocess, grad_cam, show_heatmap, average_heatmap

def main():
    #---0. initial setting
    train_flag = False
    epochs = 100
    batch_size = 256
    vsample = 1000
    seed = 1
    lr = 0.0001
    var_num = 4
    gradcam_index = 700
    layer_name = 'conv2d_2'

    #---1. dataset
    tors = 'predictors_coarse_std_Apr_msot'
    tant = 'pr_1x1_std_MJJASO_one'
    savefile = f"/docker/mnt/d/research/D2/cnn3/train_val/continuous/{tors}-{tant}.pickle"
    if os.path.exists(savefile) is True and train_flag is False:
        with open(savefile, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']
    elif os.path.exists(savefile) is False and train_flag is False:
        print(f"{savefile} is not found, change train_flag to True first")
        exit()
    else:
        predictors, predictant = load(tors, tant)
        x_train, y_train, x_val, y_val, train_dct, val_dct = shuffle(predictors, predictant, vsample)
        x_train, x_val = mask(x_train), mask(x_val)
        x_train, x_val = x_train.transpose(0,2,3,1), x_val.transpose(0,2,3,1)

    #---2, training
    lat, lon = 24, 72
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.MeanSquaredError()
    metrics = tf.keras.metrics.MeanSquaredError()
    model = build_model((lat, lon, var_num))
    model.compile(optimizer=optimizer, loss=loss , metrics=[metrics])
    weights_path = f"/docker/mnt/d/research/D2/cnn3/weights/{tors}-{tant}.h5"
    if os.path.exists(weights_path) is True and train_flag is False:
        model.load_weights(weights_path)
    elif os.path.exists(weights_path) is False and train_flag is False:
        print(f"{weights_path} is not found, change train_flag to True first")
        exit()
    else:
        his = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        #model.summary()

    #---3, validation
    pred = model.predict(x_val)[:,0]
    corr = np.corrcoef(pred, y_val)[0,1]
    plt.scatter(pred, y_val, color='pink')
    plt.plot(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), color='green')
    plt.title(f"{corr}")
    #plt.show()

    #---4. gradcam
    preprocessed_image = image_preprocess(x_val, gradcam_index)
    heatmap = grad_cam(model, preprocessed_image, y_val[gradcam_index], layer_name, lat=lat, lon=lon)
    show_heatmap(heatmap)
    #average_heatmap(x_val, model, y_val, layer_name, lat=lat, lon=lon, num=300)

    #---5. save environment
    if train_flag is True:
        model.save_weights(weights_path)
        dct = {'x_train': x_train, 'y_train': y_train,
               'x_val': x_val, 'y_val': y_val,
               'train_dct': train_dct, 'val_dct': val_dct}
        with open(savefile, 'wb') as f:
            pickle.dump(dct, f)
        print(f"{savefile} and weights are saved")
    else:
        print(f"train_flag is {train_flag} not saved")

if __name__ == '__main__':
    main()

