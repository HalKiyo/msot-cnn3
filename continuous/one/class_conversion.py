import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import bisect
import pickle
import numpy as np
import tensorflow as tf
from model3 import build_model

def open_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    x_val, y_val = data['x_val'], data['y_val']
    return x_val, y_val

def init_model(weights_path, lat=24, lon=72, var_num=4, lr=0.0001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.MeanSquaredError()
    metrics = tf.keras.metrics.MeanSquaredError()
    model = build_model((lat, lon, var_num))
    model.compile(optimizer=optimizer, loss=loss , metrics=[metrics])
    model.load_weights(weights_path)
    return model

def prediction(model, x_val):
    pred = model.predict(x_val)[:,0]
    return pred

def to_class(pr_flat, bnd, print_flag=False):
    # pr_flat must be reshaped into 1D(x.reshape(-1))
    pr_class = np.empty(len(pr_flat))
    for i, value in enumerate(pr_flat):
        label= bisect.bisect(bnd, value) # giving label
        pr_class[i] = int(label - 1)
    u, counts = np.unique(pr_class, return_counts=True)
    if print_flag is True:
        print(f"class_label: {u}")
        print(f"count: {counts}")
    return pr_class # pr_class is flattend(1D)


if __name__ == '__main__':
    # init
    save_flag = False
    class_num = 5
    discrete_mode = 'EFD'
    epochs = 100
    batch_size = 256
    seed = 1
    tors = 'predictors_coarse_std_Apr_msot'
    tant = 'pr_1x1_std_MJJASO_one'
    workdir = "/docker/mnt/d/research/D2/cnn3"
    val_path = workdir + f"/train_val/continuous/{tors}-{tant}.pickle"
    bnd_path = workdir + f"/boundaries/{tant}_{discrete_mode}_{class_num}.npy"
    weights_dir = workdir + f"/weights/continuous/{tors}-{tant}"
    weights_path = weights_dir + f"/epoch{epochs}_batch{batch_size}_seed{seed}.h5"

    # calculate
    x_val, y_val = open_pickle(val_path)
    bnd = np.load(bnd_path)
    pred, model = prediction(x_val, weights_path)
    model.summary()
    pred_class = to_class(pred.reshape(-1), bnd)
    y_class = to_class(y_val.reshape(-1), bnd, print_flag=True)

