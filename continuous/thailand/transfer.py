# Transfer learning with observations
#######################################
# SST: SODA & GODAS
# RAINFALL: APHRODITE(only ground)
# transfer training period: 1951-1992
# test period: 1993-2015
#######################################
# available data candidates
# GPCP->1979-2020, 2.5x2.5, monthly
# IMERG_final->2000-2020, 0.1, half-hourly


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] == '3'
import warnings
warnings.filterwarnings('ignore')
import pickle
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model3 import build_model
from util import load, shuffle, mask
from view import acc_map, show_map

def main():
    TRANS = trans()
    TRANS.training()

class trans:
    def __init__(self):
        self.epochs = 100
        self.batch_size = 256
        self.resolution = '1x1'
        self.tors = 'soda_coarse_std_Apr'
        self.tand = f"gpcp_{self.resolution}_std_MJJASO_thailand"
        self.old_tors = 'predictors_coarse_std_Apr_o'
        self.old_tand = f"pr_{self.resolution}_std_MJJASO_thailand"
        self.seed = 1
        self.lat, self.lon = 24, 72
        self.var_num = 1
        self.lat_grid, self.long_grid = 20, 20
        self.grid_num = self.lat_grid * self.long_grid
        self.loss = tf.keras.losses.MeanSquaredError()
        self.metrics = tf.keras.metrics.MeanSquaredError()

        self.old_weights_dir = f"/docker/mnt/d/research/D2/cnn3/weights/continuous/{self.old_tors}-{self.old_tand}"
        self.new_weights_dir = f"/docker/mnt/d/research/D3/cnn3/transfer/weights/continuous/{self.tors}-{self.tand}"
        self.pred_dir = f"/docker/mnt/d/research/D3/cnn3/transfer/result/continuous/thailand/{self.resolution}"
        self.pred_path = self.pred_dir + f"/epoch{self.epochs}_batch{self.batch_size}_seed{self.seed}.npy"
        self.savefile = f"/docker/mnt/d/research/D3/cnn3/transfer/train_val/continuous/{self.tors}-{self.tand}.pickle"

    def training(self, x_train, y_train, x_val, y_val, train_dct, val_dct):
        x_train = mask(x_train)
        x_val = mask(x_val)
        x_train = x_train.transpose(0, 2, 3, 1),
        x_val = x_val.transpose(0, 2, 3, 1)
        y_train = y_train.reshape(len(y_train), self.grid_num)
        y_val = y_val.reshape(len(y_val), self.grid_num)
        os.makedirs(self.new_weights_dir, exist_ok=True)

        for i in range(self.grid_num):
            y_train_px = y_train[:, i]

            model = build_model((self.lat, self.lon, self.var_num))
            old_weights_path = f"{self.old_weights_dir}/epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
            model.load_weights(old_weights_path)
            for i in range(5):
                model.layers[i].trainable = False
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                          loss=self.loss,
                          metrics=[self.metrics])

            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)
            model.fit(x_train,
                      y_train_px,
                      epochs=self.epochs,
                      validation_data=(x_val, y_val),
                      verbose=1,
                      callbacks=[early_stop])

            new_weights_path = f"{self.new_weights_dir}/epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
            model.save_weights(new_weights_path)

        dct = {'x_train': x_train,
               'y_train': y_train,
               'x_val': x_val,
               'y_val': y_val,
               'train_dct': train_dct,
               'val_dct': val_dct}

        with open(self.savefile, 'wb') as f:
            pickle.dump(dct, f)


if __name__ == '__main__':
    main()
