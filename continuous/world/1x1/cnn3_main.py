import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model3 import build_model
from util import load, shuffle, mask
from view import acc_map, show_map

def main():
    train_flag = True

    px = Pixel()
    if train_flag is True:
        predictors, predictant = load(px.tors, px.tant)
        px.training(*shuffle(predictors, predictant, px.vsample, px.seed, lat = px.lat_grid, lon = px.lon_grid))
        print(f"{px.weights_dir}: SAVED")
        print(f"{px.savefile}: SAVED")
        px.validation()
    else:
        print(f"train_flag is {train_flag}: not saved")

    px.show(val_index=px.val_index)
    plt.show()

class Pixel():
    def __init__(self):
        self.val_index = 0
        self.epochs = 100
        self.batch_size = 256
        self.resolution = '1x1'
        self.tors = 'predictors_coarse_std_Apr_msot'
        self.tant = f"pr_{self.resolution}_std_MJJASO_world"
        self.seed = 1
        self.vsample = 1000
        self.lat, self.lon= 24, 72
        self.var_num = 4
        self.lat_grid, self.lon_grid = 24*5, 72*5 #  43200 -> 3600/24h then 12 days
        self.grid_num = self.lat_grid * self.lon_grid
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.metrics = tf.keras.metrics.MeanSquaredError()
        self.savefile = f"/docker/mnt/d/research/D2/cnn3/train_val/continuous/{self.tors}-{self.tant}.pickle"
        self.weights_dir = f"/docker/mnt/d/research/D2/cnn3/weights/continuous/{self.tors}-{self.tant}"
        self.pred_dir = f"/docker/mnt/d/research/D2/cnn3/result/continuous/world/{self.resolution}"
        self.pred_path = self.pred_dir + f"/epoch{self.epochs}_batch{self.batch_size}_seed{self.seed}.npy"

    def training(self, x_train, y_train, x_val, y_val, train_dct, val_dct):
        x_train, x_val = mask(x_train), mask(x_val)
        x_train, x_val = x_train.transpose(0, 2, 3, 1), x_val.transpose(0, 2, 3, 1)
        y_train, y_val = y_train.reshape(len(y_train), self.grid_num), y_val.reshape(len(y_val), self.grid_num)
        os.makedirs(self.weights_dir, exist_ok=True)
        for i in range(7062, self.grid_num):
            y_train_px = y_train[:, i]
            model = build_model((self.lat, self.lon, self.var_num))
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                          loss=self.loss,
                          metrics=[self.metrics])
            model.fit(x_train, y_train_px, batch_size=self.batch_size, epochs=self.epochs)
            weights_path = f"{self.weights_dir}/epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
            model.save_weights(weights_path)
        dct = {'x_train': x_train, 'y_train': y_train,
               'x_val': x_val, 'y_val': y_val,
               'train_dct': train_dct, 'val_dct': val_dct}
        with open(self.savefile, 'wb') as f:
            pickle.dump(dct, f)

    def validation(self):
        with open(self.savefile, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']

        pred_lst = []
        rmse = []
        corr = []
        rr = []
        for i in range(self.grid_num):
            y_val_px = y_val[:, i]
            model = build_model((self.lat, self.lon, self.var_num))
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                          loss=self.loss,
                          metrics=[self.metrics])
            weights_path = f"{self.weights_dir}/epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
            model.load_weights(weights_path)

            pred = model.predict(x_val) # (400, 1000)
            pred_lst.append(pred)

            result = model.evaluate(x_val, y_val_px)
            rmse.append(round(result[1], 2))

            pred = model.predict(x_val)
            corr_i = np.corrcoef(pred[:,0], y_val_px)
            corr.append(np.round(corr_i[0,1], 2))

            rr_i = corr_i**2
            rr.append(np.round(rr_i, 2))

            print(f"Correlation Coefficient of pixel{i}: {np.round(corr_i[0,1], 2)}")

        pred_arr = np.array(pred_lst)
        np.save(self.pred_path, pred_arr)

        corr = np.array(corr)
        corr = corr.reshape(self.lat_grid, self.lon_grid)
        acc_map(corr)

    def show(self, val_index=0):
        with open(self.savefile, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']
        y_val_px = y_val[val_index].reshape(self.lat_grid, self.lon_grid)
        show_map(y_val_px)

        pred_lst = []
        if os.path.exists(self.pred_path) is True:
            pred_val = np.squeeze(np.load(self.pred_path))
            pred_arr = pred_val[:, val_index]
        else:
            for i in range(self.grid_num):
                model = build_model((self.lat, self.lon, self.var_num))
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                              loss=self.loss,
                              metrics=[self.metrics])
                weights_path = f"{self.weights_dir}/epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
                model.load_weights(weights_path)
                pred = model.predict(x_val)
                result = pred[val_index]
                pred_lst.append(result)
            pred_arr = np.array(pred_lst)
        pred_arr = pred_arr.reshape(self.lat_grid, self.lon_grid)
        show_map(pred_arr)

if __name__ == '__main__':
    main()
