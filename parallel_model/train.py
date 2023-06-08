import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util import load, shuffle, mask
from class_model import build_class_model
from continuous_model import build_continuous_model
from view_tmp import true_false_bar, show_class, accuracy_map

def main():
    ################### EDIT HERE #####################
    train_flag = False
    ###################################################

    px = Pixel()
    if train_flag is True:
        predictors, predictand_class = load(px.tors, 
                                            px.class_tand, 
                                            model_type='class')
        px.train_class(*shuffle(predictors, 
                                predictand_class, 
                                px.vsample, 
                                px.seed, 
                                px.lat_grid, 
                                px.lon_grid))
    else:
        print(f"train_flag is {train_flag}: not saved")

    px.validation_class()
    px.show(val_index=px.val_index)
    px.labelwise_accuracy_gridmean()
    plt.show()


class Pixel():
    def __init__(self):
        ########################### EDIT HERE ##################################
        ########################### common setting #############################
        self.val_index = 20 #true_index=330, false_index=20
        self.resolution = '1x1' # 1x1 or 5x5_coarse
        self.var_num = 4
        self.tors = 'predictors_coarse_std_Apr_msot'
        self.seed = 1
        self.batch_size = 256
        self.vsample = 1000
        self.lat, self.lon = 24, 72
        self.lat_grid, self.lon_grid = 20, 20
        self.grid_num = self.lat_grid*self.lon_grid 
        ########################################################################
        ########################### class model setting ########################
        self.class_num = 5
        self.descrete_mode = 'EFD'
        self.class_epochs = 150
        self.class_loss = tf.keras.losses.CategoricalCrossentropy()
        self.class_metrics = tf.keras.metrics.CategoricalAccuracy()

        self.class_tand = f"pr_{self.resolution}_std_MJJASO_thailand_{self.descrete_mode}_{self.class_num}"
        self.dir = f"/docker/mnt/d/research/D2/cnn3"
        self.class_savefile = self.dir + f"/train_val/class/{self.tors}-{self.class_tand}.pickle"
        self.class_weights_dir = self.dir + f"/weights/class/{self.tors}-{self.class_tand}"
        self.class_result_dir = self.dir + f"/result/class/thailand/{self.resolution}/{self.tors}-{self.class_tand}"
        self.class_result_path = self.class_result_dir + f"/class{self.class_num}_epoch{self.class_epochs}_batch{self.batch_size}_seed{self.seed}.npy"
        ########################################################################
        ########################### continuous model setting ###################
        ########################################################################

    def train_class(self, x_train, y_train, x_val, y_val, train_dct, val_dct):
        x_train = mask(x_train)
        x_train = x_train.transpose(0, 2, 3, 1)
        x_val   = mask(x_val)
        x_val   = x_val.transpose(0, 2, 3, 1)
        y_train = y_train.reshape(len(y_train), self.grid_num) 
        y_val   = y_val.reshape(len(y_val), self.grid_num)

        os.makedirs(self.class_weights_dir, exist_ok=True) # create weight directory
        for i in range(self.grid_num):
            y_train_px = y_train[:, i]
            y_train_one_hot = tf.keras.utils.to_categorical(y_train_px,
                                                            self.class_num)
            model = build_class_model((self.lat, self.lon, self.var_num), 
                                      self.class_num)
            model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                          loss=self.class_loss,
                          metrics=[self.class_metrics])
            his = model.fit(x_train,
                            y_train_one_hot,
                            batch_size=self.batch_size,
                            epochs=self.class_epochs)
            class_weights_path = f"{self.class_weights_dir}/class{self.class_num}_epoch{self.class_epochs}_batch{self.batch_size}_{i}.h5"
            model.save_weights(class_weights_path)
        print(f"{self.class_weights_dir}: SAVED")

        dct = {'x_train': x_train,
               'y_train': y_train,
               'x_val': x_val, 
               'y_val': y_val,
               'train_dct': train_dct,
               'val_dct': val_dct}
        with open(self.class_savefile, 'wb') as f:
            pickle.dump(dct, f)
        print(f"{self.class_savefile}: SAVED")


######################## TRAINING DONE ################################################
#######################################################################################


    def validation_class(self):
        """
        averaged gridwise accuracy map 20x20 
        (ratio of true or false in grids)
        """
        with open(self.class_savefile, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']

        # check if result_path exists
        if os.path.exists(self.class_result_path) is True:
            pred_arr = np.load(self.class_result_path)
            y_val_lst = []
            for i in range(self.grid_num):
                y_val_px = y_val[:, i]
                y_val_one_hot = tf.keras.utils.to_categorical(y_val_px, 
                                                              self.class_num)
                y_val_lst.append(y_val_one_hot)

            pred_arr = pred_arr.reshape(self.grid_num,
                                        self.vsample,
                                        self.class_num)
            y_val_arr = np.array(y_val_lst).reshape(self.grid_num,
                                                    self.vsample,
                                                    self.class_num)

            acc_lst = []
            for g in range(self.grid_num):
                pred_px = pred_arr[g, :, :]
                y_px = y_val_arr[g, :, :]
                val_true = 0
                val_false = 0
                for i, j in zip(pred_px, y_px):
                    if np.argmax(i) == np.argmax(j):
                        val_true += 1
                    else:
                        val_false += 1
                rate = (val_true)/(val_true + val_false)
                acc_lst.append(rate)
            acc = np.array(acc_lst).reshape(self.lat_grid, self.lon_grid)

        else:
            pred_lst = []
            acc = []
            for i in range(self.grid_num):
                # load model
                model = build_class_model((self.lat, self.lon, self.var_num), 
                                          self.class_num)
                model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                              loss=self.class_loss, 
                              metrics=[self.class_metrics])
                class_weights_path = f"{self.class_weights_dir}/class{self.class_num}_epoch{self.class_epochs}_batch{self.batch_size}_{i}.h5"
                model.load_weights(class_weights_path)

                # prediction
                pred = model.predict(x_val) # (400, 1000, 5)
                pred_lst.append(pred)

                # real label
                y_val_px = y_val[:, i]
                y_val_one_hot = tf.keras.utils.to_categorical(y_val_px, 
                                                              self.class_num)
                # evaluation
                result = model.evaluate(x_val, y_val_one_hot)
                acc.append(round(result[1], 2))
                print(f"CategoricalAccuracy of pixcel{i}: {result[1]}")
                acc = np.array(acc)
                acc = acc.reshape(self.lat_grid, self.lon_grid)

                # save result
                pred_arr = np.array(pred_lst)
                if os.path.exists(self.class_result_dir) is False:
                    os.makedirs(self.class_result_dir, exist_ok=True)
                np.save(self.class_result_path, pred_arr)
                print(f"{self.class_result_path} is saved")

        # view accuracy
        accuracy_map(acc)

    def show(self, val_index=0):
        """
        val_index = validation sample index
        showing true map 20x20
        showing predicted map 20x20
        """
        # load train and validation data
        with open(self.class_savefile, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']
        y_val_px = y_val[val_index].reshape(self.lat_grid, self.lon_grid)
        # show true label map
        show_class(y_val_px, class_num=self.class_num)

        # load or create prediction
        pred_lst = []
        if os.path.exists(self.class_result_path) is True:
            pred_arr = np.load(self.class_result_path)
            for i in range(self.grid_num):
                label = np.argmax(pred_arr[i, val_index])
                pred_lst.append(label)
        else:
            for i in range(self.grid_num):
                model = build_class_model((self.lat, self.lon, self.var_num),
                                          self.class_num)
                model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                              loss=self.class_loss, 
                              metrics=[self.class_metrics])
                weights_path = f"{self.class_weights_dir}/class{self.class_num}_epoch{self.class_epochs}_batch{self.batch_size}_{i}.h5"
                model.load_weights(weights_path)
                pred = model.predict(x_val)
                label = np.argmax(pred[val_index])
                pred_lst.append(label)
                print(f"pixcel{i}: {label}")

        pred_label = np.array(pred_lst)
        pred_label = pred_label.reshape(self.lat_grid, self.lon_grid)
        # show predicted label map
        show_class(pred_label, class_num=self.class_num)

    def labelwise_accuracy_singlegrid(self, px_index):
        with open(self.savefile, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']
        y_val_px = y_val[:, px_index]
        y_val_one_hot = tf.keras.utils.to_categorical(y_val_px, self.class_num)
        model = build_class_model((self.lat, self.lon, self.var_num), self.class_num)
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                      loss=self.loss, 
                      metrics=[self.metrics])
        weights_path = f"{self.weights_dir}/class{self.class_num}_epoch{self.epochs}_batch{self.batch_size}_{px_index}.h5"
        model.load_weights(weights_path)
        pred = model.predict(x_val)
        
        # draw_validation
        class_label, counts = draw_val(pred, y_val_one_hot, class_num=self.class_num)
        print(f"class_label: {class_label}\n" \
              f"counts: {counts}")

    def labelwise_accuracy_gridmean(self):
        with open(self.class_savefile, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']

        y_val_lst = []
        if os.path.exists(self.class_result_path) is True:
            pred_arr = np.load(self.class_result_path)
            for i in range(self.grid_num):
                y_val_px = y_val[:, i]
                y_val_one_hot = tf.keras.utils.to_categorical(y_val_px, self.class_num)
                y_val_lst.append(y_val_one_hot)
        else:
            pred_lst = []
            for i in range(self.grid_num):
                y_val_px = y_val[:, i]
                y_val_one_hot = tf.keras.utils.to_categorical(y_val_px, 
                                                              self.class_num)
                model = build_class_model((self.lat, self.lon, self.var_num), 
                                          self.class_num)
                model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                              loss=self.class_loss, 
                              metrics=[self.class_metrics])
                class_weights_path = f"{self.class_weights_dir}/class{self.class_num}_epoch{self.class_epochs}_batch{self.batch_size}_{i}.h5"
                model.load_weights(class_weights_path)
                pred = model.predict(x_val)
                pred_lst.append(pred)
                y_val_lst.append(y_val_one_hot)
            pred_arr = np.array(pred_lst)

        pred_arr = pred_arr.reshape(self.grid_num*self.vsample,
                                    self.class_num)
        y_val_arr = np.array(y_val_lst).reshape(self.grid_num*self.vsample,
                                                self.class_num)
        class_label, counts = true_false_bar(pred_arr, 
                                             y_val_arr, 
                                             class_num=self.class_num)
        print(f"class_label: {class_label}\n" \
              f"counts: {counts}")


if __name__ == '__main__':
    main()

