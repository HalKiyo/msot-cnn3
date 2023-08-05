import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import build_model
from view import draw_val, show_class, view_accuracy
from util import load, shuffle, mask
from gradcam import grad_cam, show_heatmap, image_preprocess
from displaycallback import DisplayCallBack

def main():
    train_flag = False
    overwrite_flag = False

    px = Pixel()
    cdDisplay = DisplayCallBack()

    if train_flag is True:
        predictors, predictant = load(px.tors, px.tant)
        x_train, y_train, x_val, y_val, train_dct, val_dct = shuffle(predictors,
                                                                     predictant,
                                                                     px.vsample,
                                                                     px.seed,
                                                                     px.lat_grid,
                                                                     px.lon_grid,
                                                                     )
        px.training(x_train, y_train, x_val, y_val, train_dct, val_dct, cdDisplay)
        print(f"{px.weights_dir}: SAVED")
        print(f"{px.train_val_path}: SAVED")
    else:
        print(f"train_flag is {train_flag}: not saved")

    px.validation(overwrite=overwrite_flag)
    px._show(val_index=px.val_index)
    plt.show()

class Pixel():
    def __init__(self):
        ###################################################################################################
        # edit here for parameter
        ###################################################################################################
        self.val_index = 20 #true_index=330, false_index=20
        self.class_num = 5
        self.descrete_mode = 'EFD'
        self.epochs = 150
        self.batch_size = 256
        self.patience_num = 1000
        self.resolution = '5x5' # 1x1 or 5x5_coarse
        ##################################################################################################
        # edit here for experiment condition
        ##################################################################################################
        self.var_num = 4 
        self.tors = 'predictors_coarse_std_Apr_msot'
        self.tant = f"pr_{self.resolution}_coarse_std_MJJASO_monsoon_{self.descrete_mode}_{self.class_num}"
        #################################################################################################
        # do not change here
        #################################################################################################
        self.seed = 1
        self.vsample = 1000
        self.lat, self.lon = 24, 72
        self.lat_grid, self.lon_grid = 14, 18
        self.grid_num = self.lat_grid*self.lon_grid 
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.metrics = tf.keras.metrics.CategoricalAccuracy()
        self.train_val_path = f"/docker/mnt/d/research/D2/cnn3/train_val/class/{self.tors}-{self.tant}.pickle"
        self.weights_dir = f"/docker/mnt/d/research/D2/cnn3/weights/class/{self.tors}-{self.tant}"
        self.result_dir = f"/docker/mnt/d/research/D2/cnn3/result/class/monsoon/{self.resolution}/{self.tors}-{self.tant}"
        self.result_path = self.result_dir + f"/class{self.class_num}_epoch{self.epochs}_batch{self.batch_size}_seed{self.seed}.npy"

    ##########################################################################################################
    # training begin
    ##########################################################################################################
    def training(self, x_train, y_train, x_val, y_val, train_dct, val_dct, display):
        """
        (5930, 4, 24, 72) -> (5930, 24, 72, 4)
        """
        x_train, x_val = mask(x_train), mask(x_val)
        x_train, x_val = x_train.transpose(0, 2, 3, 1), x_val.transpose(0, 2, 3, 1)
        y_train, y_val = y_train.reshape(len(y_train), self.grid_num), y_val.reshape(len(y_val), self.grid_num)
        os.makedirs(self.weights_dir, exist_ok=True) # create weight directory

        # gridwise model training
        for i in range(self.grid_num):
            y_train_px = y_train[:, i]
            y_train_one_hot = tf.keras.utils.to_categorical(y_train_px, self.class_num)
            y_val_px = y_val[:, i]
            y_val_one_hot = tf.keras.utils.to_categorical(y_val_px, self.class_num)

            # model setting
            model = build_model((self.lat, self.lon, self.var_num), self.class_num)
            model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                          loss=self.loss,
                          metrics=[self.metrics])

            # early stop setting
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience_num)
            print(f"\ntraining_pixel: {i}")
            his = model.fit(x_train,
                            y_train_one_hot,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            validation_data=(x_val, y_val_one_hot),
                            verbose=0,
                            callbacks=[display, early_stop],
                            )

            # save weights path
            weights_path = f"{self.weights_dir}/class{self.class_num}_epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
            model.save_weights(weights_path)

        """
        # loss visualization
        loss = his.history['loss']
        val_loss = his.history['val_loss']
        self.loss_visualization(loss, val_loss)
        plt.show()
        """

        # save train_val pickle
        dct = {'x_train': x_train, 'y_train': y_train,
               'x_val': x_val, 'y_val': y_val,
               'train_dct': train_dct, 'val_dct': val_dct}
        with open(self.train_val_path, 'wb') as f:
            pickle.dump(dct, f)

    ##########################################################################################################
    # training done
    ##########################################################################################################

    def validation(self, overwrite=False):
        """
        pred.shape = (252, 1000, 5)
        y_val.shape = (1000, 252)
        """
        with open(self.train_val_path, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']

        if os.path.exists(self.result_path) is False or overwrite is True:
            pred_lst = []
            acc = []

            for i in range(self.grid_num):
                y_val_px = y_val[:, i]
                y_val_one_hot = tf.keras.utils.to_categorical(y_val_px, self.class_num)
                model = build_model((self.lat, self.lon, self.var_num), self.class_num)
                model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                              loss=self.loss, 
                              metrics=[self.metrics])
                weights_path = f"{self.weights_dir}/class{self.class_num}_epoch{self.epochs}_batch{self.batch_size}_patience{self.patience_num}_{i}.h5"
                model.load_weights(weights_path)

                pred = model.predict(x_val) # (400, 1000, 5)
                pred_lst.append(pred)

                result = model.evaluate(x_val, y_val_one_hot)
                acc.append(round(result[1], 2))

                print(f"CategoricalAccuracy of pixcel{i}: {result[1]}")

            pred_arr = np.array(pred_lst)
            os.makedirs(self.result_dir, exist_ok=True) # create weight directory
            np.save(self.result_path, pred_arr)

        else:
            acc = []
            load_pred = np.squeeze(np.load(self.result_path))

            for i in range(self.grid_num):
                val_true = []
                val_false = []

                pred_px = load_pred[i, :, :]
                y_val_px = y_val[:, i]

                for pred, tr in zip(pred_px, y_val_px):

                    if np.argmax(pred) == int(tr):
                        val_true.append(int(tr))

                    else:
                        val_false.append(int(tr))

                acc_i = len(val_true)/(len(val_true ) + len(val_false))
                acc_i_round = np.round(acc_i, 2)
                print(f"pixel{i}: {acc_i_round}")
                acc.append(acc_i_round)

        acc = np.array(acc)
        acc = acc.reshape(self.lat_grid, self.lon_grid)
        view_accuracy(acc, lat_grid=self.lat_grid, lon_grid=self.lon_grid, vmin=0.75, vmax=1.00)

    def _show(self, val_index):
        # show true data
        with open(self.train_val_path, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']
        y_val_px = y_val[val_index].reshape(self.lat_grid, self.lon_grid)
        show_class(y_val_px, class_num=self.class_num)

        # show predicted data
        pred_lst = []
        if os.path.exists(self.result_path) is True:
            pred_arr = np.load(self.result_path)
            for i in range(self.grid_num):
                label = np.argmax(pred_arr[i, val_index])
                pred_lst.append(label)
        else:
            for i in range(self.grid_num):
                model = build_model((self.lat, self.lon, self.var_num), self.class_num)
                model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                              loss=self.loss, 
                              metrics=[self.metrics])
                weights_path = f"{self.weights_dir}/class{self.class_num}_epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
                model.load_weights(weights_path)
                pred = model.predict(x_val)
                label = np.argmax(pred[val_index])
                pred_lst.append(label)
                print(f"pixcel{i}: {label}")
        pred_label = np.array(pred_lst)
        pred_label = pred_label.reshape(self.lat_grid, self.lon_grid)
        show_class(pred_label, class_num=self.class_num)

    #################################################################################################
    # default validation done
    ##################################################################################################

    def gradcam(self, px_index, gradcam_index, layer_name):
        with open(self.train_val_path, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']
        y_val_px = y_val[:, px_index]

        model = build_model((self.lat, self.lon, self.var_num), self.class_num)
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                      loss=self.loss, 
                      metrics=[self.metrics])
        weights_path = f"{self.weights_dir}/class{self.class_num}_epoch{self.epochs}_batch{self.batch_size}_{px_index}.h5"
        model.load_weights(weights_path)

        preprocessed_image = image_preprocess(x_val, gradcam_index)
        heatmap = grad_cam(model, preprocessed_image, y_val[gradcam_index], layer_name,
                           self.lat, self.lon, self.class_num)
        show_heatmap(heatmap)

    def label_dist(self, px_index):
        with open(self.train_val_path, 'rb') as f:
            data = pickle.load(f)

        x_val, y_val = data['x_val'], data['y_val']
        y_val_px = y_val[:, px_index]
        y_val_one_hot = tf.keras.utils.to_categorical(y_val_px, self.class_num)
        model = build_model((self.lat, self.lon, self.var_num), self.class_num)
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                      loss=self.loss, 
                      metrics=[self.metrics])
        weights_path = f"{self.weights_dir}/class{self.class_num}_epoch{self.epochs}_batch{self.batch_size}_{px_index}.h5"
        model.load_weights(weights_path)
        pred = model.predict(x_val)
        class_label, counts = draw_val(pred, y_val_one_hot, class_num=self.class_num)
        print(f"class_label: {class_label}\n" \
              f"counts: {counts}")

    def label_dist_multigrid(self):
        with open(self.train_val_path, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']

        y_val_lst = []
        if os.path.exists(self.result_path) is True:
            pred_arr = np.load(self.result_path)
            for i in range(self.grid_num):
                y_val_px = y_val[:, i]
                y_val_one_hot = tf.keras.utils.to_categorical(y_val_px, self.class_num)
                y_val_lst.append(y_val_one_hot)
        else:
            pred_lst = []
            for i in range(self.grid_num):
                y_val_px = y_val[:, i]
                y_val_one_hot = tf.keras.utils.to_categorical(y_val_px, self.class_num)
                model = build_model((self.lat, self.lon, self.var_num), self.class_num)
                model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                              loss=self.loss, 
                              metrics=[self.metrics])
                weights_path = f"{self.weights_dir}/class{self.class_num}_epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
                model.load_weights(weights_path)
                pred = model.predict(x_val)
                pred_lst.append(pred)
                y_val_lst.append(y_val_one_hot)
            pred_arr = np.array(pred_lst)

        pred_arr = pred_arr.reshape(self.grid_num*self.vsample, self.class_num)
        y_val_arr = np.array(y_val_lst).reshape(self.grid_num*self.vsample, self.class_num)
        class_label, counts = draw_val(pred_arr, y_val_arr, class_num=self.class_num)
        print(f"class_label: {class_label}\n" \
              f"counts: {counts}")


if __name__ == '__main__':
    main()

