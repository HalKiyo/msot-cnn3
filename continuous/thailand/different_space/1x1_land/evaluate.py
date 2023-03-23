import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from util import open_pickle
from model import init_model
from view import diff_bar, draw_val

def main():
    EVAL = evaluate()
    x_val, y_val, pred = EVAL.load_pred()
    if EVAL.diff_bar_view_flag is True:
        EVAL.diff_evaluation(pred, y_val)
    if EVAL.true_false_view_flag is True:
        EVAL.true_false_bar(pred, y_val)

class evaluate():
    def __init__(self):
        self.val_index = 20
        self.epochs =100
        self.batch_size =256
        self.seed =1
        self.var_num = 4
        self.vsample = 1000
        self.resolution = '1x1'
        # model
        self.lat, self.lon = 12, 48
        self.lr = 0.0001
        self.lat_grid, self.lon_grid = 20, 20
        self.grid_num = self.lat_grid*self.lon_grid
        # init_model is allowd to be called once otherwise layer_name will be messed up
        self.model = init_model(lat=self.lat, lon=self.lon, var_num=self.var_num, lr=self.lr)
        # path
        self.tant = f"pr_{self.resolution}_std_MJJASO_thailand"
        self.workdir = '/docker/mnt/d/research/D2/cnn3'
        self.identifier = f"{self.tant}_{self.lat}x{self.lon}"
        self.train_val_path = f"/docker/mnt/d/research/D2/cnn3/train_val/continuous/diff_space/1x1_land/{self.identifier}.pickle"
        self.weights_dir = f"/docker/mnt/d/research/D2/cnn3/weights/continuous/diff_space/1x1_land/{self.identifier}"
        self.result_dir = f"/docker/mnt/d/research/D2/cnn3/result/continuous/thailand/{self.resolution}/diff_space/1x1_land/{self.identifier}"
        self.result_path = self.result_dir + f"/epoch{self.epochs}_batch{self.batch_size}_seed{self.seed}.npy"

        # validation
        self.diff_bar_view_flag = False
        self.true_false_view_flag = True

    def load_pred(self):
        x_val, y_val = open_pickle(self.train_val_path)
        if os.path.exists(self.result_path):
            pred_arr = np.squeeze(np.load(self.result_path))
        else:
            pred_lst = []
            for i in range(self.grid_num):
                weights_path = f"{self.weights_dir}/epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
                model = self.model
                model.load_weights(weights_path)
                pred = model.predict(x_val)
                pred_lst.append(pred)
            pred_arr = np.squeeze(np.array(pred_lst))
            np.save(self.result_path, pred_arr)
        return x_val, y_val, pred_arr # pred(400, 1000)

    def diff_evaluation(self, pred, y):
        value = pred[:, self.val_index] # pred(400, 1000)
        label = y[self.val_index, :] # y(1000, 400)
        diff = np.abs(value - label)
        diff_flat = diff.reshape(-1)
        diff_mean = np.mean(diff_flat)
        print(diff_mean)
        diff_bar(diff_flat)

    def true_false_bar(self, pred, y, criteria=0.1):
        true_count, false_count = 0, 0
        for i in range(len(y)):
            diff = np.abs(pred[:, i] - y[i, :])
            diff_mean = np.mean(diff)
            if diff_mean <= criteria:
                true_count += 1
            else:
                false_count += 1
        draw_val(true_count, false_count)

if __name__ == '__main__':
    main()
