import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from util import open_pickle
from model3 import init_model
from view import pred_accuracy

def main():
    TEST = test()
    x_val, y_val, pred = TEST.load_pred()

class test():
    def __init__(self):
        self.val_index = 0
        self.class_num = 30
        self.discrete_mode = 'EFD'
        self.epochs =250
        self.batch_size =256
        self.seed = 1
        self.var_num = 4
        self.vsample = 1000
        self.resolution = '1x1'
        # path
        self.tors = 'predictors_coarse_std_Apr_msot'
        self.tant = f"pr_{self.resolution}_std_MJJASO_thailand_{self.discrete_mode}_{self.class_num}"
        self.workdir = '/docker/mnt/d/research/D2/cnn3'
        self.val_path = self.workdir + f"/train_val/class/{self.tors}-{self.tant}.pickle"
        self.weights_dir = self.workdir + f"/weights/class/{self.tors}-{self.tant}"
        self.pred_dir = self.workdir + f"/result/class/thailand/{self.resolution}"
        self.pred_path = self.pred_dir + f"/class{self.class_num}_epoch{self.epochs}_batch{self.batch_size}_seed{self.seed}.npy"
        # model
        self.lat, self.lon = 24, 72
        self.lr = 0.001
        self.lat_grid, self.lon_grid = 20, 20
        self.grid_num = self.lat_grid*self.lon_grid
        # init_model is allowd to be called once otherwise layer_name will be messed up
        self.model = init_model(lat=self.lat, lon=self.lon, var_num=self.var_num, lr=self.lr)

        # validation
        self.diff_bar_view_flag = True
        self.true_false_view_flag = False

    def load_pred(self):
        x_val, y_val = open_pickle(self.val_path)
        if os.path.exists(self.pred_path):
            pred_arr = np.squeeze(np.load(self.pred_path))
        else:
            pred_lst = []
            for i in range(self.grid_num):
                weights_path = self.weights_dir + f"/class{self.class_num}_epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
                model = self.model
                model.load_weights(weights_path)
                pred = model.predict(x_val)
                pred_lst.append(pred)
            pred_arr = np.squeeze(np.array(pred_lst))
            np.save(self.pred_path, pred_arr)
        return x_val, y_val, pred_arr # pred(400, 1000)
