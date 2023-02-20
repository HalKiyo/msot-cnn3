import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from util import open_pickle
from model3 import init_model
from view import draw_val

class evaluate():
    def __init__(self):
        self.epochs =100
        self.batch_size =256
        self.seed =1
        self.var_num = 4
        self.vsample = 1000
        self.resolution = '1x1'
        # path
        self.tors = 'predictors_coarse_std_Apr_msot'
        self.tant = f"pr_{self.resolution}_std_MJJASO_thailand"
        self.workdir = '/docker/mnt/d/research/D2/cnn3'
        self.val_path = self.workdir + f"/train_val/continuous/{self.tors}-{self.tant}.pickel"
        self.weights_dir = self.workdir + f"/weights/continuous/{self.tors}-{self.tant}"
        self.weights_path = self.workdir + f"/epoch{self.epochs}_batch{self.batch_size}_seed{self.seed}.h5"
        self.pred_dir = self.workdir + f"/result/continuous/thailand/{self.resolution}"
        self.pred_path = self.pred_dir + f"/epoch{self.epochs}_batch{self.batch_size}_seed{self.seed}.npy"
        # model
        self.lat, self.lon = 24, 72
        self.lr = 0.001
        self.lat_grid, self.lon_grid = 20, 20
        self.grid_num = self.lat_grid*self.lon_grid
        # init_model is allowd to be called once otherwise layer_name will be messed up
        self.model = init_model(self.weights_path, lat=self.lat, lon=self.lon, var_num=self.var_num, lr=self.lr)

        # validation
        self.true_false_bar_view_flag = False

    def load_pred(self):
        x_val, y_val = open_pickle(self.val_path)
        if os.path.exists(self.pred_path):
            pred_arr = np.squeeze(np.load(self.pred_path))
        else:
            pred_lst = []
            for i in range(self.grid_num):
                weights_path = f"{self.weights_dir}/epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
                model = init_model(weights_path, lat=self.lat, lon=self.lon, var_num=self.var_num, lr=self.lr)
                pred = model.predict(x_val)
                pred_lst.append(pred)
            pred_arr = np.squeeze(np.array(pred_lst))
            np.save(self.pred_path, pred_arr)
        return x_val, y_val, pred

    def true_false_bar(self, pred, y):
        true_lst, false_lst = mk_true_false_list(pred, y)
        print_acc(true_lst, false_lst)
        draw_val(true_lst, false_lst)
