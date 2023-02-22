import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from util import open_pickle
from model3 import init_model
from view import pred_accuracy, box_crossentropy

def main():
    EVAL = evaluate()
    x_val, y_val, pred = EVAL.load_pred() # (400, 1000, 5), (1000, 400)
    if EVAL.diff_bar_view_flag is True:
        EVAL.diff_evaluation(pred, y_val)
    if EVAL.true_false_view_flag is True:
        EVAL.true_false_bar(pred, y_val)
    EVAL.crossentropy(pred, y_val)

class evaluate():
    def __init__(self):
        self.val_index = 0
        self.class_num = 5
        self.discrete_mode = 'EFD'
        self.epochs = 150
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
        self.diff_bar_view_flag = False
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

    def diff_evaluation(self, pred, y):
        pred_onehot = pred[:, self.val_index] # pred(400, 1000, 5)
        label = y[self.val_index, :] # y(1000, 400)

        px_true, px_false = 0, 0
        for i in range(len(pred)):
            pred_label = np.argmax(pred_onehot[i])
            if int(pred_label) == label[i]:
                px_true += 1
            else:
                px_false += 1
        pred_accuracy(px_true, px_false)

    def true_false_bar(self, pred, y, criteria=200):
        true_count, false_count = 0, 0
        for i in range(len(y)): # val_num
            px_true = 0
            for j in range(len(pred)): # px_num
                pred_label = np.argmax(pred[j, i])
                if int(pred_label) == y[i, j]:
                    px_true += 1
            print(px_true)
            if px_true <= criteria:
                false_count += 1
            else:
                true_count += 1
        pred_accuracy(true_count, false_count)

    def crossentropy(self, val_pred, val_label, criteria=200):
        true = {f"true, false": []}
        false = {f"true, false": []}

        for i in range(len(val_label)): # val_num
            px_true = 0
            cross = []
            for j in range(len(val_pred)): # px_num
                max_cross = np.max(val_pred[j, i])
                cross.append(max_cross)
                pred_label = np.argmax(val_pred[j, i])
                if int(pred_label) == val_label[i, j]:
                    px_true += 1

            cross_mean = np.mean(cross)
            if px_true <= criteria:
                false[f"true, false"].append(cross_mean)
            else:
                true[f"true, false"].append(cross_mean)

        box_crossentropy(true, false)

if __name__ == '__main__':
    main()
