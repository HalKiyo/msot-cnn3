import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from util import open_pickle
from model3 import init_model
from view import diff_bar, draw_val

def main():
    EVAL = evaluate()
    x_val, y_val, pred = EVAL.load_pred()
    if EVAL.diff_bar_view_flag is True:
        EVAL.diff_evaluation(pred, y_val)
    if EVAL.true_false_view_flag is True:
        EVAL.true_false_bar(pred, y_val)
    if EVAL.auc_view_flag is True:
        roc = EVAL.auc(pred.T, y_val)
        print(roc)

class evaluate():
    def __init__(self):
        self.val_index = 20
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
        self.val_path = self.workdir + f"/train_val/continuous/{self.tors}-{self.tant}.pickle"
        self.weights_dir = self.workdir + f"/weights/continuous/{self.tors}-{self.tant}"
        self.pred_dir = self.workdir + f"/result/continuous/thailand/{self.resolution}"
        self.pred_path = self.pred_dir + f"/epoch{self.epochs}_batch{self.batch_size}_seed{self.seed}.npy"
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
        self.auc_view_flag = True

    def load_pred(self):
        x_val, y_val = open_pickle(self.val_path)
        if os.path.exists(self.pred_path):
            pred_arr = np.squeeze(np.load(self.pred_path))
        else:
            pred_lst = []
            for i in range(self.grid_num):
                weights_path = f"{self.weights_dir}/epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
                model = self.model
                model.load_weights(weights_path)
                pred = model.predict(x_val)
                pred_lst.append(pred)
            pred_arr = np.squeeze(np.array(pred_lst))
            np.save(self.pred_path, pred_arr)
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

    def roc(self, sim, obs, percentile=20):
        """
        this roc function just returns single event
        if multiple events are needed to be evaluated,
        call auc function below
        """
        # percentile should be absolute number 
        sim = np.abs(sim)
        obs = np.abs(obs)

        # make criteria
        sim_per = np.percentile(sim, percentile)
        obs_per = np.percentile(obs, percentile)

        # calculate number of obs percentile
        over_per = sum(obs > obs_per)
        under_per = sum(obs <= obs_per)

        # save count of hit and false pixcel
        hit_count = 0
        false_count = 0
        for p in range(len(obs)):
                if sim[p] > sim_per and obs[p] > obs_per:
                    hit_count += 1
                elif sim[p] > sim_per and obs[p] <= obs_per:
                    false_count += 1

        # calculate HitRate and FalseAlertRate
        hr = hit_count/over_per
        far = false_count/under_per

        return hr, far

    def auc(self, sim, obs):
        result = [[0,0]]
        # percentile variation list
        per_list = np.arange(10, 100, 10)
        per_list = per_list[::-1]

        # calculate different percentile result
        for i in per_list:
            # calculate multiple varidation events
            hr_all, far_all = [], []
            for j in range(len(obs)):
                hr_n, far_n = self.roc(sim[j], obs[j], percentile=i)
                hr_all.append(hr_n)
                far_all.append(far_n)
            hr, far = np.mean(hr_all), np.mean(far_all)
            result.append([hr, far])

        result.append([1,1])
        return result

if __name__ == '__main__':
    main()
