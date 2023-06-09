import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt

from util import open_pickle
from class_model import init_class_model
from continuous_model import init_continuous_model
#from view import sand

def main():
    EVAL = evaluate()
    x_val, y_val_class, pred_class = EVAL.load_class() # pred:(400, 1000, 5), xy_val:(1000, 400)
    print(f"gridmean of prob_distribution of val_index " \
          f"{EVAL.val_index}: {np.mean( [ max(pred_class[i, EVAL.val_index]) for i in range(400) ] )}")

    x_val, y_val_continuous, pred_continuous = EVAL.load_continuous() # pred:(400, 1000), y_val:(1000, 400)

    """
    EVAL.accuracy_bar_singlesample(pred, y_val)
    EVAL.samplewise_accuracy_bar(pred, y_val, criteria=300)
    EVAL.max_probability(pred, y_val, criteria=300)
    EVAL.probability_distribution(pred, y_val, pixel_index=100)
    plt.show()
    """

class evaluate():
    def __init__(self):
        ############################# EDIT HERE ###########################
        ############################# common setting ######################
        self.px_index = 150
        self.val_index = 20 #true_index=330, false_index=20
        self.resolution = '1x1'
        self.var_num = 4
        self.tors = 'predictors_coarse_std_Apr_msot'

        self.seed = 1
        self.batch_size =256
        self.vsample = 1000
        self.lat, self.lon = 24, 72
        self.lr = 0.0001
        self.lat_grid, self.lon_grid = 20, 20
        self.grid_num = self.lat_grid*self.lon_grid
        self.dir = f"/docker/mnt/d/research/D2/cnn3"
        ###################################################################
        ########################### class model setting ###################
        self.class_num = 5
        self.discrete_mode = 'EFD'
        self.class_epochs = 150
        self.class_tand = f"pr_{self.resolution}_std_MJJASO_thailand_{self.discrete_mode}_{self.class_num}"

        self.class_train_val_path = self.dir + f"/train_val/class/{self.tors}-{self.class_tand}.pickle"
        self.class_weights_dir = self.dir + f"/weights/class/{self.tors}-{self.class_tand}"
        self.class_result_dir = self.dir + f"/result/class/thailand/{self.resolution}/{self.tors}-{self.class_tand}"
        self.class_result_path = self.class_result_dir + f"/class{self.class_num}_epoch{self.class_epochs}_batch{self.batch_size}_seed{self.seed}.npy"
        # init_model is allowd to be called once otherwise layer_name will be messed up
        self.class_model = init_class_model(lat=self.lat, lon=self.lon, var_num=self.var_num, lr=self.lr)
        ##################################################################
        ########################### continuous model setting #############
        self.continuous_epochs = 100
        self.continuous_tand = f"pr_{self.resolution}_std_MJJASO_thailand"

        self.continuous_train_val_path = self.dir + f"/train_val/continuous/{self.tors}-{self.continuous_tand}.pickle"
        self.continuous_weights_dir = self.dir + f"/weights/continuous/{self.tors}-{self.continuous_tand}"
        self.continuous_result_dir = self.dir + f"/result/continuous/thailand/{self.resolution}/{self.tors}-{self.continuous_tand}"
        self.continuous_result_path = self.continuous_result_dir + f"/epoch{self.continuous_epochs}_batch{self.batch_size}_seed{self.seed}.npy"
        # init_model is allowd to be called once otherwise layer_name will be messed up
        self.continuous_model = init_continuous_model(lat=self.lat, lon=self.lon, var_num=self.var_num, lr=self.lr)
        ##################################################################

    def load_class(self):
        x_val, y_val_class = open_pickle(self.class_train_val_path)
        if os.path.exists(self.class_result_path):
            pred_arr = np.squeeze(np.load(self.class_result_path))
        else:
            pred_lst = []
            for i in range(self.grid_num):
                class_weights_path = self.class_weights_dir + f"/class{self.class_num}_epoch{self.class_epochs}_batch{self.batch_size}_{i}.h5"
                model = self.class_model
                model.load_weights(class_weights_path)
                pred_class = model.predict(x_val)
                pred_lst.append(pred_class)
            pred_arr = np.squeeze(np.array(pred_lst))
            np.save(self.class_result_path, pred_arr)
            print(f"{self.class_result_path} is SAVED")
        return x_val, y_val_class, pred_arr # x_val(1000, 400, 4) y_val(1000, 400) pred(400, 1000, 5)

    def load_continuous(self):
        x_val, y_val_continuous = open_pickle(self.continuous_train_val_path)
        if os.path.exists(self.continuous_result_path):
            pred_arr = np.squeeze(np.load(self.continuous_result_path))
        else:
            pred_lst = []
            for i in range(self.grid_num):
                continuous_weights_path = self.continuous_weights_dir + f"/epoch{self.continuous_epochs}_batch{self.batch_size}_{i}.h5"
                model = self.class_model
                model.load_weights(continuous_weights_path)
                pred_continuous = model.predict(x_val)
                pred_lst.append(pred_continuous)
            pred_arr = np.squeeze(np.array(pred_lst))
            np.save(self.continuous_result_path, pred_arr)
            print(f"{self.continuous_result_path} is SAVED")
        return x_val, y_val_continuous, pred_arr # x_val(1000, 400, 4) y_val(1000, 400) pred(400, 1000)

################################ pred loaded ################################
#############################################################################
    def reliability_vs_nrmse(self, pred_class, pred_continuous, y_val_continuous):
        """
        pred_class: (400, 1000, 5)
        pred_continuous: (400, 1000)
        y_val_continuous: (1000, 400)
        """


#############################################################################
    def accuracy_bar_singlesample(self, pred, y):
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

    def samplewise_accuracy_bar(self, pred, y, criteria=300):
        true_count, false_count = 0, 0
        true_list = []
        for i in range(len(y)): # val_num
            px_true = 0
            for j in range(len(pred)): # px_num
                pred_label = np.argmax(pred[j, i])
                if int(pred_label) == y[i, j]:
                    px_true += 1
            true_list.append(px_true)

            if px_true <= criteria:
                false_count += 1
            else:
                true_count += 1

        # draw histgram of hitrate within a validation sample
        true_array = np.array(true_list)
        bimodal_dist(true_array)

        pred_accuracy(true_count, false_count)

    def probability_distribution(self, val_pred, val_label, pixel_index=150):
        """
        val_pred = (400, 1000, 5)
        """
        pred = val_pred[pixel_index, self.val_index]
        pred_label = np.argmax(pred)
        if int(pred_label) == val_label[self.val_index, pixel_index]:
            flag = True
        else:
            flag = False
        view_probability(pred, flag)

    def max_probability(self, val_pred, val_label, criteria=200):
        true = {f"true, false": []}
        false = {f"true, false": []}

        for i in range(len(val_label)): # val_num
            px_true = 0
            cross = []
            for j in range(len(val_pred)): # px_num
                ############# max_corss = 信頼度 ###################
                max_cross = np.max(val_pred[j, i])
                ####################################################
                cross.append(max_cross)
                pred_label = np.argmax(val_pred[j, i])
                if int(pred_label) == val_label[i, j]:
                    px_true += 1

            # cross_mean is mean of max_cross in 'i'th validation sample
            cross_mean = np.mean(cross)

            # count true events in 'i'th validation sample
            if px_true <= criteria:
                false[f"true, false"].append(cross_mean)
            else:
                true[f"true, false"].append(cross_mean)

        # draw percentiles
        t25, t50, t75 = np.percentile(true[f"true, false"], [25, 50, 75])
        f25, f50, f75 = np.percentile(false[f"true, false"], [25, 50, 75])
        print(f"true{t25}, {t50}, {t75}")
        print(f"false{f25}, {f50}, {f75}")

        box_crossentropy(true, false)

if __name__ == '__main__':
    main()
