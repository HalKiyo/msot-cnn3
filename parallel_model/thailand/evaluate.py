import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt

from util import open_pickle
from class_model import init_class_model
from continuous_model import init_continuous_model
from view import scatter_and_marginal_density, cluster_scatter

def main():
    EVAL = evaluate()
    x_val, y_val_class, pred_class = EVAL.load_class() # pred:(400, 1000, 5), xy_val:(1000, 400)
    print(f"gridmean of prob_distribution of val_index " \
          f"{EVAL.val_index}: {np.mean( [ max(pred_class[i, EVAL.val_index]) for i in range(400) ] )}")

    x_val, y_val_continuous, pred_continuous = EVAL.load_continuous() # pred:(400, 1000), y_val:(1000, 400)

    EVAL.nrmse_vs_reliability(pred_class, 
                              pred_continuous,
                              y_val_class,
                              y_val_continuous,
                              continuous_threshold=0.2,
                              class_threshold=133)
    plt.show()


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
    def nrmse_vs_reliability(self,
                             pred_class,
                             pred_continuous,
                             y_val_class,
                             y_val_continuous,
                             class_threshold=133,
                             continuous_threshold=0.2):
        """
        pred_class: (400, 1000, 5)
        pred_continuous: (400, 1000)
        y_val_class: (1000, 400)
        y_val_continuous: (1000, 400)
        """
        # reliability and nrmse calculation
        accuracy_lst = []
        nrmse_lst = []
        reliability_lst = []

        true_reliability_lst =  []
        false_reliability_lst =  []
        else_reliability_lst =  []

        true_nrmse_lst = []
        false_nrmse_lst = []
        else_nrmse_lst = []

        true_accuracy_lst = []
        false_accuracy_lst = []
        else_accuracy_lst = []

        for sample in range(self.vsample):
            # accuracy
            grids_reliability = []
            grid_true, grid_false = 0, 0
            class_one_hot = pred_class[:, sample, :]
            class_label = y_val_class[sample, :]
            for g in range(self.grid_num):
                pred_label = np.argmax(class_one_hot[g, :])
                reliability = np.max(class_one_hot[g, :])
                grids_reliability.append(reliability)
                if int(pred_label) == class_label[g]:
                    grid_true += 1
                else:
                    grid_false += 1
            #accuracy = (grid_true)/(grid_true + grid_false)
            accuracy = grid_true
            accuracy_lst.append(accuracy)

            # nrmse
            continuous = pred_continuous[:, sample]
            continuous_label = y_val_continuous[sample, :]
            gridmean_nrmse = np.sqrt(np.mean((continuous - continuous_label)**2))
            nrmse_lst.append(gridmean_nrmse)

            # reliability
            gridmean_reliability = np.mean(grids_reliability)
            reliability_lst.append(gridmean_reliability)

            # classification
            if int(accuracy) >= class_threshold and gridmean_nrmse <= continuous_threshold:
                true_accuracy_lst.append(accuracy)
                true_nrmse_lst.append(gridmean_nrmse)
                true_reliability_lst.append(gridmean_reliability)
            elif int(accuracy) < class_threshold and gridmean_nrmse > continuous_threshold:
                false_accuracy_lst.append(accuracy)
                false_nrmse_lst.append(gridmean_nrmse)
                false_reliability_lst.append(gridmean_reliability)
            else:
                else_accuracy_lst.append(accuracy)
                else_nrmse_lst.append(gridmean_nrmse)
                else_reliability_lst.append(gridmean_reliability)

        # plot
#        scatter_and_marginal_density(accuracy_lst,
#                                     true_accuracy_lst,
#                                     true_nrmse_lst,
#                                     true_reliability_lst,
#                                     false_accuracy_lst,
#                                     false_nrmse_lst,
#                                     false_reliability_lst,
#                                     else_accuracy_lst,
#                                     else_nrmse_lst,
#                                     else_reliability_lst,
#                                     )
        cluster_scatter(accuracy_lst,
                        nrmse_lst,
                        reliability_lst)

#############################################################################

if __name__ == '__main__':
    main()
