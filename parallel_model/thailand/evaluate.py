import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt

from util import open_pickle, get_model_index
from class_model import init_class_model
from continuous_model import init_continuous_model
from view import scatter_and_marginal_density, ensemble_step, \
                 ensemble_violin, gcm_bars, concentration_bar, pdf_box, \
                 scatter_and_marginal_and_bimodal, \
                 show_accuracy_vs_reliability

def main():
    EVAL = evaluate()
    train_model, train_year, val_model, val_year = get_model_index(EVAL.class_train_val_path)

    # class load
    x_val, y_val_class, pred_class = EVAL.load_class() # pred:(400, 1000, 5), xy_val:(1000, 400)

    # continuous load
    x_val, y_val_continuous, pred_continuous = EVAL.load_continuous() # pred:(400, 1000), y_val:(1000, 400)

    # evaluation
    """
    EVAL.nrmse_vs_reliability(pred_class, 
                              pred_continuous,
                              y_val_class,
                              y_val_continuous,
                              continuous_threshold=0.3,
                              class_threshold=150)

    EVAL.predicted_probability_density(pred_class,
                                       y_val_class)

    EVAL.continuos_class_coherence(pred_class,
                                   pred_continuous,
                                   y_val_class,
                                   )

    EVAL.concentration_demo(pred_class,
                            y_val_class)

    EVAL.gcmwise_true_false(pred_class,
                            y_val_class,
                            val_model)

    EVAL.ensemble_pdf_boxplot(pred_class,
                              y_val_class)

    EVAL.nrmse_vs_accuracy(pred_class, 
                           pred_continuous,
                           y_val_class,
                           y_val_continuous,
                           continuous_threshold=0.3,
                           class_threshold=300)
    """

    EVAL.accuracy_vs_reliability(pred_class, 
                               y_val_class,
                               class_threshold=300)

    # plot
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
                             continuous_threshold=0.3):
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
        scatter_and_marginal_density(accuracy_lst,
                                     nrmse_lst,
                                     reliability_lst,
                                     true_accuracy_lst,
                                     true_nrmse_lst,
                                     true_reliability_lst,
                                     false_accuracy_lst,
                                     false_nrmse_lst,
                                     false_reliability_lst,
                                     else_accuracy_lst,
                                     else_nrmse_lst,
                                     else_reliability_lst,
                                     )

    def predicted_probability_density(self,
                                     pred_class,
                                     y_val_class):
        """
        pred_class: (400, 1000, 5)
        y_val_class: (1000, 400)
        label: 0 or 1 or 2 or 3 or 4
        result: True or False
        prob: 0 < probability < 1
        """
        true_density = {f"{i}": [] for i in range(self.class_num)}
        false_density = {f"{i}": [] for i in range(self.class_num)}
        dict_for_df = {f"{i}": {'result': [], 'label': [], 'prob': []} for i in range(self.class_num)}

        for sample in range(self.vsample):
            # true or false
            class_one_hot = pred_class[:, sample, :]
            class_label = y_val_class[sample, :]
            for g in range(self.grid_num):
                predicted_label = np.argmax(class_one_hot[g, :])
                predicted_label = int(predicted_label)

                if predicted_label == class_label[g]:
                    true_density[f"{int(predicted_label)}"].append(class_one_hot[g, :])
                    for c in range(self.class_num):
                        dict_for_df[f'{predicted_label}']['result'].append('True')
                        dict_for_df[f'{predicted_label}']['label'].append(f"{c}")
                        dict_for_df[f'{predicted_label}']['prob'].append(class_one_hot[g, c])
                else:
                    false_density[f"{int(predicted_label)}"].append(class_one_hot[g, :])
                    for c in range(self.class_num):
                        dict_for_df[f'{predicted_label}']['result'].append('False')
                        dict_for_df[f'{predicted_label}']['label'].append(f"{c}")
                        dict_for_df[f'{predicted_label}']['prob'].append(class_one_hot[g, c])

        """
        # draw step_plot(take much time)
        ensemble_step(true_density,
                      false_density,
                      class_num=self.class_num)
        """

        ensemble_violin(true_density,
                        false_density,
                        dict_for_df,
                        )

    def gcmwise_true_false(self,
                           pred_class,
                           y_val_class,
                           val_model,
                           class_threshold=300,
                           model_num=42):
        model_name = ['AWI-ESM-1-1-LR',
                      'BCC-CSM2-MR',
                      'BCC-ESM1',
                      'CAS-ESM2-0',
                      'CESM2-FV2',
                      'CESM2-WACCM',
                      'CESM2',
                      'CMCC-CM2-HR4',
                      'CMCC-CM2-SR5',
                      'CMCC-ESM2',
                      'CNRM-CM6-1-HR',
                      'CNRM-CM6-1',
                      'CNRM-ESM2-1',
                      'CanESM5-CanOE',
                      'CanESM5',
                      'EC-Earth3-Veg-LR',
                      'EC-Earth3-Veg',
                      'EC-Earth3',
                      'FGOALS-f3-L',
                      'FGOALS-g3',
                      'GISS-E2-1-G-CC',
                      'GISS-E2-1-G',
                      'GISS-E2-1-H',
                      'GISS-E2-2-G',
                      'GISS-E2-2-H',
                      'GISS-E3-G',
                      'HadGEM3-GC31',
                      'ICON-ESM-LR',
                      'IPSL-CM5A2-INCA',
                      'IPSL-CM6A-LR-INCA',
                      'IPSL-CM6A-LR',
                      'MIROC-ES2H',
                      'MIROC-ES2L',
                      'MIROC6',
                      'MPI-ESM-1-2-HAM',
                      'MPI-ESM1-2-HR',
                      'MPI-ESM1-2-LR',
                      'MRI-ESM2-0',
                      'NorESM2-MM',
                      'TaiESM1',
                      'UKESM1-0-LL',
                      'UKESM1-1-LL'
                      ]
        true_gcm = {f"{i}": 0 for i in range(model_num)}
        false_gcm = {f"{i}": 0 for i in range(model_num)}

        for sample in range(self.vsample):
            # true or false
            class_one_hot = pred_class[:, sample, :]
            class_label = y_val_class[sample, :]
            truegrid_count = 0
            for g in range(self.grid_num):
                predicted_label = np.argmax(class_one_hot[g, :])
                predicted_label = int(predicted_label)
                if predicted_label == class_label[g]:
                    truegrid_count += 1
            if truegrid_count >= class_threshold:
                true_gcm[f"{val_model[sample]}"] += 1
            else:
                false_gcm[f"{val_model[sample]}"] += 1

        gcm_bars(true_gcm, false_gcm, model_name, model_num=model_num)

    def continuos_class_coherence(self, 
                                  pred_class, 
                                  pred_continuous,
                                  y_val_class,
                                  class_threshold=390,
                                  ):
        """
        match_index containes [grid_index, sample_index, x, label, bool]
        """
        boundaries = [-3.43, -0.34, -0.12, 0.09, 0.34, 2.73]
        true_match_index = [] 
        true_unmatch_index = []
        false_match_index = [] 
        false_unmatch_index = []

        for sample in range(self.vsample):
            class_one_hot = pred_class[:, sample, :]
            class_label = y_val_class[sample, :]
            correct_grid_count =0
            for g in range(self.grid_num):
                predicted_label = np.argmax(class_one_hot[g, :])
                predicted_label = int(predicted_label)
                if predicted_label == class_label[g]:
                    correct_grid_count += 1

            for g in range(self.grid_num):
                predicted_label = np.argmax(class_one_hot[g, :])
                predicted_label = int(predicted_label)
                x = pred_continuous[g, sample]
                if predicted_label == class_label[g]:
                    flag = True
                else:
                    flag = False

                if correct_grid_count >= class_threshold:
                    if boundaries[predicted_label] <= x <= boundaries[predicted_label+1]:
                        true_match_index.append([g, sample, x, predicted_label, flag])
                    else:
                        true_unmatch_index.append([g, sample, x, predicted_label, flag])
                else:
                    if boundaries[predicted_label] <= x <= boundaries[predicted_label+1]:
                        false_match_index.append([g, sample, x, predicted_label, flag])
                    else:
                        false_unmatch_index.append([g, sample, x, predicted_label, flag])

        true_match_array = np.array(true_match_index)
        true_unmatch_array = np.array(true_unmatch_index)
        false_match_array = np.array(false_match_index)
        false_unmatch_array = np.array(false_unmatch_index)

        print(len(true_match_array), len(true_unmatch_array), len(false_match_array), len(false_unmatch_array))

        for target in [true_match_array, true_unmatch_array, false_match_array, false_unmatch_array]:
            sand = target[:, 4]
            box = np.count_nonzero(sand)
            tmp = len(target) - box # numbe of false 
            print(tmp)
            print(tmp/len(target))

    def concentration_demo(self,
                          pred_class,
                          y_val_class):
        """
        pred_class: (400, 1000, 5)
        y_val_class: (1000, 400)

        sampling True and False typical probability density curve
        threshold = 0.753, 0.769, 0.790
        threshold = 0.927, 0.938, 0.947
        """
        true_density = {f"{i}": [] for i in range(self.class_num)}
        false_density = {f"{i}": [] for i in range(self.class_num)}

        t_threshold_up = 0.95
        t_threshold_bottom = 0.88
        f_threshold_up = 0.75
        f_threshold_bottom = 0.6

        for sample in range(self.vsample):
            class_one_hot = pred_class[:, sample, :]
            class_label = y_val_class[sample, :]
            for g in range(self.grid_num):
                concentration = np.max(class_one_hot[g, :])
                predicted_label = np.argmax(class_one_hot[g, :])
                predicted_label = int(predicted_label)

                if t_threshold_bottom <= concentration <= t_threshold_up:
                    true_density[f"{int(predicted_label)}"].append(class_one_hot[g, :])
                elif f_threshold_bottom <= concentration <= f_threshold_up:
                    false_density[f"{int(predicted_label)}"].append(class_one_hot[g, :])

        concentration_bar(true_density,
                         false_density,
                         )

    def ensemble_pdf_boxplot(self,
                              pred_class,
                              y_val_class,
                              class_threshold=300):
        """
        pred_class: (400, 1000, 5)
        y_val_class: (1000, 400)

        true -> average of each label probability -> boxplot
        label = dict, key: 0:4
        key = dict, key: 0:4
        """
        true_dct = {f"{i}": {'0': [], '1': [], '2': [], '3': [], '4': []} for i in range(self.class_num)}
        false_dct = {f"{i}": {'0': [], '1': [], '2': [], '3': [], '4': []} for i in range(self.class_num)}

        for sample in range(self.vsample):
            class_one_hot = pred_class[:, sample, :]
            class_label = y_val_class[sample, :]

            # true or false
            correct_grid_count =0
            for g in range(self.grid_num):
                predicted_label = np.argmax(class_one_hot[g, :])
                predicted_label = int(predicted_label)
                if predicted_label == class_label[g]:
                    correct_grid_count += 1

            # labelwise pdf at each grid
            true_pdf_of_label_eachgrid = {f"{i}": [] for i in range(self.class_num)}
            false_pdf_of_label_eachgrid = {f"{i}": [] for i in range(self.class_num)}
            for g in range(self.grid_num):
                predicted_label = np.argmax(class_one_hot[g, :])
                predicted_label = int(predicted_label)
                if correct_grid_count >= class_threshold:
                    true_pdf_of_label_eachgrid[f"{predicted_label}"].append([class_one_hot[g, :]])
                else:
                    false_pdf_of_label_eachgrid[f"{predicted_label}"].append([class_one_hot[g, :]])

            # mean pdf of the same predicted label 
            if correct_grid_count >= class_threshold:
                true_mean = [np.mean(true_pdf_of_label_eachgrid[f"{i}"], axis = 0) \
                             for i in range(self.class_num)]
                for label in range(self.class_num):
                    dct = true_dct[f"{label}"]
                    mean = np.array(true_mean[label])
                    mean = np.squeeze(mean)
                    if mean.ndim == 0:
                        print('empty') 
                    else:
                        for i in range(self.class_num):
                            dct[f"{i}"].append(mean[i])
            else:
                false_mean = [np.mean(false_pdf_of_label_eachgrid[f"{i}"], axis = 0) \
                             for i in range(self.class_num)]
                for label in range(self.class_num):
                    dct = false_dct[f"{label}"]
                    mean = np.array(false_mean[label])
                    mean = np.squeeze(mean)
                    if mean.ndim == 0:
                        print('empty') 
                    else:
                        for i in range(self.class_num):
                            dct[f"{i}"].append(mean[i])

        pdf_box(true_dct,
                false_dct
               )

    def nrmse_vs_accuracy(self,
                          pred_class,
                          pred_continuous,
                          y_val_class,
                          y_val_continuous,
                          class_threshold=300,
                          continuous_threshold=0.3):
        """
        pred_class: (400, 1000, 5)
        pred_continuous: (400, 1000)
        y_val_class: (1000, 400)
        y_val_continuous: (1000, 400)
        """
        # reliability and nrmse calculation
        accuracy_lst = []
        nrmse_lst = []

        true_accuracy_lst = []
        false_accuracy_lst = []

        true_nrmse_lst = []
        false_nrmse_lst = []

        for sample in range(self.vsample):
            # accuracy
            grid_true, grid_false = 0, 0
            class_one_hot = pred_class[:, sample, :]
            class_label = y_val_class[sample, :]
            for g in range(self.grid_num):
                pred_label = np.argmax(class_one_hot[g, :])
                if int(pred_label) == class_label[g]:
                    grid_true += 1
                else:
                    grid_false += 1
            accuracy = grid_true
            accuracy_lst.append(accuracy)

            # nrmse
            continuous = pred_continuous[:, sample]
            continuous_label = y_val_continuous[sample, :]
            gridmean_nrmse = np.sqrt(np.mean((continuous - continuous_label)**2))
            nrmse_lst.append(gridmean_nrmse)

            # classification
            if int(accuracy) >= class_threshold and gridmean_nrmse <= continuous_threshold:
                true_accuracy_lst.append(accuracy)
                true_nrmse_lst.append(gridmean_nrmse)
            elif int(accuracy) < class_threshold and gridmean_nrmse > continuous_threshold:
                false_accuracy_lst.append(accuracy)
                false_nrmse_lst.append(gridmean_nrmse)

        # plot
        scatter_and_marginal_and_bimodal(accuracy_lst,
                                         nrmse_lst,
                                         true_accuracy_lst,
                                         true_nrmse_lst,
                                         false_accuracy_lst,
                                         false_nrmse_lst,
                                         )

    def accuracy_vs_reliability(self,
                          pred_class,
                          y_val_class,
                          class_threshold=300):
        """
        pred_class: (400, 1000, 5)
        pred_continuous: (400, 1000)
        y_val_class: (1000, 400)
        y_val_continuous: (1000, 400)
        """
        # reliability and nrmse calculation
        accuracy_lst = []
        true_accuracy_lst = []
        false_accuracy_lst = []

        reliability_lst = []
        true_reliability_lst =  []
        false_reliability_lst =  []

        for sample in range(self.vsample):
            # accuracy & grids_reliability
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
            accuracy = grid_true
            accuracy_lst.append(accuracy)

            # reliability
            gridmean_reliability = np.mean(grids_reliability)
            reliability_lst.append(gridmean_reliability)

            # classification
            if int(accuracy) >= class_threshold:
                true_accuracy_lst.append(accuracy)
                true_reliability_lst.append(gridmean_reliability)
            elif int(accuracy) < class_threshold:
                false_accuracy_lst.append(accuracy)
                false_reliability_lst.append(gridmean_reliability)

        # plot
        show_accuracy_vs_reliability(accuracy_lst,
                                     reliability_lst,
                                     true_accuracy_lst,
                                     true_reliability_lst,
                                     false_accuracy_lst,
                                     false_reliability_lst,
                                     )
#############################################################################


if __name__ == '__main__':
    main()
