import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from gradcam import grad_cam, show_heatmap, image_preprocess, box_gradcam_continuous, box_gradcam_class
from class_conversion import open_pickle, init_model, prediction, to_class, mk_true_false_list, print_acc
from view import draw_val


class evaluate():
    def __init__(self):
        # file 
        self.class_num = 10
        self.discrete_mode = 'EFD'
        self.epochs = 100
        self.batch_size = 256
        self.seed = 1
        self.var_num = 4
        self.vsample = 1000
        # path
        # val_path and weights_path doesn't need {discrete_mode} and {class_num}
        # since they are continuous prediction
        self.tors = 'predictors_coarse_std_Apr_msot'
        self.tant = 'pr_1x1_std_MJJASO_one'
        self.workdir = "/docker/mnt/d/research/D2/cnn3"
        self.val_path = self.workdir + f"/train_val/continuous/{self.tors}-{self.tant}.pickle"
        self.weights_dir = self.workdir + f"/weights/continuous/{self.tors}-{self.tant}"
        self.weights_path = self.weights_dir + f"/epoch{self.epochs}_batch{self.batch_size}_seed{self.seed}.h5"
        self.bnd_path = self.workdir + f"/boundaries/{self.tant}_{self.discrete_mode}_{self.class_num}.npy"
        self.heatmap_dir = self.workdir + f"/heatmap/continuous/{self.tors}-{self.tant}_{self.discrete_mode}_{self.class_num}"
        self.heatmap_original_path = self.heatmap_dir + f"/ORIGINAL_{self.epochs}_batch{self.batch_size}_seed{self.seed}.npy"
        self.heatmap_converted_path = self.heatmap_dir + f"/CONVERTED_{self.epochs}_batch{self.batch_size}_seed{self.seed}.npy"
        # model
        # init_model is allowed to be called once otherwise layer_name will be messed up
        self.lat, self.lon = 24, 72
        self.lr = 0.0001
        self.model = init_model(self.weights_path, lat=self.lat, lon=self.lon, var_num=self.var_num, lr=self.lr)
        # validation
        self.validation_view_flag = False
        # gradcam
        self.gradcam_view_flag = False
        self.layer_name = 'conv2d_2'
        self.true_false_bool = False
        self.false_index = 0
        self.true_index = 0
        # gradcam-box
        self.grad_box_view_flag = False
        self.threshold = 0.6 # for extent of importance map
        self.criteria = 0.4 # for continuous prediction
        # gradcam-mean
        self. gradmean_view_flag = False
        self.gradmean_option = "PredictionTure" # PredictionTrue, PredictionFalse, SameLabelPrediction, SameLabelFalse
        self.gradmean_label = 7 # what's this?

    def load_pred(self):
        x_val, y_val = open_pickle(self.val_path)
        bnd = np.load(self.bnd_path)
        pred = prediction(self.model, x_val)
        pred_class = to_class(pred.reshape(-1), bnd, print_flag=False)
        y_class = to_class(y_val.reshape(-1), bnd, print_flag=False)
        return x_val, y_val, pred, pred_class, y_class

    def check_false_by_label(self, pred, y):
        false_dct = {f"{i}": [] for i in range(self.class_num)}
        for target_label in range(self.class_num):
            print(f"target_label={target_label}")
            for i, j in zip(pred, y):
                if int(i) != int(j) and int(j) == target_label:
                    false_dct[f"{target_label}"].append(int(i))
            print(false_dct[f"{target_label}"])
        return false_dct

    def ture_false_bar(self, pred, y):
        true_lst, false_lst = mk_true_false_list(pred, y)
        print_acc(true_lst, false_lst, class_num=self.class_num)
        draw_val(true_lst, false_lst, class_num=self.class_num)

    def gradcam_converted(self, x_val, pred_class, y_class, true_index=0, false_index=0):
        true_lst, false_lst = mk_true_false_list(pred_class, y_class)
        if self.true_false_bool is True:
            selected_index = int(true_lst[true_index])
        else:
            selected_index = int(false_lst[false_index])
        preprocessed_image = image_preprocess(x_val, gradcam_index=selected_index)
        heatmap = grad_cam(self.model, preprocessed_image, y_class[selected_index], self.layer_name, lat=self.lat, lon=self.lon)
        show_heatmap(heatmap)

    def mk_heatmap_original(self, x_val, y):
        # pred and y must be continuous number
        if os.path.exists(self.heatmap_original_path) is True:
            heatmap_arr = np.load(self.heatmap_original_path)
        else:
            heatmap_arr = np.empty((self.vsample, self.lat, self.lon)) # shape=(1000, 24, 72)
            for index in range(len(y)):
                preprocessed_image = image_preprocess(x_val, gradcam_index=index)
                heatmap = grad_cam(self.model, preprocessed_image, y[index], self.layer_name, lat=self.lat, lon=self.lon)
                heatmap_arr[index] = heatmap 
                print(index)
            os.makedirs(self.heatmap_dir, exist_ok=True)
            np.save(self.heatmap_original_path, heatmap_arr)
        return heatmap_arr

    def mk_heatmap_converted(self, x_val, y_class):
        if os.path.exists(self.heatmap_converted_path) is True:
            heatmap_arr = np.load(self.heatmap_converted_path)
        else:
            heatmap_arr = np.empty((self.vsample, self.lat, self.lon)) # shape=(1000, 24, 72)
            for index in range(len(y_class)):
                preprocessed_image = image_preprocess(x_val, gradcam_index=index)
                heatmap = grad_cam(self.model, preprocessed_image, y_class[index], self.layer_name, lat=self.lat, lon=self.lon)
                heatmap_arr[index] = heatmap 
                print(index)
            os.makedirs(self.heatmap_dir, exist_ok=True)
            np.save(self.heatmap_converted_path, heatmap_arr)
        return heatmap_arr

    def gradbox_original(self, heatmap_arr, pred, y):
        # pred and y must be continuous number
        box_gradcam_continuous(heatmap_arr, pred, threshold=self.threshold, criteria=self.criteria)

    def gradbox_converted(self, heatmap_arr, pred_class, y_class):
        # pred and y must be class form
        box_gradcam_class(heatmap_arr, pred_class, y_class, threshold=self.threshold, class_num=self.class_num)

    def gradmean_converted(self, heatmap_arr, pred_class, y_class):
        if self.gradmean_option== "PredictionTrue":
            # predicted label is the same, prediction is true
            prediction_true = []
            for ind, pr, y in enumerate(zip(pred_class, y_class)):
                if  pr == y and y == self.gradmean_label:
                    prediction_true.append(ind)
            indeces = prediction_true
            print(f"gradmean result; sample: {len(indeces)}, label: {self.gradmean_label}, flag: {self.gradmean_option}")
        elif self.gradmean_option == "PredctionFalse":
            # predicted label is the random, prediction is false
            prediction_false = []
            for ind, pr, y in enumerate(zip(pred_class, y_class)):
                if  pr != y and y == self.gradmean_label:
                    prediction_false.append(ind)
            indeces = prediction_false
            print(f"gradmean result; sample: {len(indeces)}, label: {self.gradmean_label}, flag: {self.gradmean_option}")
        elif self.gradmean_option == "SameLabelPrediction":
            # predicted label is the same
            same_label_prediction = [] 
            for ind, pr in enumerate(pred_class):
                if pr == self.gradmean_label:
                    same_label_prediction.append(ind)
            indeces = same_label_prediction
            print(f"gradmean result; sample: {len(indeces)}, label: {self.gradmean_label}, flag: {self.gradmean_option}")
        elif self.gradmean_option == "SameLabelFalse":
            # predicte label is the same and predition is false
            same_label_false = [] 
            for ind, (pr, y) in enumerate(zip(pred_class, y_class)):
                if  pr != y and pr == self.gradmean_label:
                    same_label_false.append(ind)
            indeces = same_label_false
            print(f"gradmean result; sample: {len(indeces)}, label: {self.gradmean_label}, flag: {self.gradmean_option}")
        else:
            print("error: gradmean_option is wrong")
            exit()

        heatmap_mean = heatmap_arr[indeces].mean(axis=0)
        show_heatmap(heatmap_mean)


if __name__ == '__main__':
    # view_flag bool must be added in main function
    EVAL = evaluate()
    x_val, y_val, pred, pred_class, y_class = EVAL.load_pred()
    if EVAL.validation_view_flag is True:
        false_dct = EVAL.check_false_by_label(pred_class, y_class)
        EVAL.ture_false_bar(pred_class, y_class)
    if EVAL.gradcam_view_flag is True:
        EVAL.gradcam_converted(x_val, pred_class, y_class)
    if EVAL.grad_box_view_flag is True:
        heatmap_arr_orginal = EVAL.mk_heatmap_original(x_val, y_val)
        heatmap_arr_converted = EVAL.mk_heatmap_converted(x_val, y_class)
        EVAL.gradbox_original(heatmap_arr_orginal, pred, y_val)
        EVAL.gradbox_converted(heatmap_arr_converted, pred_class, y_class)
    if EVAL.gradmean_view_flag is True:
        EVAL.gradmean_converted(heatmap_arr_converted, pred_class, y_class)

