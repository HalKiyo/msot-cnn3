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
        self.discrete_mode = 'EWD'
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
        self.lat, self.lon = 24, 72
        self.lr = 0.0001
        # validation
        self.validation_view_flag = False
        # gradcam
        self.grad_view_flag = False
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
        self.gradmean_flag = "PredictionTure" # PredictionTrue, PredictionFalse, SameLabelPrediction, SameLabelFalse
        self.gradmean_label = 7
# view_flag bool must be added in main function

    def load_pred(self):
        x_val, y_val = open_pickle(self.val_path)
        bnd = np.load(self.bnd_path)
        model = init_model(self.weights_path, lat=self.lat, lon=self.lon, var_num=self.var_num, lr=self.lr)
        pred = prediction(model, x_val)
        pred_class = to_class(pred.reshape(-1), bnd)
        y_class = to_class(y_val.reshape(-1), bnd, print_flag=True)
        return pred_class, y_class

    def check_false_by_label(self, pred, y):
        #true_lst, false_lst = mk_true_false_list(pred, y)
        false_dct = {f"{i}": [] for i in range(self.class_num)}
        for target_label in range(self.class_num):
            print(f"target_label={target_label}")
            for i, j in zip(pred, y):
                if i != j and j == target_label:
                    false_dct[f"{j}"].append(i)
            print(false_dct[f"{j}"])
        return false_dct

    def ture_false_bar(self, pred, y):
        true_lst, false_lst = mk_true_false_list(pred, y)
        print_acc(true_lst, false_lst, class_num=self.class_num)
        draw_val(true_lst, false_lst, class_num=self.class_num)

    def individual_gradcam(self, x_val, pred, y, true_index=0, false_index=0):
        true_lst, false_lst = mk_true_false_list(pred, y)
        if self.true_false_bool is True:
            selected_index = true_lst[true_index]
        else:
            selected_index = false_lst[false_index]
        preprocessed_image = image_preprocess(x_val, gradcam_index=selected_index)
        model = init_model(self.weights_path, lat=self.lat, lon=self.lon, var_num=self.var_num, lr=self.lr)
        heatmap = grad_cam(model, preprocessed_image, y[selected_index], self.layer_name, lat=self.lat, lon=self.lon)
        show_heatmap(heatmap)

    def gradbox_converted(self, x_val, pred_class, y_class):
        # pred and y must be class form
        model = init_model(self.weights_path, lat=self.lat, lon=self.lon, var_num=self.var_num, lr=self.lr)
        if os.path.exists(self.heatmap_converted_path) is True:
            heatmap_arr = np.load(self.heatmap_converted_path)
        else:
            heatmap_arr = np.empty((self.vsample, self.lat, self.lon)) # shape=(1000, 24, 72)
            for index in range(len(y_class)):
                preprocessed_image = image_preprocess(x_val, gradcam_index=index)
                heatmap = grad_cam(model, preprocessed_image, y_class[index], self.layer_name, lat=self.lat, lon=self.lon)
                heatmap_arr[index] = heatmap 
                print(index)
            os.makedirs(self.heatmap_dir, exist_ok=True)
            np.save(self.heatmap_converted_path, heatmap_arr)
        box_gradcam_class(heatmap_arr, pred_class, y_class, threshold=self.threshold, class_num=self.class_num)

    def gradbox_original(self, x_val, pred, y):
        # pred and y must be continuous number
        model = init_model(self.weights_path, lat=self.lat, lon=self.lon, var_num=self.var_num, lr=self.lr)
        if os.path.exists(self.heatmap_original_path) is True:
            heatmap_arr = np.load(self.heatmap_original_path)
        else:
            heatmap_arr = np.empty((self.vsample, self.lat, self.lon)) # shape=(1000, 24, 72)
            for index in range(len(y)):
                preprocessed_image = image_preprocess(x_val, gradcam_index=index)
                heatmap = grad_cam(model, preprocessed_image, y[index], self.layer_name, lat=self.lat, lon=self.lon)
                heatmap_arr[index] = heatmap 
                print(index)
            os.makedirs(self.heatmap_dir, exist_ok=True)
            np.save(self.heatmap_original_path, heatmap_arr)
        box_gradcam_continuous(heatmap_arr, pred, threshold=self.threshold, criteria=self.criteria)


    def gradmean_converted(self, heatmap_arr, pr_class, y_class):
        if self.gradmean_flag == "PredictionTrue":
            # predicted label is the same, prediction is true
            prediction_true = []
            for ind in len(y_class):
                if np.argmax(pr) == np.argmax(y) and int(np.argmax(y)) == gradmean_label:
                    prediction_true.append(ind)
            indeces = prediction_true
            print(f"gradmean result; sample: {len(indeces)}, label: {gradmean_label}, flag: {self.gradmean_flag}")
        elif self.gradmean_flag == "PredctionFalse":
            # predicted label is the random, prediction is false
            prediction_false = []
            for ind, (pr, y) in enumerate(zip(pred_val, y_val_one_hot)):
                if np.argmax(pr) != np.argmax(y) and int(np.argmax(y)) == gradmean_label:
                    prediction_false.append(ind)
            indeces = prediction_false
            print(f"gradmean result; sample: {len(indeces)}, label: {gradmean_label}, flag: {self.gradmean_flag}")
        elif self.gradmean_flag == "SameLabelPrediction":
            # predicted label is the same
            same_label_prediction = [] 
            for ind, pr in enumerate(pred_val):
                if int(np.argmax(pr)) == gradmean_label:
                    same_label_prediction.append(ind)
            indeces = same_label_prediction
            print(f"gradmean result; sample: {len(indeces)}, label: {gradmean_label}, flag: {self.gradmean_flag}")
        elif self.gradmean_flag == "SameLabelFalse":
            # predicte label is the same and predition is false
            same_label_false = [] 
            for ind, (pr, y) in enumerate(zip(pred_val, y_val_one_hot)):
                if np.argmax(pr) != np.argmax(y) and int(np.argmax(pr)) == gradmean_label:
                    same_label_false.append(ind)
            indeces = same_label_false
            print(f"gradmean result; sample: {len(indeces)}, label: {gradmean_label}, flag: {self.gradmean_flag}")
        else:
            print("error: gradmean_flag is wrong")
            exit()

        heatmap_mean = heatmap_arr[indeces].mean(axis=0)
        show_heatmap(heatmap_mean)


if __name__ == '__main__':
    # view_flag bool must be added in main function

