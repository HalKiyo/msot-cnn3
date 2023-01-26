import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from gradcam import grad_cam, show_heatmap, image_preprocess, box_gradcam
from class_conversion import open_pickle, prediction, to_class
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
        # path
        self.tors = 'predictors_coarse_std_Apr_msot'
        self.tant = 'pr_1x1_std_MJJASO_one'
        self.workdir = "/docker/mnt/d/research/D2/cnn3"
        self.val_path = self.workdir + f"/train_val/continuous/{self.tors}-{self.tant}.pickle"
        self.bnd_path = self.workdir + f"/boundaries/{self.tant}_{self.discrete_mode}_{self.class_num}.npy"
        self.weights_dir = self.workdir + f"/weights/continuous/{self.tors}-{self.tant}"
        self.weights_path = self.weights_dir + f"/epoch{self.epochs}_batch{self.batch_size}_seed{self.seed}.h5"
        # model
        self.lat, self.lon = 24, 72
        self.lr = 0.0001
        # validation
        self.validation_view_flag = False
        # gradcam
        self.grad_view_flag = False
        self.layer_name = 'conv2d_2'
        # gradcam-box
        self.grad_box_flag = True
        self.threshold = 0.6
        # gradcam-mean
        self. gradmean_view_flag = True
        self.gradmean_flag = "PredictionTure" # PredictionTrue, PredictionFalse, SameLabelPrediction, SameLabelFalse
        self.gradmean_label = 7

    def load_pred(self):
        x_val, y_val = open_pickle(self.val_path)
        bnd = np.load(self.bnd_path)
        pred = prediction(x_val, self.weights_path, lat=self.lat, lon=self.lon, var_num=self.var_num, lr=self.lr)
        pred_class = to_class(pred.reshape(-1), bnd)
        y_class = to_class(y_val.reshape(-1), bnd, print_flag=True)
        return pred_class, y_class

    def validation(self, pred, y):
        for validation_label in range(self.class_num):
            print(f"validation_label={validation_label}")
            wrong = []
            for i, j in zip(pred, y):
                if i != j and j == validation_label:
                    wrong.append(i)
            print(wrong)

    def ture_false_bar(self, pred, y):
        class_label, counts = draw_val(pred, y, class_num=self.class_num)
        print(f"class_label: {class_label} \ncounts: {counts}")

# 3. What to show: individual gradcam, boxplot of gradcam, heatmap average of gradcam
    def individual_gradcam(self, pred, y, true_index=0, false_index=0)
        if prob_flag is True:
            gradcam_index = true_lst[true_index]
        else:
            gradcam_index = false_lst[false_index]
        preprocessed_image = image_preprocess(x_val, gradcam_index=gradcam_index)
        heatmap = grad_cam(model, preprocessed_image, y_val[gradcam_index], layer_name,
                           lat=24, lon=72, class_num=class_num)
        show_heatmap(heatmap)


def main():
    #---3. true/false barplot
    if validation_view_flag is True:

    #---4. individual gradcam
    if grad_view_flag is True:
        if prob_flag is True:
            gradcam_index = true_lst[true_index]
        else:
            gradcam_index = false_lst[false_index]
        preprocessed_image = image_preprocess(x_val, gradcam_index=gradcam_index)
        heatmap = grad_cam(model, preprocessed_image, y_val[gradcam_index], layer_name,
                           lat=24, lon=72, class_num=class_num)
        show_heatmap(heatmap)

    #---4.1 boxplot of gradcam 
    if grad_box_flag is True:
        heatmap_dir = f"/docker/mnt/d/research/D2/cnn3/heatmap/class/{tors}-{tant}"
        heatmap_path = heatmap_dir + f"/class{class_num}_epoch{epochs}_batch{batch_size}.npy"
        if os.path.exists(heatmap_path) is True:
            heatmap_arr = np.load(heatmap_path)
        else:
            heatmap_arr = np.empty((vsample, lat, lon)) # shape=(1000, 24, 72)
            for index, (pr, y) in enumerate(zip(pred_val, y_val_one_hot)):
                preprocessed_image = image_preprocess(x_val, gradcam_index=index)
                heatmap = grad_cam(model, preprocessed_image, y_val[index], layer_name,
                                   lat=24, lon=72, class_num=class_num)
                heatmap_arr[index] = heatmap 
                print(index)
            os.makedirs(heatmap_dir, exist_ok=True)
            np.save(heatmap_path, heatmap_arr)
        box_gradcam(heatmap_arr, pred_val, y_val_one_hot, threshold=threshold, class_num=class_num)

    #---4.2 heatmap average of gradcam
        if gradmean_view_flag is True:
            if gradmean_flag == "PredictionTrue":
                # predicted label is the same, prediction is true
                prediction_true = []
                for ind, (pr, y) in enumerate(zip(pred_val, y_val_one_hot)):
                    if np.argmax(pr) == np.argmax(y) and int(np.argmax(y)) == gradmean_label:
                        prediction_true.append(ind)
                indeces = prediction_true
                print(f"gradmean result; sample: {len(indeces)}, label: {gradmean_label}, flag: {gradmean_flag}")
            elif gradmean_flag == "PredctionFalse":
                # predicted label is the random, prediction is false
                prediction_false = []
                for ind, (pr, y) in enumerate(zip(pred_val, y_val_one_hot)):
                    if np.argmax(pr) != np.argmax(y) and int(np.argmax(y)) == gradmean_label:
                        prediction_false.append(ind)
                indeces = prediction_false
                print(f"gradmean result; sample: {len(indeces)}, label: {gradmean_label}, flag: {gradmean_flag}")
            elif gradmean_flag == "SameLabelPrediction":
                # predicted label is the same
                same_label_prediction = [] 
                for ind, pr in enumerate(pred_val):
                    if int(np.argmax(pr)) == gradmean_label:
                        same_label_prediction.append(ind)
                indeces = same_label_prediction
                print(f"gradmean result; sample: {len(indeces)}, label: {gradmean_label}, flag: {gradmean_flag}")
            elif gradmean_flag == "SameLabelFalse":
                # predicte label is the same and predition is false
                same_label_false = [] 
                for ind, (pr, y) in enumerate(zip(pred_val, y_val_one_hot)):
                    if np.argmax(pr) != np.argmax(y) and int(np.argmax(pr)) == gradmean_label:
                        same_label_false.append(ind)
                indeces = same_label_false
                print(f"gradmean result; sample: {len(indeces)}, label: {gradmean_label}, flag: {gradmean_flag}")
            else:
                print("error gradmean_flag is wrong")
                exit()

            heatmap_mean = heatmap_arr[indeces].mean(axis=0)
            show_heatmap(heatmap_mean)


if __name__ == '__main__':
    main()

