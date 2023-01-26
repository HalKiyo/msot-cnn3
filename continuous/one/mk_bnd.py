import os
import bisect
import numpy as np
from view import draw_distribution

def main():
    # init
    save_flag = False
    class_num = 5
    discrete_mode = 'EFD'
    tors = 'predictors_coarse_std_Apr_msot'
    tant = 'pr_1x1_std_MJJASO_one'
    workdir = "/docker/mnt/d/research/D2/cnn3"
    val_path = workdir + f"/train_val/continuous/{tors}-{tant}.pickle"
    # For thailand in continuous mode, you need download it from xeno server
    one_path = workdir + f"predictant/continuous/{tant}.npy"
    one_spath = workdir + f"/boundaries/{tant}_{discrete_mode}_{class_num}.npy"

    # calculate
    one = np.load(one_path) # one=(42, 165)
    one_flat = one.reshape(42*165) # one_flat=(42*165,)
    if discrete_mode == 'EFD':
        one_class, one_bnd = one_EFD(one, class_num=class_num)
    elif discrete_mode == 'EWD':
        one_class, one_bnd = one_EWD(one, class_num=class_num)
    print(f"one_bnd: {one_bnd}")
    save_npy(one_spath, one_class, save_flag=save_flag)
    draw_distribution(one_flat, one_bnd)

def save_npy(path, data, save_flag=False):
    if save_flag is True:
        np.save(path, data)
        print("class boundaries has been SAVED")
    else:
        print("class boundaries is ***NOT*** saved yet")

def one_EWD(pr_flat, class_num=5):
    lim = max(abs(max(pr_flat)), abs(min(pr_flat)))
    dx = 2*lim/class_num

    bnd_show = []
    bnd_show.append(-lim - 1e-10) # min boudary for show must be a bit lower than real min
    bnd_show.append(lim + 1e-10) # max boudary for show must be a bit higher than real max

    bnd_pred = []
    bnd_pred.append(-lim - 1e10) # min boudary for prediction must be much lower than min
    bnd_pred.append(lim + 1e10) # max boudary for prediction must be much higher than real max

    # class_num = even or odd?
    if class_num%2 == 0:
        origin = 0
        bnd_show.append(origin)
        bnd_pred.append(origin)
    else:
        origin = dx/2
        bnd_show.append(origin)
        bnd_show.append(-origin)
        bnd_pred.append(origin)
        bnd_pred.append(-origin)

