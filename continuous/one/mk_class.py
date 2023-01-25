import os
import bisect
import numpy as np

def main():
    save_flag = False
    class_num = 5
    discrete_mode = 'EFD'
    tors = 'predictors_coarse_std_Apr_msot'
    tant = 'pr_1x1_std_MJJASO_one'
    workdir = f"/docker/mnt/d/research/D2/cnn3/train_val/continuous/{tors}-{tant}.pickle"
