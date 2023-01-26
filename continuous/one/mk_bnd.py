import numpy as np
from view import draw_distribution

def main():
    # init
    save_flag = False
    class_num = 5
    discrete_mode = 'EWD'
    tors = 'predictors_coarse_std_Apr_msot'
    tant = 'pr_1x1_std_MJJASO_one'
    workdir = "/docker/mnt/d/research/D2/cnn3"
    # For thailand in continuous mode, you need download it from xeno server
    one_path = workdir + f"/predictant/continuous/{tant}.npy"
    bnd_spath = workdir + f"/boundaries/{tant}_{discrete_mode}_{class_num}.npy"

    # calculate
    one = np.load(one_path) # one=(42, 165)
    one_flat = one.reshape(42*165) # one_flat=(42*165,)
    if discrete_mode == 'EFD':
        bnd = one_EFD(one_flat, class_num=class_num)
    elif discrete_mode == 'EWD':
         bnd = one_EWD(one_flat, class_num=class_num)
    save_npy(bnd_spath, bnd, save_flag=save_flag)

def save_npy(path, data, save_flag=False):
    if save_flag is True:
        np.save(path, data)
        print("class boundaries has been SAVED")
    else:
        print("class boundaries is ***NOT*** saved yet")

def one_EFD(pr_flat, class_num=5):
    flat_sorted = np.sort(pr_flat)

    if len(flat_sorted)%class_num == 0:
        batch_sample = int(len(flat_sorted)/class_num)
    else:
        print('sample size is indivisible by class_num')

    # flat_sorted[0] -> minimum value is added
    bnd_show = [flat_sorted[i] for i in range(0, len(flat_sorted), batch_sample)]
    # flat_sorted[-1] + 1e-10 -> maximum value + delta is added
    bnd_show.append(flat_sorted[-1] + 1e-10)
    # flat_sorted[0] - 1e-10 -> minimum value - delta is replaced
    bnd_show[0] = bnd_show[0] - 1e-10

    print(f"one_bnd: {bnd_show}")
    draw_distribution(pr_flat, bnd_show)

    bnd_pred = [flat_sorted[i] for i in range(0, len(flat_sorted), batch_sample)]
    bnd_pred.append(flat_sorted[-1] + 1e10)
    bnd_pred[0] = bnd_pred[0] - 1e10

    return np.array(bnd_pred)

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

    if class_num == 4 or 5:
        bnd_show.append(origin+dx)
        bnd_show.append(-origin-dx)
        bnd_pred.append(origin+dx)
        bnd_pred.append(-origin-dx)
    elif class_num >= 6:
        loop_num = int(class_num/2)
        for i in range(loop_num-1):
            bnd_show.append(origin+dx*(i+1))
            bnd_show.append(-origin-dx*(i+1))
            bnd_pred.append(origin+dx*(i+1))
            bnd_pred.append(-origin-dx*(i+1))

    bnd_show = np.sort(bnd_show)
    print(f"one_bnd: {bnd_show}")
    draw_distribution(pr_flat, bnd_show)

    return np.array(np.sort(bnd_pred))

if __name__ == '__main__':
    main()
