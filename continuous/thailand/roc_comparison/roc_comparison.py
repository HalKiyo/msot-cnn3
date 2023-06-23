import numpy as np
import pickle
from scipy import stats
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def main():
	result_msot_MJJASO = f"/docker/mnt/d/research/D2/cnn3/result/continuous/thailand/1x1" \
						f"/predictors_coarse_std_Apr_msot-pr_1x1_std_MJJASO_thailand" \
						f"/epoch100_batch256_seed1.npy"
	val_msot_MJJASO = f"/docker/mnt/d/research/D2/cnn3/train_val/continuous" \
					f"/predictors_coarse_std_Apr_msot-pr_1x1_std_MJJASO_thailand.pickle"

	result_m_MJJASO = f"/docker/mnt/d/research/D2/cnn3/result/continuous/thailand/1x1" \
					f"/predictors_coarse_std_Apr_m-pr_1x1_std_MJJASO_thailand" \
					f"/epoch100_batch256_seed1.npy"
	val_m_MJJASO = f"/docker/mnt/d/research/D2/cnn3/train_val/continuous" \
					f"/predictors_coarse_std_Apr_m-pr_1x1_std_MJJASO_thailand.pickle"

	result_s_MJJASO = f"/docker/mnt/d/research/D2/cnn3/result/continuous/thailand/1x1" \
					f"/predictors_coarse_std_Apr_s-pr_1x1_std_MJJASO_thailand" \
						f"/epoch100_batch256_seed1.npy"
	val_s_MJJASO = f"/docker/mnt/d/research/D2/cnn3/train_val/continuous" \
				f"/predictors_coarse_std_Apr_s-pr_1x1_std_MJJASO_thailand.pickle"

	result_o_MJJASO = f"/docker/mnt/d/research/D2/cnn3/result/continuous/thailand/1x1" \
					f"/predictors_coarse_std_Apr_o-pr_1x1_std_MJJASO_thailand" \
						f"/epoch100_batch256_seed1.npy"
	val_o_MJJASO = f"/docker/mnt/d/research/D2/cnn3/train_val/continuous" \
				f"/predictors_coarse_std_Apr_o-pr_1x1_std_MJJASO_thailand.pickle"

	result_t_MJJASO =    f"/docker/mnt/d/research/D2/cnn3/result/continuous/thailand/1x1" \
						f"/predictors_coarse_std_Apr_t-pr_1x1_std_MJJASO_thailand" \
						f"/epoch100_batch256_seed1.npy"
	val_t_MJJASO = f"/docker/mnt/d/research/D2/cnn3/train_val/continuous" \
				f"/predictors_coarse_std_Apr_t-pr_1x1_std_MJJASO_thailand.pickle"

	result_mo_MJJASO =   f"/docker/mnt/d/research/D2/cnn3/result/continuous/thailand/1x1" \
				f"/predictors_coarse_std_Apr_mo-pr_1x1_std_MJJASO_thailand" \
				f"/epoch100_batch256_seed1.npy"
	val_mo_MJJASO = f"/docker/mnt/d/research/D2/cnn3/train_val/continuous" \
				f"/predictors_coarse_std_Apr_mo-pr_1x1_std_MJJASO_thailand.pickle"

	result_so_MJJASO =  f"/docker/mnt/d/research/D2/cnn3/result/continuous/thailand/1x1" \
					f"/predictors_coarse_std_Apr_so-pr_1x1_std_MJJASO_thailand" \
					f"/epoch100_batch256_seed1.npy"
	val_so_MJJASO = f"/docker/mnt/d/research/D2/cnn3/train_val/continuous" \
					f"/predictors_coarse_std_Apr_so-pr_1x1_std_MJJASO_thailand.pickle"

	result_ot_MJJASO = f"/docker/mnt/d/research/D2/cnn3/result/continuous/thailand/1x1" \
					f"/predictors_coarse_std_Apr_ot-pr_1x1_std_MJJASO_thailand" \
						f"/epoch100_batch256_seed1.npy"
	val_ot_MJJASO = f"/docker/mnt/d/research/D2/cnn3/train_val/continuous" \
					f"/predictors_coarse_std_Apr_ot-pr_1x1_std_MJJASO_thailand.pickle" 

	input_path_list = [val_msot_MJJASO,
					val_o_MJJASO,
					val_m_MJJASO,
					val_t_MJJASO,
					val_s_MJJASO]
	result_path_list = [result_msot_MJJASO,
						result_o_MJJASO,
						result_m_MJJASO,
						result_t_MJJASO,
						result_s_MJJASO]
	name_list = ['Experiment 8', 'Experiment 1', 'Experiment 2', 'Experiment 3', 'Experiment 4']
	color_list = ['dimgrey', 'dodgerblue', 'olive', 'goldenrod', 'powderblue']

# result_mean has (11, 2)shape
	roc_list = []
	auc_list = []
	for inp, res, name in zip(input_path_list, result_path_list, name_list):
		print(f"experiment result of {name}")
		x_val, y_val, pred = load_pred(inp, res)
		result, result_mean, auc_all, mean_auc = auc_sample_mean(pred.T, y_val)
		roc_list.append(result_mean)
		auc_list.append(mean_auc)
	roc_array = np.array(roc_list)
	auc_array = np.array(auc_list)
		
# draw comparison graphs
	draw_roc_curve(roc_array, auc_array, name_list, color_list)


####################################################################################

def load_pred(path, result_path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    x_val, y_val = data['x_val'], data['y_val']
    pred_arr = np.squeeze(np.load(result_path))
    return x_val, y_val, pred_arr

# ROC curve of AUC_mean of all experiments
def roc(sim, obs, percentile=20):
    sim = np.abs(sim)
    obs = np.abs(obs)
    
    sim_per =np.percentile(sim, percentile)
    obs_per = np.percentile(obs, percentile)
    
    over_per = sum(obs > obs_per)
    under_per = sum(obs <= obs_per)
    
    hit_count = 0
    false_count = 0
    for p in range(len(obs)):
        if sim[p] > sim_per and obs[p] > obs_per:
            hit_count += 1
        elif sim[p] > sim_per and obs[p] <= obs_per:
            false_count += 1
            
    hr = hit_count/over_per
    far = false_count/under_per
    
    return hr, far

def auc_sample_mean(sim, obs):
    """
    input: validation sample
        Calculation based on percentile of 1000 validation samples in each pixcel
        pred.T -> sim: (1000, 1728)
        y_val -> obs: (1000, 1728)
    output: AUC of pixcel map
    
    number of AUC -> pixcel map
    """
    # percentile variation list
    per_list = np.arange(10, 100, 10) # 10...90
    per_list = per_list[::-1]
    
    # result(11, 2, 1728) -> (percentile, hr or far, pixcel)
    result = []
    
    # initialize hr & far
    hr_all, far_all = [], []
    for i in range(len(obs.T)):
        hr_all.append(0)
        far_all.append(0)
    result.append([hr_all, far_all])
    
    # different percentile hr & far
    for per in per_list:
        # calculate multiple varidatoin events
        # len=1728
        hr_all, far_all = [], []
        # calculate roc
        for px in range(len(obs.T)):
            hr_n, far_n = roc(sim[:, px],
                              obs[:, px],
                              percentile=per)
            hr_all.append(hr_n)
            far_all.append(far_n)
        result.append([hr_all, far_all])
        
    # summerize hr & far
    hr_all, far_all = [], []
    for i in range(len(obs.T)):
        hr_all.append(1)
        far_all.append(1)
    result.append([hr_all, far_all])
    
    # result(11, 2, 1728)
    result = np.array(result)
    
    # calculate auc_all
    auc_all = []
    for px in range(len(obs.T)):
        fpr = result[:, 1, px]
        fpr = np.sort(fpr)
        tpr = result[:, 0, px]
        tpr = np.sort(tpr)
        AUC = auc(fpr, tpr)
        auc_all.append(AUC)
    auc_all = np.array(auc_all)
    
    # calculate 95% intervals
    n = len(auc_all)
    sample_mean = np.mean(auc_all)
    sample_var = stats.tvar(auc_all)
    interval = stats.norm.interval(alpha=0.95,
                                   loc=sample_mean,
                                   scale=np.sqrt(sample_var/n))
    print(f"auc_95%reliable_mean {sample_mean} spans {interval}")
    
    # hr_mean(11, far_mean(11)
    hr_mean = np.mean(result[:, 0, :], axis=1)
    far_mean = np.mean(result[:, 1, :], axis=1)
    result_mean = np.array([hr_mean, far_mean])
    result_mean = result_mean.T #(11, 2)
    
    # mean_auc
    fpr = result_mean[:, 1]
    tpr = result_mean[:, 0]
    mean_auc = auc(fpr, tpr)
    
    return result, result_mean, auc_all, mean_auc

def draw_roc_curve(roc_array, auc_array, name_list, color_list):
	plt.rcParams["font.size"] = 18
	fig, ax = plt.subplots(figsize=(8, 8))
	for i in range(len(name_list)):
		plt.plot(roc_array[i, :, 1], 
				roc_array[i, :, 0], 
				label=f"{name_list[i]} (AUC = {np.round(auc_array[i], 2)})",
				color=color_list[i],
				linestyle=":",
				linewidth=2,
				)
		plt.scatter(roc_array[i, :, 1], 
					roc_array[i, :, 0],
					s=100,
					color=color_list[i])
	plt.axis("square")
	plt.xlabel("FPR[False Positive Rate]")
	plt.ylabel("TPR[True Positive Rate]")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()
