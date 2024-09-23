### This file contains two functions for running adaptive conformal inference in order to reproduce Figures 1, 2, 4, 5, 6, and 7 in https://arxiv.org/abs/2106.00170. 

import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
import torch
import argparse
from pathlib import Path
import numpy as np
from src.model_utils import build_model
import src.datasets as datasets
from src.torch_utils import torch2numpy
from reevaluate import get_test_dataset
from tqdm import tqdm
#from arch import arch_model

### Main method for forming election night predictions of county vote totals as in Figure 2

def ACI(Y, X, alpha, gamma, t_init=500, split_size=0.75, update_method="Simple", momentum_bw=0.95, max_band = 10):
    T = len(Y)
    # Initialize data storage variables
    alpha_trajectory = [alpha] * (T - t_init)
    adapt_err_seq = [0] * (T - t_init)
    no_adapt_error_seq = [0] * (T - t_init)
    alpha_t = alpha
    band_native = []
    band_adapt = []
    
    for t in range(t_init, T):
        # Split data into training and calibration set
        train_points = np.random.choice(t, size=int(split_size*t), replace=False)
        cal_points = np.setdiff1d(np.arange(t), train_points)
        X_train, Y_train = X[train_points], Y[train_points]
        X_cal, Y_cal = X[cal_points], Y[cal_points]
        
        # Fit quantile regression on training setting
        model_upper = QuantReg(Y_train, X_train)
        model_lower = QuantReg(Y_train, X_train)
        res_upper = model_upper.fit(q=1-alpha/2)
        res_lower = model_lower.fit(q=alpha/2)
        
        # Compute conformity score on calibration set and on new data example
        pred_low_for_cal = res_lower.predict(X_cal)
        pred_up_for_cal = res_upper.predict(X_cal)
        scores = np.maximum(Y_cal - pred_up_for_cal, pred_low_for_cal - Y_cal)
        q_up = res_upper.predict(X[t].reshape(1, -1))[0]
        q_low = res_lower.predict(X[t].reshape(1, -1))[0]
        new_score = max(Y[t] - q_up, q_low - Y[t])
        
        # Compute errt for both methods
        conf_quant_naive = np.quantile(scores, 1-alpha)
        no_adapt_error_seq[t-t_init] = float(conf_quant_naive < new_score)
        band_native.append(conf_quant_naive)
        
        if alpha_t >= 1:
            adapt_err_seq[t-t_init] = 1
            band_adapt.append(0)
        elif alpha_t <= 0:
            adapt_err_seq[t-t_init] = 0
            band_adapt.append(max_band)
        else:
            conf_quant_adapt = np.quantile(scores, 1-alpha_t)
            adapt_err_seq[t-t_init] = float(conf_quant_adapt < new_score)
            band_adapt.append(conf_quant_adapt)
        # update alpha_t
        alpha_trajectory[t-t_init] = alpha_t
        if update_method == "Simple":
            alpha_t += gamma * (alpha - adapt_err_seq[t-t_init])
        elif update_method == "Momentum":
            w = momentum_bw ** np.arange(t-t_init+1)[::-1]
            w /= w.sum()
            alpha_t += gamma * (alpha - np.sum(adapt_err_seq[:t-t_init+1] * w))
        
        # if t % 100 == 0:
        #     print(f"Done {t} time steps")
    
    return alpha_trajectory, adapt_err_seq, no_adapt_error_seq, (band_native, band_adapt)

def create_x_y(dataset_path, context_length=10):

    #test_dataset = get_test_dataset(config)
    test_dataset = datasets.BouncingBallDataset(path=dataset_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20)

    Xs = []
    Ys = []
    # calculate coverage and width of prediction intervals
    for test_batch, test_label in tqdm(test_loader):
       #print(test_batch.shape)
        prediction_length = test_batch.shape[1]-context_length

        X = []
        Y = []
        for t in range(prediction_length):
            X.append(test_batch[:, t:t+context_length].numpy())
            Y.append(test_label[:, t+context_length].numpy())
        
        Xs.append(np.array(X).swapaxes(0, 1).squeeze())
        Ys.append(np.array(Y).swapaxes(0, 1))

    return np.concatenate(Xs, 0), np.concatenate(Ys, 0)

def create_x_y_real(data, context_length=10):
    Xs = []
    Ys = []
    for data_entry in data:
        X = []
        Y = []
        for t in range(len(data_entry)-context_length):
            X.append(data_entry[t:t+context_length])
            Y.append(data_entry[t+context_length])
        Xs.append(np.array(X))
        Ys.append(np.array(Y))
    return np.array(Xs), np.array(Ys)

def ACI_from_dataset(dataset_path, context_length=10, test_size=100, t_init=100):
    Xs, Ys = create_x_y(dataset_path, context_length=context_length)
    print(Xs.shape, Ys.shape)

    dataset_name = dataset_path.split("/")[-1].split(".")[0]
    alpha_trajectories = []
    adapt_err_seqs = []
    no_adapt_error_seqs = []
    band_natives = []
    band_adapts = []

    if test_size <= 0:
        test_size = Xs.shape[0]

    for i in tqdm(range(test_size)):
        X, Y = Xs[i], Ys[i]
        alpha_trajectory, adapt_err_seq, no_adapt_error_seq, (band_native, band_adapt) = ACI(Y, X, alpha=0.1, gamma=0.01, t_init=t_init)
        alpha_trajectories.append(alpha_trajectory)
        adapt_err_seqs.append(adapt_err_seq)
        no_adapt_error_seqs.append(no_adapt_error_seq)
        band_natives.append(band_native)
        band_adapts.append(band_adapt)

    np.savez(f"./results/{dataset_name}_aci.npz", 
             alpha_trajectories=alpha_trajectories, 
             adapt_err_seqs=adapt_err_seqs, 
             no_adapt_error_seqs=no_adapt_error_seqs, 
             band_natives=band_natives, 
             band_adapts=band_adapts)
    
    cov_adapt, cov_no_adapt = 1-np.mean(adapt_err_seqs), 1-np.mean(no_adapt_error_seqs)
    std_adapt, std_no_adapt = np.std(np.mean(adapt_err_seqs, axis=1)), np.std(np.mean(no_adapt_error_seqs, axis=1))
    mean_band_native, std_band_native = np.mean(band_natives), np.std(np.mean(band_natives, axis=1))
    mean_band_adapt, std_band_adapt = np.mean(band_adapts), np.std(np.mean(band_adapts, axis=1))
    
    print(f"Adaptive Coverage: {cov_adapt} +/- {std_adapt}")
    print(f"No Adaptive Coverage: {cov_no_adapt} +/- {std_no_adapt}")
    print(f"Adaptive Width: {mean_band_adapt} +/- {std_band_adapt}")
    print(f"No Adaptive Width: {mean_band_native} +/- {std_band_native}")
    return cov_adapt, std_adapt, cov_no_adapt, std_no_adapt, mean_band_native, std_band_native, mean_band_adapt, std_band_adapt