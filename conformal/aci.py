### This file contains two functions for running adaptive conformal inference in order to reproduce Figures 1, 2, 4, 5, 6, and 7 in https://arxiv.org/abs/2106.00170. 

import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from arch import arch_model

### Main method for forming election night predictions of county vote totals as in Figure 2
def run_election_night_pred(Y, X, alpha, gamma, t_init=500, split_size=0.75, update_method="Simple", momentum_bw=0.95):
    T = len(Y)
    # Initialize data storage variables
    alpha_trajectory = [alpha] * (T - t_init)
    adapt_err_seq = [0] * (T - t_init)
    no_adapt_error_seq = [0] * (T - t_init)
    alpha_t = alpha
    
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
        
        if alpha_t >= 1:
            adapt_err_seq[t-t_init] = 1
        elif alpha_t <= 0:
            adapt_err_seq[t-t_init] = 0
        else:
            conf_quant_adapt = np.quantile(scores, 1-alpha_t)
            adapt_err_seq[t-t_init] = float(conf_quant_adapt < new_score)
        
        # update alpha_t
        alpha_trajectory[t-t_init] = alpha_t
        if update_method == "Simple":
            alpha_t += gamma * (alpha - adapt_err_seq[t-t_init])
        elif update_method == "Momentum":
            w = momentum_bw ** np.arange(t-t_init+1)[::-1]
            w /= w.sum()
            alpha_t += gamma * (alpha - np.sum(adapt_err_seq[:t-t_init+1] * w))
        
        if t % 100 == 0:
            print(f"Done {t} time steps")
    
    return alpha_trajectory, adapt_err_seq, no_adapt_error_seq

### Main method for forming volatility predictions as in Figure 1
def garch_conformal_forecasting(returns, alpha, gamma, lookback=1250, garch_p=1, garch_q=1, start_up=100, verbose=False, update_method="Simple", momentum_bw=0.95):
    T = len(returns)
    start_up = max(start_up, lookback)
    garch_spec = arch_model(returns, vol='Garch', p=garch_p, q=garch_q)
    alpha_t = alpha
    # Initialize data storage variables
    err_seq_oc = [0] * (T - start_up + 1)
    err_seq_nc = [0] * (T - start_up + 1)
    alpha_sequence = [alpha] * (T - start_up + 1)
    scores = [0] * (T - start_up + 1)
    
    for t in range(start_up, T):
        if verbose:
            print(t)
        # Fit garch model and compute new conformity score
        garch_fit = garch_spec.fit(last_obs=t-1, disp='off')
        sigma_next = garch_fit.forecast(horizon=1).variance.values[-1][0]
        scores[t-start_up] = abs(returns[t]**2 - sigma_next**2) / sigma_next**2
        
        recent_scores = scores[max(0, t-start_up-lookback+1):t-start_up]
        
        # compute errt for both methods
        err_seq_oc[t-start_up] = float(scores[t-start_up] > np.quantile(recent_scores, 1-alpha_t))
        err_seq_nc[t-start_up] = float(scores[t-start_up] > np.quantile(recent_scores, 1-alpha))
        
        # update alpha_t
        alpha_sequence[t-start_up] = alpha_t
        if update_method == "Simple":
            alpha_t += gamma * (alpha - err_seq_oc[t-start_up])
        elif update_method == "Momentum":
            w = momentum_bw ** np.arange(t-start_up+1)[::-1]
            w /= w.sum()
            alpha_t += gamma * (alpha - np.sum(err_seq_oc[:t-start_up+1] * w))
        
        if t % 100 == 0:
            print(f"Done {t} steps")
    
    return alpha_sequence, err_seq_oc, err_seq_nc