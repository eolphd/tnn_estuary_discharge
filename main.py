#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, glob, pathlib, re, sys, time, math, datetime
import re
import time
import numpy as np
import pandas as pd
from pandas import read_csv, DataFrame, concat
from sklearn.metrics import r2_score, mean_squared_error
import scipy as sp
import scipy.stats as st
from scipy import stats
import matplotlib.pyplot as plt
import hydroeval as he
import statistics
import tensorflow as tf
from tensorflow.keras import backend as K
# to execute tf in eager mode
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)
# Model Modules
import model as model
import utils as utils
import preprocess as preprocess
import fit as fit
import predict as predict

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

METRIC_MSE='mean_squared_error'

train_MSE_metric = tf.keras.metrics.MeanSquaredError('loss', dtype=tf.float32)
val_MSE_metric = tf.keras.metrics.MeanSquaredError()
test_MSE_metric = tf.keras.metrics.MeanSquaredError()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# ARCHITECTURAL VARIABLES AND HYPERPARAMETERS 
steps_for = 10 # Future time steps_for (target variables and weather forecasts)
num_target_features = 1
n_test = 365 # Reserve last n_test days for testing
remove_last_days = -220
station = 's79'
conversion_factor = 0.02831683199881 # cfs to cms

csv_path_targets = 'QH_SLEW_imp.csv'
csv_path_sgate = 'S79_forecasts.csv'
csv_xgboost = 'Q_s79_xgboost.csv'
PATH = r'/blue/olabarrieta/eorozcolopez/proj/'
PATH_DATA = os.path.join(PATH, 'data/')
PATH_RESULTS = os.path.join(PATH, 'results/')
PATH_WEIGHTS = os.path.join(PATH, 'weights/')

### xgboost ###
xg = read_csv(PATH_DATA + csv_xgboost).iloc[:,1:]

#####################
### Forecasts
sgate_df = read_csv(PATH_DATA + csv_path_sgate)
sgate_df = sgate_df.iloc[:,1:2]
Xnp_t = sgate_df.to_numpy()

Xnp_0 = Xnp_t[::10,:].copy()
Xnp_1 = Xnp_t[1::10,:].copy()
Xnp_2 = Xnp_t[2::10,:].copy()
Xnp_3 = Xnp_t[3::10,:].copy()
Xnp_4 = Xnp_t[4::10,:].copy()
Xnp_5 = Xnp_t[5::10,:].copy()
Xnp_6 = Xnp_t[6::10,:].copy()
Xnp_7 = Xnp_t[7::10,:].copy()
Xnp_8 = Xnp_t[8::10,:].copy()
Xnp_9 = Xnp_t[9::10,:].copy()

Xnp_sgate = np.concatenate((Xnp_0,Xnp_1,Xnp_2,Xnp_3,Xnp_4,Xnp_5,Xnp_6,Xnp_7,Xnp_8,Xnp_9), axis=1)
Xnp_sgate = Xnp_sgate[remove_last_days-n_test-steps_for+1:remove_last_days-steps_for+1,:]# * conversion_factor
NWM_df = pd.DataFrame(Xnp_sgate)

######################

### WEATHER FORECASTS ###

Xnp = np.load(PATH_DATA + 'PRATE_FL_6h_00_20150115_20250105.npy')
Xnp1 = np.reshape(Xnp, (-1, Xnp.shape[1], Xnp.shape[2], 4))
Xnp1 = np.mean(Xnp1, axis=-1)
Xnp = np.reshape(Xnp1, (Xnp.shape[0], Xnp.shape[1], Xnp.shape[2], -1))
Xnp_past = Xnp.copy()

#######################

# HYPERPARAMETERS #
eval_ratio = [0.1,0.2,0.15,0.1,0.15,0.1,0.1]# Ratio to split validation subset 
steps_back_gfs = [0,0,0,0,0]
steps_back =[1,1,1,1,1]# sequence length of past hydrology input to the decoder
EPOCHS = [70,60,85,70,65]#
num_layers = [4,4,1,2,2]
num_heads = [8,1,2,16,2]
FNN = [2048,1024,1024,1024,512]
dropout_rate=[0.07,0.02,0.05,0.05,0.1]
batch_size = [64,64,32,16,16]
emb_dim = [1024,1024,1024,512,256]
LR0 = [1e-5,5e-6,5e-6,3e-6,3e-5]
CKPT = [1,2,18,29]

output_list = []
output_list_val =  []
alea_val_list = []
alea_var_list = []
epis_mc_std_list = []
lb = []
ub = []
std = []
std_val = []
preds_std = []
for i in range(len(CKPT)):

    ckpt_s = CKPT[i]

    Y_df = read_csv(PATH_DATA + csv_path_targets)
    Ydf = Y_df.copy().iloc[:,:]

    datestr = -n_test+remove_last_days
    dateend = remove_last_days
    Y_df["date"] = pd.to_datetime(Y_df["date"], format="%m/%d/%Y")
    TEMP = Y_df.iloc[datestr:dateend,:].copy()
    dates_test = pd.to_datetime(TEMP.loc[:, 'date'], format="%d-%m-%Y")  # Force D-M-Y format

    # Y_real for testing performance evaluation
    Y_real = Y_df.loc[:,'Q_02292900'].copy().to_numpy()
    Y_real = np.reshape(Y_real, (-1,1))        

    # Time feature engineering
    datadate = Y_df['date'] =  pd.to_datetime(Y_df['date'], format='%m/%d/%Y')
    Y_df['num'] = np.array(range(len(Y_df)))
    Y_df['day'] = datadate.dt.day
    Y_df['month'] = datadate.dt.month
    Y_df['year'] = datadate.dt.year
    Y_df = utils.timestamp(Y_df, 'month', 12)
    Y_df= utils.timestamp(Y_df, 'day', 30.4375)
    Ydf = Y_df.copy().iloc[:,1:]

    last_column = Ydf.pop('Q_02276877') 
    Ydf.insert(len(Ydf.columns), 'Q_02276877', last_column) 
    last_column = Ydf.pop('Q_02292010') 
    Ydf.insert(len(Ydf.columns), 'Q_02292010', last_column) 
    second_column = Ydf.pop('Q_02292900')
    Ydf.insert(0, 'Q_02292900', second_column) 

    Ynp = Ydf.iloc[:remove_last_days,:].to_numpy()

    # data preprocessing
    train_dataset, dev_dataset, test_dataset, train_X, train_Y_in, train_Y_out_inv, dev_X, dev_Y_in, dev_Y_out, test_X, test_Y_in, test_Y_out_inv, test_Y_out, test_Y_out_real, scaler_Y_out, scaler_X, scaler_Y_in  =  preprocess.preprocess( 
                Xnp,
                Xnp_past,
                Y_real,
                Ynp,
                steps_for, 
                steps_back_gfs[i], 
                steps_back[i], 
                n_test, 
                batch_size[i], 
                eval_ratio[i], 
                num_target_features
                )

    # transformer model
    transformer = model.Transformer(
                num_layers = num_layers[i],
                d_model = emb_dim[i],
                num_heads = num_heads[i],
                FNN = FNN[i],
                num_patches_inp = train_X.shape[1],
                num_patches_tar = train_Y_in.shape[1],
                features = num_target_features, #X_out.shape[-1],
                rate = dropout_rate[i],
                steps_for = steps_for
    )
    ### lOAD CHECKPOINT ###
    checkpoint_path = PATH_WEIGHTS + '/S79/saved/ckpt_%s_%s-1' %(station, ckpt_s)

    ckpt = tf.train.Checkpoint(transformer=transformer)#,
    # Restore the checkpointed values to the `model` object.
    ckpt.restore(checkpoint_path)
    
    
    ### Comment if NOT training
    # train_MSE, val_MSE  = fit.train_val(transformer, #train_MSE, val_MSE, y_hat_val, tar_real_val 
    #                          num_layers[i], 
    #                          emb_dim[i], 
    #                          emb_dim[i], 
    #                          num_heads[i], 
    #                          FNN[i], 
    #                          train_X, 
    #                          train_Y_out, 
    #                          dropout_rate[i], 
    #                          EPOCHS[i], 
    #                          train_dataset, 
    #                          dev_dataset,
    #                          steps_for,
    #                          num_target_features,
    #                         # batch_size,
    #                        LR0[i]
    #                    )

    ### TESTING ###

    def mc_inference(x, num_samples=100):
        stds_lb = []
        stds_ub = []
        preds = []
        preds_std = []

        for _ in range(num_samples):
            test_output, test_output_std, test_MSE = predict.test(transformer, x, steps_for, num_target_features) 
            preds.append(test_output)
            preds_std.append(test_output_std)
            
        preds = np.array(preds)
        preds_std = np.array(preds_std)
        alea_std = (preds_std.mean(axis=0))
        epis_mc_std = preds.var(axis=0)
        test_output = preds.mean(axis=0)
        test_output = scaler_Y_out.inverse_transform(test_output.reshape(-1, test_output.shape[-1])).reshape(test_output.shape)
        return test_output, alea_std, epis_mc_std

    mean_pred, alea_var, epis_mc_std = mc_inference(test_dataset, num_samples=20)

    output_list.append(mean_pred)
    alea_var_list.append(alea_var)
    epis_mc_std_list.append(epis_mc_std)

mean_pred = np.mean(output_list, axis=0)
std_pred = np.std(output_list, axis=0)
epis_mc_std = np.sqrt(np.mean(epis_mc_std_list, axis=0)) * (np.std(train_Y_out_inv))
epis_std = np.std(output_list, axis=0)
alea_std = np.mean(alea_var_list, axis=0) * (np.std(train_Y_out_inv))

epis_mc_std = [np.sqrt(a_i) * np.std(train_Y_out_inv) for a_i in (epis_mc_std_list)]
epis_mc_std_lb = [a_i - 1.96*b_i for a_i, b_i in zip(output_list, epis_mc_std)]
epis_mc_std_ub = [a_i + 1.96*b_i for a_i, b_i in zip(output_list, epis_mc_std)]
lower_bound = np.mean(epis_mc_std_lb, axis=0)
upper_bound = np.mean(epis_mc_std_ub, axis=0)

test_output = np.reshape(mean_pred, (mean_pred.shape[0], -1))
test_output = pd.DataFrame(test_output)
test_Y_out = np.reshape(test_Y_out_inv, (test_Y_out_inv.shape[0], -1))
test_Y_out = pd.DataFrame(test_Y_out)
lower_bound = np.reshape(lower_bound, (lower_bound.shape[0], -1))
upper_bound = np.reshape(upper_bound, (upper_bound.shape[0], -1))
lower_bound = pd.DataFrame(lower_bound)
upper_bound = pd.DataFrame(upper_bound)

test_Y_out_real = np.squeeze(test_Y_out_real, axis=-1)
test_Y_out_real = pd.DataFrame(test_Y_out_real)
# save simulated, observed, and model parameters
test_Y_out.to_csv(PATH_RESULTS+'_'+'y_test_%s_ckpt_%s.csv' %(station, ckpt_s))
test_Y_out_real.to_csv(PATH_RESULTS+'_'+'y_test_real_%s_ckpt_%s.csv' %(station, ckpt_s))
test_output.to_csv(PATH_RESULTS+'_'+'test_hat_%s_ckpt_%s.csv' %(station, ckpt_s))
lower_bound.to_csv(PATH_RESULTS+'_'+'test_hat_std_lb_%s_ckpt_%s.csv' %(station, ckpt_s))
upper_bound.to_csv(PATH_RESULTS+'_'+'test_hat_std_ub_%s_ckpt_%s.csv' %(station, ckpt_s))

# ======================================================================================================================================= #
## VISUALIZATION

import plotly.express as px
import plotly.graph_objects as go
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
dark_style = {
    'figure.facecolor': 'white',#'#212946',
    'axes.facecolor': 'white',#'#212946',
    'savefig.facecolor':'#212946',
    'axes.grid': True,
    'axes.grid.which': 'both',
    'axes.spines.left': True,
    'axes.spines.right': True,
    'axes.spines.top': True,
    'axes.spines.bottom': True,
    'grid.color': '#DCDCDC',#'#2A3459',
    'grid.linewidth': '1',
    'text.color': '0.2',
    'axes.labelcolor': '0.2',
    'xtick.color': '0.2',
    'ytick.color': '0.2',
    'font.size': 12
}
plt.rcParams.update(dark_style)

# numpy array to store all simulations
pred_all = np.zeros(shape=(n_test, 1))
pred_all_std = np.zeros(shape=(n_test, 1))

for k in range(num_target_features):
    pred_temp = np.zeros(shape=(n_test, steps_for))
    data_temp = np.zeros(shape=(n_test, steps_for))
    # print(data_temp.shape)
    for i in range(steps_for):
        pred = pd.read_csv(PATH_RESULTS+'_'+'test_hat_%s_ckpt_%s.csv' %(station, ckpt_s)).iloc[:,1:]*conversion_factor #(cfs to cms) 
        pred_all = (pred.iloc[:,k*steps_for:k*steps_for+steps_for])
        pred_df = pd.DataFrame(pred_all.iloc[:,i:i+1])

        data = pd.read_csv(PATH_RESULTS+'_'+'y_test_%s_ckpt_%s.csv' %(station, ckpt_s)).iloc[:,1:]*conversion_factor
        data = data.iloc[:,k*steps_for: k*steps_for+steps_for]
        data = pd.DataFrame(data.iloc[:,i])

        data_real = pd.read_csv(PATH_RESULTS+'_'+'y_test_real_%s_ckpt_%s.csv' %(station, ckpt_s)).iloc[:,1:]*conversion_factor
        data_real = data_real.iloc[:,k*steps_for: k*steps_for+steps_for]
        data_real = pd.DataFrame(data_real.iloc[:,i])

        pred_std = pd.read_csv(PATH_RESULTS+'_'+'test_hat_std_lb_%s_ckpt_%s.csv' %(station, ckpt_s)).iloc[:,1:]*conversion_factor #(cfs to cms) 
        pred_all_std_lb = (pred_std.iloc[:,k*steps_for:k*steps_for+steps_for])
        pred_df_std_lb = pd.DataFrame(pred_all_std_lb.iloc[:,i:i+1])
        pred_std = pd.read_csv(PATH_RESULTS+'_'+'test_hat_std_ub_%s_ckpt_%s.csv' %(station, ckpt_s)).iloc[:,1:]*conversion_factor #(cfs to cms) 
        pred_all_std_ub = (pred_std.iloc[:,k*steps_for:k*steps_for+steps_for])
        pred_df_std_ub = pd.DataFrame(pred_all_std_ub.iloc[:,i:i+1])

        lb = np.squeeze(pred_df_std_lb.to_numpy())
        lb[lb<0]=0
        ub = np.squeeze(pred_df_std_ub.to_numpy())
        # print('lb',lb)

        bounds = pd.DataFrame([lb, ub]).T
        columns = ("lb ub").split()
        bounds.columns = columns

        kge, r, a, beta = he.evaluator(he.kge, pred_df.iloc[:,0], data.iloc[:,0])
        print('a', a)
        print('r', r)
        print('beta', beta)

        #performance metrics
        NSE1 = 1 - np.sum((pred_df.iloc[:,0] - data.iloc[:,0] )**2) /  np.sum((data.iloc[:,0]  - np.mean(data.iloc[:,0] ))**2)
        RMSE1 = math.sqrt(mean_squared_error(data.iloc[:, 0].to_numpy(), pred_df.iloc[:,0]))
        R1 = r2_score(pred_df.iloc[:,0], data.iloc[:, 0].to_numpy())
        
        NSE_NWM = 1 - np.sum((NWM_df.iloc[:,i] - data.iloc[:,0] )**2) /  np.sum((data.iloc[:,0]  - np.mean(data.iloc[:,0] ))**2)
        RMSE_NWM = math.sqrt(mean_squared_error(data.iloc[:, 0].to_numpy(), NWM_df.iloc[:,i]))
        R_NWM = r2_score(NWM_df.iloc[:,i], data.iloc[:, 0].to_numpy())
        # pbias = (np.sum(data.iloc[:, 0] - pred_all.iloc[:,-1]))*100/np.sum(data.iloc[:, 0])
        print('NSE_NWM',NSE_NWM)
        print('RMSE_NWM', RMSE_NWM)
        print('R_NWM', R_NWM)

        pred_in = 0
        pred_out = 0

        for j in range(len(data)):
            if (data.iloc[j,0] > bounds.iloc[j,0]) & (data.iloc[j,0] < bounds.iloc[j,-1]):
                pred_in += 1
            else:
                pred_out += 1
        coverage = pred_in/(pred_out+pred_in)
        print('pred_in:', pred_in)
        print('pred_out:', pred_out)
        print('coverage:', coverage)

        # figure
        import matplotlib.dates as mdates
        # figure
        fig, ax=plt.subplots(figsize=(12, 3))
        ax.plot(dates_test, data_real.iloc[:,0], label='Observed', linestyle = '-', linewidth= '1.5', color='black')
        ax.plot(dates_test, NWM_df.iloc[:,i], label='NWM', linestyle = (0, (3, 1, 1, 1, 1, 1)), linewidth= '1.5', color='goldenrod', alpha=1)
        ax.plot(dates_test, xg.iloc[:,i], label='XGBOOST', linestyle = '-.', linewidth= '1.5', color='deepskyblue', alpha=1)
        ax.plot(dates_test, pred_df.iloc[:,0] ,label='TNN', linestyle = '--', linewidth = '1.5', c='red')
        
        # Format the x-axis to display dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))  # Format as "YYYY-MM-DD"
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Major ticks every 2 months
        # ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))  # Minor ticks every month
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")  # Rotate date labels
        # ax.minorticks_on()
        ax.set_ylabel('Streamflow $(m^3 s^{-1})$', fontsize = 16)
        ax.set_ylim(-30, 400)
        ax.grid(False)
        fig.legend(bbox_to_anchor=(0.79,0.92), ncol=4, fontsize = 14)
        fig.tight_layout()
        plt.savefig('figure_S79_%i.png' %(i), bbox_inches='tight', facecolor='w', dpi=300)
        plt.show()

# numpy array to store all simulations
pred_np = np.zeros(shape=(n_test, 1))

ncol = 2
nrow = 2#(num_target_features)//ncol+1
fig, ax = plt.subplots(nrow, ncol, figsize = (20,15))

idx1 = 0
idx2 = 0
# load predictions and data
for k in range(num_target_features):
    pred_temp = np.zeros(shape=(n_test, steps_for))
    data_temp = np.zeros(shape=(n_test, steps_for))

    scatter_list = []
    steps_for_list = []
    NSE = []
    RMSE = []
    R = []
    KGE=[]
    NSE_NWM = []
    KGE_NWM = []
    RMSE_NWM = []

    for i in range(steps_for):
        # load and prepare dataframes
        pred = pd.read_csv(PATH_RESULTS+'_'+'test_hat_%s_ckpt_%s.csv' %(station, ckpt_s)).iloc[:,1:]*conversion_factor #(cfs to cms) 
        pred_np[:,:] = (pred.iloc[:,i*num_target_features+k:i*num_target_features+1+k])
        pred_all = pd.DataFrame(pred_np)
        data = pd.read_csv(PATH_RESULTS+'_'+'y_test_%s_ckpt_%s.csv' %(station, ckpt_s)).iloc[:,1:]*conversion_factor
        data_temp[:,:] = data.iloc[:,i*num_target_features+k: i*num_target_features+1+k]
        data = data_temp[:,i]
        data = pd.DataFrame(data)
        kge, r, _, beta = he.evaluator(he.kge, pred_all.iloc[:,0], data.iloc[:,0])

        #performance metrics
        NSE1 = 1 - np.sum((pred_all.iloc[:,0] - data.iloc[:,0] )**2) /  np.sum((data.iloc[:,0]  - np.mean(data.iloc[:,0] ))**2)
        RMSE1 = math.sqrt(mean_squared_error(data.iloc[:, 0].to_numpy(), pred_all.iloc[:,0]))
        R1 = r2_score(pred_all.iloc[:,0], data.iloc[:, 0].to_numpy())

        NSE.append(NSE1)
        RMSE.append(RMSE1)
        R.append(R1)
        KGE.append(kge)
        uni = '$m^3 s^{-1}$'

        if k%2==0:
            ax[idx1,0].scatter(data, pred_all.iloc[:,0],
                        c=colors[i],
                        marker=marker[i],
                        linewidths=linewidth,
                        alpha=alpha, 
                        edgecolor='black'
                                  )
            scatter1 = ax[idx1,0].scatter(data, pred_all.iloc[:,0],
                    c=colors[i],
                    marker=marker[i],
                    linewidths=linewidth,
                    alpha=alpha, 
                    edgecolor='black'
                              )

            scatter_list.append(scatter1)
            steps_for_list.append('t+%i: KGE: %.2f, NSE: %.2f, RMSE: %.2f' %(i+1, kge, NSE1, RMSE1)+uni)

            ax[idx1,0].plot([np.min(data, axis=0), np.max(data, axis=0)*1], [np.min(data, axis=0),np.max(data, axis=0)*1], color='black', linestyle='--', linewidth=1)
            plt.subplots_adjust(left=0.1,
                                bottom=0.1, 
                                right=0.9, 
                                top=0.9, 
                                wspace=1.2, 
                                hspace=0.25)

        else:
            ax[idx2,1].scatter(data, pred_all.iloc[:,0],
                        c=colors[i],
                        marker=marker[i],
                        linewidths=linewidth,
                        alpha=alpha, 
                        edgecolor='black'
                                  )
            scatter1 = ax[idx2,1].scatter(data, pred_all.iloc[:,0],
                    c=colors[i],
                    marker=marker[i],
                    linewidths=linewidth,
                    alpha=alpha, 
                    edgecolor='black'
                              )

            scatter_list.append(scatter1)
            steps_for_list.append('t+%i: KGE: %.2f, NSE: %.2f, RMSE: %.2f' %(i+1, kge, NSE1, RMSE1)+uni)

            ax[idx2,1].plot([np.min(data, axis=0), np.max(data, axis=0)*1], [np.min(data, axis=0),np.max(data, axis=0)*1], color='black', linestyle='--', linewidth=1)
            plt.subplots_adjust(left=0.1,
                                bottom=0.1, 
                                right=0.9, 
                                top=0.9, 
                                wspace=1.2, 
                                hspace=0.25)

    if k%2==0:
        ax[idx1,0].set_xlabel('Observed ' + uni , fontsize=18)
        ax[idx1,0].set_ylabel('Simulated ' + uni , fontsize=18) 
        ax[idx1,0].legend(scatter_list,
                   steps_for_list, 
                    scatterpoints=1,
                   bbox_to_anchor=(1,1.03), ncol=1,fancybox=True, shadow=True, facecolor='white', fontsize=12
                  )
        idx1 += 1

    else:
        ax[idx2,1].set_xlabel('Observed ' + uni , fontsize=18)
        ax[idx2,1].set_ylabel('Simulated ' + uni , fontsize=18) 
        ax[idx2,1].legend(scatter_list,
                   steps_for_list, 
                    scatterpoints=1,
                   bbox_to_anchor=(1,1.03), ncol=1,fancybox=True, shadow=True, facecolor='white', fontsize=12
                  )
        idx2 += 1
plt.savefig('figure_s79_simObs.png', bbox_inches='tight', facecolor='w', dpi=300)
plt.show()

