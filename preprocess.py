import numpy as np
import pandas as pd
import utils
import tensorflow as tf
import sklearn
from sklearn.preprocessing import StandardScaler
import joblib
from joblib import dump, load

def save_list_to_txt(data_list, file_path):
    with open(file_path, 'w') as file:
        json.dump(data_list, file)
    print(f"List saved to {file_path}")

def preprocess(Xnp, Xnp_b, Y_real, Ynp, steps_for, steps_back_gfs, steps_back, n_test, batch_size, eval_ratio, num_target_features):
    """Truncate time series into (samples, time steps, features)"""
    
    X_in, X_inb, Y_in, Y_out, Y_real_ = utils.truncate(
    Xnp, 
    Y_real,
    Ynp,
    steps_back_gfs = steps_back_gfs,  
    steps_back = steps_back,                                       
    steps_for = steps_for
    )

    X_in = X_in.transpose((0,1,-1,2,3))
    X_in1 = X_in.reshape(X_in.shape[0], X_in.shape[1] * X_in.shape[2], X_in.shape[3] * X_in.shape[-1])
        
    test_X_in = X_in1[-n_test:, :, :]
    test_Y_in = Y_in[-n_test:, :, :]
    test_Y_out = Y_out[-n_test:, -steps_for:, :num_target_features]
    test_Y_out_inv = test_Y_out.copy()
    test_Y_out_real = Y_real_[-n_test:, -steps_for:, :num_target_features].copy()
    
    X_in_shuffled = X_in1[:-n_test, :, :].copy()
    Y_in_shuffled = Y_in[:-n_test, :, :].copy()
    Y_out_shuffled = Y_out[:-n_test, :, :].copy()
    
    """finally, split the training and validation subsets"""
    """1) mask for inverse indexing the dev subset that will be used to perform K-fold cross-validation"""
    
    n_eval = int(eval_ratio * len(X_in_shuffled))
    train_X_in = X_in_shuffled[:-n_eval, :, :]
    train_Y_in = Y_in_shuffled[:-n_eval, :, :]
    train_Y_out = Y_out_shuffled[:-n_eval, -steps_for:, :num_target_features]
    train_Y_out_inv = train_Y_out.copy()
    dev_X_in = X_in_shuffled[-n_eval:, :, :]
    dev_Y_in = Y_in_shuffled[-n_eval: , :, :]
    dev_Y_out = Y_out_shuffled[-n_eval:, -steps_for:, :num_target_features] 
    
    """2) zero-normalize training, development (validation), and testing subsets using mean and std from training subset. Then reshape the inputs to decoder"""
    
    from joblib import load

    scaler_X = load('scaler_X_s79.bin')
    scaler_Y_in = load('scaler_Y_in_s79.bin')
    scaler_Y_out = load('scaler_Y_out_s79.bin')
    
    train_X_in = scaler_X.fit_transform(train_X_in.reshape(-1, train_X_in.shape[-1])).reshape(train_X_in.shape)
    train_Y_in = scaler_Y_in.fit_transform(train_Y_in.reshape(-1, train_Y_in.shape[-1])).reshape(train_Y_in.shape)
    train_Y_out = scaler_Y_out.fit_transform(train_Y_out.reshape(-1, train_Y_out.shape[-1])).reshape(train_Y_out.shape)

    dev_X_in = scaler_X.transform(dev_X_in.reshape(-1, dev_X_in.shape[-1])).reshape(dev_X_in.shape)
    dev_Y_in = scaler_Y_in.transform(dev_Y_in.reshape(-1, dev_Y_in.shape[-1])).reshape(dev_Y_in.shape)
    dev_Y_out = scaler_Y_out.transform(dev_Y_out.reshape(-1, dev_Y_out.shape[-1])).reshape(dev_Y_out.shape)

    test_X_in = scaler_X.transform(test_X_in.reshape(-1, test_X_in.shape[-1])).reshape(test_X_in.shape)
    test_Y_in = scaler_Y_in.transform(test_Y_in.reshape(-1, test_Y_in.shape[-1])).reshape(test_Y_in.shape)
    test_Y_out = scaler_Y_out.transform(test_Y_out_inv.reshape(-1, test_Y_out_inv.shape[-1])).reshape(test_Y_out_inv.shape)
    
    train_X_in = np.nan_to_num(train_X_in, nan=0)
    dev_X_in = np.nan_to_num(dev_X_in, nan=0)
    test_X_in = np.nan_to_num(test_X_in, nan=0)

       
    train_X = train_X_in
    dev_X= dev_X_in
    test_X = test_X_in
        
    not_include = -2
    train_Y_in[:,-steps_for:, :not_include] = 0
    dev_Y_in[:,-steps_for:, :not_include] = 0
    test_Y_in[:,-steps_for:, :not_include] = 0  
    
    # Prepare the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y_in, train_Y_out))
    train_dataset = train_dataset.shuffle(train_X.shape[0]).batch(batch_size).prefetch(1)

    # prepare the development dataset
    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_X, dev_Y_in, dev_Y_out))
    dev_dataset = dev_dataset.batch(batch_size)

    # # prepare the test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_Y_in, test_Y_out))
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, dev_dataset, test_dataset, train_X, train_Y_in, train_Y_out_inv, dev_X, dev_Y_in, dev_Y_out, test_X, test_Y_in, test_Y_out_inv, test_Y_out, test_Y_out_real, scaler_Y_out, scaler_X, scaler_Y_in 

