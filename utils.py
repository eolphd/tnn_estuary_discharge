import numpy as np
import pandas as pd
import scipy.stats as st

def timestamp(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

#=========================================================================================== #

def truncate(x, y_real, y, steps_back_gfs, steps_back, steps_for):
    in1_, in1b_, in2_, out_, out_real_ = [], [], [], [], []
    if steps_back_gfs >= steps_back:
        for i in range(steps_back_gfs, len(y)-steps_for): #(len(x)-tback1-steps_for+1):
            in1_.append(x[i:i+1,  :, :].tolist())
            in1b_.append(x[i-steps_back_gfs:i,  :, :].tolist())
            in2_.append(y[i-steps_back:(i+steps_for), :].tolist())
            out_.append(y[i-steps_back:(i+steps_for), :].tolist())
            out_real_.append(y_real[i-steps_back:(i+steps_for), :].tolist())
    else:
        for i in range(steps_back, len(y)-steps_for): #(len(x)-tback1-steps_for+1):
            in1_.append(x[i:i+1, :, :].tolist())
            in1b_.append(x[i-steps_back_gfs:i, :, :].tolist())
            in2_.append(y[i-steps_back:(i+steps_for), :].tolist())
            out_.append(y[i-steps_back:(i+steps_for), :].tolist())
            out_real_.append(y_real[i-steps_back:(i+steps_for), :].tolist())
       
    return np.array(in1_), np.array(in1b_),  np.array(in2_), np.array(out_), np.array(out_real_)

#=========================================================================================== #

def shuffle(a, b, c):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=42).permutation(len(a))
    # print('lena',len(a))
    return a[p], b[p], c[p]

# =========================================================================================== #