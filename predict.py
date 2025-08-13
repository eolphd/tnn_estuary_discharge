import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import backend as K
from pandas import read_csv

test_MSE_metric = tf.keras.metrics.MeanSquaredError()

def loss(y_real, y_hat):
    return tf.reduce_mean(tf.square(y_real - y_hat))

# negloglik = lambda y, rv_y: -rv_y.log_prob(y)

def negloglik(y, rv_y):
    # lambda_reg = 50
    return -rv_y.log_prob(y) 

def create_causal_mask(seq_length):
    """
    Create a causal mask for self-attention in the decoder.
    """
    mask = tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)  # Lower triangular matrix
    return mask  # Shape: (seq_length, seq_length)

# @tf.function
def test_step(x_batch_dev, tar_inp_dev, tar_real_dev, transformer, steps):
    # Run the forward pass of the model to get the probabilistic output
    
    batch, seq_length, d_model = tf.shape(tar_inp_dev)

    causal_mask = create_causal_mask(seq_length)
    causal_mask = tf.reshape(causal_mask, (1, 1, seq_length, seq_length))  # Shape: (1, 1, seq_length, seq_length)
    causal_mask = tf.cast(causal_mask, dtype=tf.float32)  # Convert to float
    
    test_output_dist = transformer((x_batch_dev, tar_inp_dev), True, False, look_ahead_mask=causal_mask, padding_mask=False, steps_for=steps)
    
    loss_value_test = negloglik(tf.cast(tar_real_dev, dtype=tf.float32), test_output_dist)
    # Update validation metric using the mean of the output distribution
    test_MSE_metric.update_state(tar_real_dev, test_output_dist.mean())

    return test_output_dist.mean(),test_output_dist.stddev()#, loss_value_test

def test(transformer, test_dataset, steps, num_target_features):
    output1 = np.empty((1,steps,num_target_features))
    output1_std = np.empty((1,steps,num_target_features))
    for (x_batch_test, tar_inp_test, tar_real_test) in test_dataset:

        test_output, test_output_std = test_step(x_batch_test, tar_inp_test, tar_real_test, transformer,steps)
        test_np = test_output.numpy()
        test_std_np = test_output_std.numpy()
        output1 = np.append(output1, test_np, axis=0)
        output1_std = np.append(output1_std, test_std_np, axis=0)
        
        test_MSE = test_MSE_metric.result()
        test_MSE_metric.reset_states()
        
    output = output1[1:,:,:]
    output_std = output1_std[1:,:,:]
    return output, output_std, test_MSE
