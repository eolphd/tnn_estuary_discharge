import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

import datetime
import time

METRIC_MSE='mean_squared_error'
train_MSE_metric = tf.keras.metrics.MeanSquaredError('loss', dtype=tf.float32)
val_MSE_metric = tf.keras.metrics.MeanSquaredError()
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

def exponential_decay_fn(lr0, s, EPOCHS):
    return lr0 * 0.1 ** (EPOCHS / s)

def loss(y_real, y_hat):
    return tf.reduce_mean(tf.square(y_real - y_hat))

# negloglik = lambda y, rv_y: -rv_y.log_prob(y)

def negloglik(y, rv_y):
    # lambda_reg = 50
    return -rv_y.log_prob(y) 

def create_nan_mask(input_tensor):
    """
    Creates an attention mask where NaNs are replaced with 0 (masked) and valid values with 1.
    """
    mask = tf.where(tf.math.is_nan(input_tensor), 0.0, 1.0)  # Replace NaNs with 0, others with 1
    return mask

def create_causal_mask(seq_length):
    """
    Create a causal mask for self-attention in the decoder.
    """
    mask = tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)  # Lower triangular matrix
    return mask  # Shape: (seq_length, seq_length)

# @tf.function
def train_step(x_batch_train, tar_inp, tar_real, transformer, optimizer, steps):
    # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:
        
        # Run the forward pass of the layer. The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
        batch, seq_length, d_model = tf.shape(tar_inp)
        
        output_dist = transformer((x_batch_train, tar_inp), True, False, look_ahead_mask=True, padding_mask=False, steps_for=steps)  # output is now a distribution

        loss_value = negloglik(tf.cast(tar_real, dtype=tf.float32), output_dist)

    # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, transformer.trainable_weights)
    
    # Run one step of gradient descent by updating the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, transformer.trainable_weights))
    
    # Update training metric using the mean of the output distribution.
    train_MSE_metric.update_state(tar_real, output_dist.mean())
    
    return loss_value#, attention_weights_train

# @tf.function
def test_step(x_batch_dev, tar_inp_dev, tar_real_dev, transformer, steps):
    # Run the forward pass of the model to get the probabilistic output
    
    batch, seq_length, d_model = tf.shape(tar_inp_dev)
    
    test_output_dist = transformer((x_batch_dev, tar_inp_dev), False, False, look_ahead_mask=True, padding_mask=False, steps_for=steps)
    
    # Compute the loss value for this minibatch using negative log-likelihood
    loss_value_dev = negloglik(tf.cast(tar_real_dev, dtype=tf.float32), test_output_dist)

    # Update validation metric using the mean of the output distribution
    val_MSE_metric.update_state(tar_real_dev, test_output_dist.mean())

    return test_output_dist.mean(), loss_value_dev


def train_val(transformer, num_layers, emb_dim_enc, emb_dim_dec, num_heads, dff, train_X, train_y, dropout_rate, EPOCHS, train_dataset, dev_dataset, steps, num_target_features, LR0):

    start_time = time.time()

    for epoch in range(EPOCHS):

        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            LR0,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        
        # Training
        for step, (x_batch_train, tar_inp, tar_real) in enumerate(train_dataset):
            
            loss_value = train_step(x_batch_train, tar_inp, tar_real, transformer, optimizer, steps)

        # # Display metrics
        # train_MSE = train_MSE_metric.result()
        # with train_summary_writer.as_default():
        #     tf.summary.scalar('loss', train_MSE, step=epoch)

        # Display metrics at the end of each epoch.
        train_MSE = train_MSE_metric.result()
        print("\nEpoch %d" % (epoch + 1,))
        print("Training MSE: %.4f" % (float(train_MSE),))
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_MSE_metric.result(), step=epoch)
            
        train_MSE_metric.reset_states()

        # Validation
        for (x_batch_dev, tar_inp_dev, tar_real_dev) in dev_dataset:
            dev_output_dist, loss_value_dev = test_step(x_batch_dev, tar_inp_dev, tar_real_dev, transformer, steps)

        val_MSE = val_MSE_metric.result()
        val_MSE_metric.reset_states()
        print("Val MSE: %.4f" % (float(val_MSE),))
        if epoch + 1 == EPOCHS:
            print("\nEpoch %d" % (epoch + 1,))
            print("Training MSE: %.4f" % (float(train_MSE),))
            print("Validation MSE: %.4f" % (float(val_MSE),))
            minutes, seconds = divmod(time.time() - start_time, 60)
            print("Time taken: %.2f min %.2f s" % (minutes, seconds))
            
    return train_MSE, val_MSE