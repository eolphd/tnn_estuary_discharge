from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
# import tf_keras

import tensorflow_probability as tfp

sns.reset_defaults()
sns.set_context(context='talk',font_scale=0.7)


tfd = tfp.distributions

if tf.test.gpu_device_name() != '/device:GPU:0':
    print('WARNING: GPU device not found.')
else:
    print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))

import collections
import logging
import os
import pathlib
import re
import sys
import time
import numpy as np
import pandas as pd
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as st
from scipy import stats
import math
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add, Dense, Dropout, Embedding, GlobalAveragePooling1D, Input, Layer, LayerNormalization, MultiHeadAttention
import keras


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

# MASKING
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# # POINT WISE FEED FORWARD NETWORK
def point_wise_feed_forward_network(d_model, FNN):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(FNN, activation='relu'),  # (batch_size, seq_len, FNN)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

# ENCODER
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, FNN, rate):
        super(EncoderLayer, self).__init__()
        """
        Each encoder layer consists of sublayers:
        
        1. Multi-head attention (with padding mask)
        2. Point wise feed forward networks.
        
        Each of these sublayers has a residual connection around it followed by a layer normalization. Residual connections
        help in avoiding the vanishing gradient problem in deep networks.
        
        The output of each sublayer is LayerNorm(x + sublayer(x)). The normalization is done on the d_model (last) axis. 
        There are N encoder layers in the transformer.
        
        Arguments:
        FNN - fully connected dimension
        """

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = point_wise_feed_forward_network(d_model, FNN)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        """
        Forward pass for the Encoder layer
        
        Arguments:
        x - tensor of shape (batch_size, input_seq_len, d_model)
        training - boolean set to true to activate the training mode for dropout layers
        mask - boolean mask to ensure that the padding is not treated as part of the input
        
        Returns:
        out2 - tensor of shape (batch_size, input_seq_len, d_model)
        """      
        # 1. calculate self-attention using multi-head attention
        attn_output = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)
        
        # 2. apply dropout to the self-attention output
        attn_output = self.dropout1(attn_output, training=training)
        
        # 3. apply layer normalization on the sum of the input and the attention output (residual connection) to get the output
        # of the multi-head attention layer
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        
        # 4. pass the output of the mha layer through a ffn
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        
        # 5. apply dropout layer to ffn output
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # 6. apply layer normalization on sum of the output from mha and ffn output to get the  output of the encoder layer
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, FNN,
               vocab_size, rate): 
        super(Encoder, self).__init__()
        
        """
#         Encoder
#         The Encoder consists of:

#         1. Input Embedding
#         2. Positional Encoding 
#         3. N encoder layers
        
#         The output of the encoder is the input to the decoder.
#         """

        self.d_model = d_model
        self.num_layers = num_layers
        
#         self.embedding = tf.keras.layers.Embedding(vocab_size, self.d_model)

        self.pos_encoding = positional_encoding(vocab_size, self.d_model)
#         self.time_encoding = Time2Vector(vocab_size)
#         self.concat = tf.keras.layers.concatenate()

        self.enc_layers = [EncoderLayer(d_model, num_heads, FNN, rate)
                           for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask=None):
        """
        Forward pass for the encoder
        
        Arguments:
        x - tensor of shape (batch_size, input_seq_len)
        training - boolean set to true to activate the training model for dropout layers
        mask - boolean mask to ensure that the padding is not treated as part of the input
        
        Returns:
        out2 - tensor of shape (batch_size, input_seq_len, d_model)
        """
        
        seq_len = tf.shape(x)[1]

        # add the positional encoding to embedding
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, FNN, rate): #rate=0.1):
        super(DecoderLayer, self).__init__()
        
        """
        Decoder layer
        is composed by two multi-head attention blocks, one that takes the new input and uses self-attention, and the other one that
        combines it with the output of the encoder, followed by a fully connected block.
        
        Each decoder layer consists of sublayers:

        1. Masked multi-head attention (with look ahead mask and padding mask).
        2. Multi-head attention (with padding mask). V (value) and K (key) receive the encoder output as inputs. 
            Q (query) receives the output from the masked multi-head attention sublayer.
        3. Point wise feed forward networks.
        
        Each of these sublayers has a residual connection around it followed by a layer normalization. 
        The output of each sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the d_model (last) axis.

        There are N decoder layers in the transformer.

        As Q receives the output from decoder's first attention block, and K receives the encoder output, 
        the attention weights represent the importance given to the decoder's input based on the encoder's output. 
        In other words, the decoder predicts the next word by looking at the encoder output and self-attending to its own output. 
        See the demonstration above in the scaled dot product attention section.
        
        Arguments:
        FNN - fully connected dimension
        """

        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        self.ffn = point_wise_feed_forward_network(d_model, FNN)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        
    # def create_nan_mask(self,input_tensor):
    #     """
    #     Creates an attention mask where NaNs are replaced with 0 (masked) and valid values with 1.
    #     """
    #     mask = tf.where(tf.math.is_nan(input_tensor), 0.0, 1.0)  # Replace NaNs with 0, others with 1
    #     # mask = tf.reduce_min(mask, axis=-1)  # Reduce last dimension if input has features
    #     # mask = tf.expand_dims(mask, axis=1)  # Shape: (batch_size, 1, seq_len)
    #     # mask = tf.tile(mask, [1, tf.shape(input_tensor)[1], 1])  # Shape: (batch_size, seq_len, seq_len)
    #     return mask

    def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        """
        Forward pass for the decoder layer.
        
        Arguments:
        x - tensor of shape (batch_size, target_seq_len, d_model)
        enc_output - tensor of shape (batch_size, input_seq_len, d_model)
        training - boolean, set to true to activate the training mode for dropout layers
        look_ahead_mask - boolean mask for the target_input
        padding_mask - boolean mask for the second multihead attention layer
        
        Returns:
        out3 - tensor of shape (batch_size, target_seq_len, d_model)
        attn_weights_block1 - tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        attn_weights_block2 - tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """
        # BLOCK 1
        
        # attention_mask = self.create_nan_mask(x)
        # print(x)
        # print(attention_mask)
     
        # 1. calculate self-attention and return attention scores as attn_weights_block1
        attn1 = self.mha1(x, x, x)  # (batch_size, target_seq_len, d_model)
        
        # 2. apply dropout layer on the attention output
        attn1 = self.dropout1(attn1, training=training)
        
        # 3. apply layer normalization to the sum of the attention output (residual connection) and the input
        out1 = self.layernorm1(attn1 + x)

        # BLOCK 2
        # 4. calculate self-attention using the Q from the first block and K and V from the encoder output
        attn2 = self.mha2(value=enc_output, key=enc_output, query=out1)  # (batch_size, target_seq_len, d_model)
        
        # 5. apply dropout layer on the attention output
        attn2 = self.dropout2(attn2, training=training)
        
        # 6. apply layer normalization to the sum of the attention output and the output of the first block (residual connection)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        
        # BLOCK 3
        # 7. pass the output of the second block through a ffn
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        
        # 8. apply a dropout layer to the ffn output
        ffn_output = self.dropout3(ffn_output, training=training)
        
        # 9. apply layer normalization to the sum of the ffn output and the output of the second block
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3#, attn_weights_block1, attn_weights_block2

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, FNN,
               vocab_size, rate):
        super(Decoder, self).__init__()
        
        """
#         Decoder
#         The Decoder consists of:

#         Output Embedding (not included here due to the numerical nature of the time series)
#         Positional Encoding
#         N decoder layers
#         The target is put through an embedding which is summed with the positional encoding. 
#         The output of this summation is the input to the decoder layers. 
#         The output of the decoder is the input to the final linear layer.'''
#         """

        self.d_model = d_model
        
        self.num_layers = num_layers
    
        self.pos_encoding = positional_encoding(vocab_size, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, FNN, rate)
                               for _ in range(self.num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):
        """
        Forward pass for the decoder
        
        Arguments:
        x - tensor of shape (batch_size, target_seq_len, d_model)
        enc_output - tensor of shape (batch_size, input_seq_len, d_model)
        training - boolean set to true to activate the training mode for dropout layers
        look_ahead_mask - boolean mask for the target_input
        padding_mask - boolean mask for the second multihead attention layer
        
        Returns:
        x - tensor of shape (batch_size, target_seq_len, d_model)
        attention_weights - dictionary of tensors containing all the attention weights 
            each of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        # calculate positional encoding and add to time series embedding
        x += self.pos_encoding[:, :seq_len, :]

        # apply a dropout layer to x
        x = self.dropout(x, training=training)

        # use a for loop to pass x through a stack of decoder layers and update attention weights
        for i in range(self.num_layers):
            # pass x and the encoder output through a stack of decoder layers
            x = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

            # update attention_weights dictionary with the attention weights of block 1 and block 2
            # attention_weights[f'decoder_layer{i+1}_block1'] = block1
            # attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x#, attention_weights
    
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, FNN, num_patches_inp, num_patches_tar, features, rate, steps_for): #input_vocab_size,
               #target_vocab_size, rate=0.1): #pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
            
        """
        Transformer
        Transformer consists of the encoder, decoder and a final linear layer. 
        The output of the decoder is the input to the linear layer and its output is returned.
        
        Flow of data through Transformer:
        1. input passes through an encoder, which is just repeated encoder layers (multi-head attention of input, ffn to help detect features)
        2. encoder output passes through a decoder, consisting of the decoder layers (multi-head attention on generated output, multi-head attention with the Q from the first multi-head attention layer and the K and V from the encoder)
        3. After the Nth decoder layer, two dense layers and a softmax are applied to generate prediction for the next output in the sequence.
        """
        self.conv1d = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding="same", activation="relu")
        
        self.pos_embeddings_enc = PatchEncoder(num_patches_inp, projection_dim=d_model)
        self.pos_embeddings_dec = PatchEncoder(num_patches_tar, projection_dim=d_model)
                
        self._embedding = tf.keras.layers.Dense(d_model)
        self._embedding_dec = tf.keras.layers.Dense(d_model) # a separate embedding is needed for the decoder to have > 1 features/time_steps in the last dimension of the target set

        self.features = features
        self.encoder = Encoder(num_layers, d_model, num_heads, FNN, num_patches_inp, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, FNN, num_patches_tar, rate)

        self.final_layer = tf.keras.layers.Dense(features)
        # self.final_layer = tf.keras.layers.Dense(features+features)
        self.final_layer_dist = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t, 
            # (loc=t[...,:features],
                                 # scale=1e-3 + tf.math.softplus(0.05 * t[...,features:])))
                                 scale=1e-3 + tf.math.softplus(0.05 * t)))


        # self.final_layer_dist = tfp.layers.IndependentNormal(1)


    def call(self, inputs, training, enc_padding_mask,
           look_ahead_mask, padding_mask,steps_for):
        """
        Forward pass for the entire Transformer.
        Arguments:
        inp - input tensor of shape (batch_size, input_seq_len, fully_connected_dim)
        tar - target tensor of shape (batch_size, input_seq_len, fully_connected_dim)
        training - boolean set to true to activate the training mode for dropout layers
        look_ahead_mask - boolean mask for the target
        dec_padding_mask - boolean mask for the second multihead attention layer
        attention_weights - dictionary of tensors containing all the attention weights for the decoder, each of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """
        inp, tar = inputs

        inp = self._embedding(inp)    
        tar = self._embedding_dec(tar)

        enc_output = self.encoder(inp,
                                  training, 
                                  enc_padding_mask)  # (batch_size, inp_seq_len, d_model=fully_connected_dim=num_features)

        dec_output = self.decoder(
            tar, 
            enc_output, 
            training,
            look_ahead_mask, 
            padding_mask)  # dec_output.shape == (batch_size, tar_seq_len, d_model)
        
        final_output = self.final_layer(dec_output[:,-steps_for:,:self.features])
        final_output = self.final_layer_dist(final_output)
        
        # final_output = self.final_layer_dist(dec_output[:,-steps_for:,:self.features])
        # print(final_output_dist)
    
        return final_output#, attention_weights


    
    ### PREPROCESS ###
import pandas as pd
import utils
import tensorflow as tf
import sklearn
from sklearn.preprocessing import StandardScaler
import joblib
from joblib import dump, load

def truncate(x, y_real, y, steps_back, steps_for):
    in1_, in2_, out_, out_real_ = [], [], [], []

    for i in range(steps_back, len(y)-steps_for): #(len(x)-tback1-steps_for+1):
        in1_.append(x[:, :, :].tolist())
        in2_.append(y[i-steps_back:(i+steps_for), :].tolist())
        out_.append(y[i-steps_back:(i+steps_for), :].tolist())
        out_real_.append(y_real[i-steps_back:(i+steps_for), :].tolist())
       
    return np.array(in1_), np.array(in2_), np.array(out_), np.array(out_real_)

def preprocess(Xnp, Ynp, steps_for, steps_back_gfs, steps_back, batch_size, eval_ratio, num_target_features, station):
    """Truncate time series into (samples, time steps, features)"""

    X_in = Xnp
    test_Y_in = Ynp[np.newaxis,:,:]

    X_in = X_in.transpose((0,-1,1,2))
    test_X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2] * X_in.shape[3])

      """2) zero-normalize training, development (validation), and testing subsets using mean and std from training subset. Then reshape the inputs to decoder"""

    scaler_X=load('/blue/olabarrieta/eorozcolopez/proj/scaler_X%s.bin'%station)
    scaler_Y_in=load('/blue/olabarrieta/eorozcolopez/proj/scaler_Y_in%s.bin'%station)

    test_X_in = scaler_X.transform(test_X_in.reshape(-1, test_X_in.shape[-1])).reshape(test_X_in.shape)
    test_Y_in = scaler_Y_in.transform(test_Y_in.reshape(-1, test_Y_in.shape[-1])).reshape(test_Y_in.shape) 
   
    column = np.zeros((1,steps_for, test_Y_in.shape[-1]))
    test_Y_in = np.concatenate([test_Y_in, column], axis=1)

    return  test_X_in, test_Y_in

#=========================================================================================== #

def timestamp(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data


def shuffle(a, b, c):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=42).permutation(len(a))
    # print('lena',len(a))
    return a[p], b[p], c[p]

##########################################################################################################################################

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import backend as K
from pandas import read_csv

test_MSE_metric = tf.keras.metrics.MeanSquaredError()

def loss(y_real, y_hat):
    return tf.reduce_mean(tf.square(y_real - y_hat))

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
def test_step(x_batch_dev, tar_inp_dev, transformer, steps):
    # Run the forward pass of the model to get the probabilistic output
    
    batch, seq_length, d_model = tf.shape(tar_inp_dev)

    causal_mask = create_causal_mask(seq_length)
    causal_mask = tf.reshape(causal_mask, (1, 1, seq_length, seq_length))  # Shape: (1, 1, seq_length, seq_length)
    causal_mask = tf.cast(causal_mask, dtype=tf.float32)  # Convert to float
    
    test_output_dist = transformer((x_batch_dev, tar_inp_dev), False, False, look_ahead_mask=causal_mask, padding_mask=False, steps_for=steps)

    return test_output_dist.mean(),test_output_dist.stddev()#, loss_value_test

def predict(transformer, test_dataset, steps, num_target_features):
    output1 = np.empty((1,steps,num_target_features))
    output1_std = np.empty((1,steps,num_target_features))
    for (x_batch_test, tar_inp_test) in test_dataset:

        test_output, test_output_std = test_step(x_batch_test, tar_inp_test, transformer,steps)
        test_np = test_output.numpy()
        test_std_np = test_output_std.numpy()
        output1 = np.append(output1, test_np, axis=0)
        output1_std = np.append(output1_std, test_std_np, axis=0)
        
    output = output1[1:,:,:]
    output_std = output1_std[1:,:,:]
    return output, output_std