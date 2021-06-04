"""
Bidirectional LSTM architecture with dropout layers for predicting splicing efficiency from sequence

Example usage: python lstm_model.py ../data/train_dev_test/train.csv ../data/train_dev_test/dev.csv
"""
import sys
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Conv1D
from tensorflow.keras.layers import LSTM, Bidirectional, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import int_shape

import wandb
from wandb.keras import WandbCallback

from utils import get_X_Y, get_X_Y_window

# Hyperparameters
ALPHA = 0.001
EPOCHS = 10
BATCH_SIZE = 64
DROPOUT_RATE = 0.1

# Disables cuDNN until the Driver is updated
# tf.compat.v1.disable_eager_execution()

dev_filename = sys.argv[1]
dev_df = pd.read_csv(dev_filename)

train_filename = sys.argv[2]
train_df_1 = pd.read_csv(train_filename)
train_df_2 = None
if len(sys.argv) > 3:
    train_df_2 = pd.read_csv(sys.argv[3])

train_df = train_df_1
if train_df_2 is not None:
    train_df = pd.concat([train_df_1, train_df_2])

def model(input_shape):    
    X_input = Input(shape = input_shape)

    # Conv layer
    X = Conv1D(filters=64, kernel_size=15, strides=4)(X_input) 
    X = BatchNormalization()(X)                             
    X = Activation("relu")(X)                                
    X = Dropout(rate=DROPOUT_RATE)(X)                             

    # Needed for running LSTM layer on GPU
    batch_shape = (BATCH_SIZE, int_shape(X)[1], int_shape(X)[2])
    
    # Two Bidirectional LSTM layers
    X = Bidirectional(LSTM(units=64, batch_input_shape=batch_shape, \
    	return_sequences=True))(X)
    X = Dropout(rate=DROPOUT_RATE)(X)                                 
    X = BatchNormalization()(X)                           

    # Needed for running LSTM layer on GPU
    batch_shape = (BATCH_SIZE, int_shape(X)[1], int_shape(X)[2])
     
    X = Bidirectional(LSTM(units=64, batch_input_shape=batch_shape, \
    	return_sequences=True))(X)
    X = Dropout(rate=DROPOUT_RATE)(X)                               
    X = BatchNormalization()(X)                             
    
    # Fully connected layers
    X = Flatten()(X)
    fc_sizes = [16, 1]
    for fc_size in fc_sizes: 
    	X = Dense(units=fc_size, activation='relu')(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model  

train_X, train_Y = get_X_Y_window(train_df, window_size=20)
dev_X, dev_Y = get_X_Y_window(dev_df, window_size=20)

wandb.init(project='splicing', config={'learning_rate': ALPHA, 
	'epochs': EPOCHS,
	'batch_size': BATCH_SIZE,
	'dropout_rate': DROPOUT_RATE,
	'loss_function': 'mean_squared_error',
	'architecture': 'bi-lstm',
	'dataset': 'fullseq_all'
	})

model = model(input_shape = (train_X.shape[1], train_X.shape[2]))
model.summary()

opt = Adam(learning_rate=ALPHA, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=["accuracy"])

model.fit(train_X, train_Y, validation_data=(dev_X, dev_Y), 
	callbacks=[WandbCallback()], batch_size=BATCH_SIZE, epochs=EPOCHS)
model.save("trained_models/lstm_model_window20.h5")

# loss, acc = model.evaluate(dev_X, dev_Y)
# print("Dev set accuracy = ", acc)

