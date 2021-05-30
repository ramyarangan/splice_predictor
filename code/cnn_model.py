"""
Example usage: python cnn_model.py ../data/train_dev_test_equal/dev.csv ../data/train_dev_test_equal/train_pt1.csv ../data/train_dev_test_equal/train_pt2.csv
"""
import sys
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Input, Conv1D
from tensorflow.keras.layers import BatchNormalization, Add, Flatten
from tensorflow.keras.optimizers import Adam

import wandb
from wandb.keras import WandbCallback

from utils import get_X_Y, get_X_Y_window

# Hyperparameters
ALPHA = 0.001
EPOCHS = 10
BATCH_SIZE = 64
DROPOUT_RATE = 0.1

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

def residual_block(X, F, f, w):
    X_shortcut = X
    X = Conv1D(filters=F, kernel_size=1, strides=1, padding='valid')(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    
    X = Conv1D(filters=F, kernel_size=f, strides=w, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    
    X = Conv1D(filters=F, kernel_size=1, strides=1, padding='valid')(X)
    X = BatchNormalization()(X)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def model(input_shape):    
    X_input = Input(shape = input_shape)

    # Conv layer
    X = Conv1D(filters=32, kernel_size=1, strides=1)(X_input) 
    X_shortcut = Conv1D(filters=32, kernel_size=1, strides=1)(X)
    X = residual_block(X, 32, 11, 1)
    X = residual_block(X, 32, 11, 1)
    X = residual_block(X, 32, 11, 1)
    X = residual_block(X, 32, 11, 1)
    X = Conv1D(filters=32, kernel_size=1, strides=1)(X)
    X = Add()([X, X_shortcut])
    X = Conv1D(filters=3, kernel_size=1, strides=1)(X)
    X = Activation('softmax')(X)
    X = BatchNormalization()(X)
    X = Flatten()(X)
    X = Dense(units=1, activation='relu')(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model  

train_X, train_Y = get_X_Y_window(train_df, window_size=20)
dev_X, dev_Y = get_X_Y_window(dev_df, window_size=20)

wandb.init(project='splicing', config={'learning_rate': ALPHA, 
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'dropout_rate': DROPOUT_RATE,
    'loss_function': 'mean_squared_error',
    'architecture': 'cnn',
    'dataset': 'fullseq_all'
    })

model = model(input_shape = (train_X.shape[1], train_X.shape[2]))
model.summary()

opt = Adam(learning_rate=ALPHA, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=["accuracy"])

model.fit(train_X, train_Y, validation_data=(dev_X, dev_Y), 
    callbacks=[WandbCallback()], batch_size=BATCH_SIZE, epochs=EPOCHS)
model.save("trained_models/cnn_model_window20.h5")

# loss, acc = model.evaluate(dev_X, dev_Y)
# print("Dev set accuracy = ", acc)
