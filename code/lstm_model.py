"""
Example usage: python lstm_model.py ../data/train_dev_test/train.csv ../data/train_dev_test/dev.csv
"""
import sys
import pandas as pd
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Conv1D
from tensorflow.keras.layers import LSTM, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam

import wandb
from wandb.keras import WandbCallback

from utils import get_X_Y

# Hyperparameters
ALPHA = 0.001
EPOCHS = 10
BATCH_SIZE = 64

train_filename = sys.argv[1]
train_df = pd.read_csv(train_filename)

dev_filename = sys.argv[2]
dev_df = pd.read_csv(dev_filename)


def model(input_shape):    
    X_input = Input(shape = input_shape)

    # Conv layer
    X = Conv1D(filters=64, kernel_size=15, strides=4)(X_input) 
    X = BatchNormalization()(X)                             
    X = Activation("relu")(X)                                
    X = Dropout(rate=0.8)(X)                             

    batch_shape = (BATCH_SIZE, input_shape[0], input_shape[1])
    # Two Bidirectional LSTM layers
    X = Bidirectional(LSTM(units=64, batch_input_shape=batch_shape, \
    	return_sequences=True))(X)
    X = Dropout(rate=0.8)(X)                                 
    X = BatchNormalization()(X)                           
    
    X = Bidirectional(LSTM(units=64, batch_input_shape=batch_shape, \
    	return_sequences=True))(X)
    X = Dropout(rate=0.8)(X)                               
    X = BatchNormalization()(X)                             
    
    # Fully connected layers
    fc_sizes = [64, 1]
    for fc_size in fc_sizes: 
    	X = Dense(units=fc_size, activation='relu')(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model  

train_X, train_Y = get_X_Y(train_df)
dev_X, dev_Y = get_X_Y(dev_df)

wandb.init(project='splicing', config={'learning_rate': ALPHA, 
	'epochs': EPOCHS,
	'batch_size': BATCH_SIZE,
	'loss_function': 'mean_squared_error',
	'architecture': 'bi-lstm',
	'dataset': 'fullseq_all'
	})
config = wandb.config

model = model(input_shape = (train_X.shape[1], train_X.shape[2]))
model.summary()

opt = Adam(lr=ALPHA, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=["accuracy"])


model.fit(train_X, train_Y, validation_data=(dev_X, dev_Y), 
	callbacks=[WandbCallback()], batch_size=BATCH_SIZE, epochs=EPOCHS)
model.save("lstm_model.h5")

# loss, acc = model.evaluate(dev_X, dev_Y)
# print("Dev set accuracy = ", acc)

# predictions = model.predict(x)
