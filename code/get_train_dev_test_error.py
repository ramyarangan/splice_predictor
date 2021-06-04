"""
Example usage: python error_analysis.py cnn_model_1resblock_fancyleaf52.h5 ../data/train_dev_test_equal/
"""
import sys
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf 

from utils import *

model_file = sys.argv[1]
train_dev_test_dir = sys.argv[2]

def get_training_dev_test_loss(model_file, \
	train_dev_test_dir='../data/train_dev_test_equal/', \
	do_intron=False, do_full=False, window_size=20):
	model = tf.keras.models.load_model("trained_models/" + model_file)
	# model.summary()

	dev_df = pd.read_csv(train_dev_test_dir + 'dev.csv')
	X, Y = get_X_Y_window(dev_df, window_size=20)
	if do_intron:
		X, Y = get_X_Y_intron(dev_df)
	if do_full:
		X, Y = get_X_Y(dev_df)
	predictions = np.array(model.predict(X)).flatten()
	loss = compute_loss(Y, predictions)
	r2 = compute_R2(Y, predictions)
	print("Dev set MSE loss: %f" % loss)
	print("Dev set R2: %f" % r2)
	plt.scatter(Y, predictions)
	plt.show()

	train_df_1 = pd.read_csv(train_dev_test_dir + 'train_pt1.csv')
	train_df_2 = pd.read_csv(train_dev_test_dir + 'train_pt2.csv')

	train_df = train_df_1
	train_df = pd.concat([train_df_1, train_df_2])
	X, Y = get_X_Y_window(train_df, window_size=20)
	if do_intron:
		X, Y = get_X_Y_intron(train_df)
	if do_full:
		X, Y = get_X_Y(train_df)
	predictions = np.array(model.predict(X)).flatten()
	loss = compute_loss(Y, predictions)
	r2 = compute_R2(Y, predictions)
	print("Training set MSE loss: %f" % loss)
	print("Training set R2: %f" % r2)

	test_df = pd.read_csv(train_dev_test_dir + 'test.csv')
	X, Y = get_X_Y_window(test_df, window_size=20)
	if do_intron:
		X, Y = get_X_Y_intron(test_df)
	if do_full:
		X, Y = get_X_Y(test_df)
	predictions = np.array(model.predict(X)).flatten()
	loss = compute_loss(Y, predictions)
	r2 = compute_R2(Y, predictions)
	print("Test set MSE loss: %f" % loss)
	print("Test set R2: %f" % r2)

get_training_dev_test_loss(model_file, train_dev_test_dir=train_dev_test_dir)
