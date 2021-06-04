"""
Example usage: python error_analysis.py cnn_model_1resblock_fancyleaf52.h5 ../data/train_dev_test_equal/dev.csv
"""
import sys
import os
import seaborn as sns
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd

import tensorflow as tf 
from tf_keras_vis.saliency import Saliency

from utils import *

model_file = sys.argv[1]
dev_filename = sys.argv[2]

# Plot distribution of predicted values
def plot_splicing_eff_dist(predictions, dev_Y):
	min_val = min(min(dev_Y), min(predictions))
	max_val = max(max(dev_Y), max(predictions))
	bins = np.linspace(min_val, max_val, 30)
	plt.hist([predictions, dev_Y], bins, label=['pred', 'dev'])
	plt.legend()
	plt.xlabel("Splicing efficiency")
	plt.ylabel("Number of examples")
	plt.title("Distribution of splicing efficiency values")
	plt.show()

# Plot average mean_squared_error per library
def plot_avg_mse_library(dev_df, predictions, dev_Y):
	library_types = dev_df.type.unique()
	losses = []
	min_val = min(min(dev_Y), min(predictions))
	max_val = max(max(dev_Y), max(predictions))
	bins = np.linspace(min_val, max_val, 30)
	library_types_ordered = ['synthetic', 'synthetic mutated', 'synthetic hairpin', \
		'synthetic alternative background', 'endogenous', 'endogenous - mutated sites', \
		'orthologous', 'synthetic control', 'synthetic hairpin - control', \
		'synthetic alternative background control', 'endogenous ss control', 'orthologous ss control']
	for library in library_types_ordered:
		all_idxs = np.arange(dev_df.shape[0])
		library_idxs = all_idxs[dev_df['type'] == library]
		library_loss = compute_loss(dev_Y[library_idxs], predictions[library_idxs])
		losses += [library_loss]
		plt.hist([predictions[library_idxs], dev_Y[library_idxs]], bins, label=['pred', 'dev'])
		plt.legend()
		plt.xlabel("Splicing efficiency")
		plt.ylabel("Number of examples")
		plt.title(library)
		plt.show()
	print(library_types_ordered)
	print(losses)
	plt.bar(library_types_ordered, losses, color='forestgreen', width=0.4)
	plt.xlabel("Library type")
	plt.xticks(rotation = 45)
	plt.ylabel("Mean squared error loss")
	plt.title("Loss by library type")
	plt.show()

# Saliency maps: https://github.com/keisen/tf-keras-vis/blob/master/examples/attentions.ipynb
def model_modifier(cloned_model):
	cloned_model.layers[-1].activation = tf.keras.activations.linear
	return cloned_model

def score_function(output):
	return output # tuple(np.array(output).flatten())

# num_plot is the number of entries to plot in the heatmap
# num_avg is the number of entries to compute the summary plot for
def plot_saliency_for_windows(dev_X, predictions, num_plot=100, num_avg=1000):
	if not os.path.isfile('analysis/saliency.csv'):
		saliency = Saliency(model, model_modifier=model_modifier, clone=False)
		saliency_map = saliency(score_function, dev_X, smooth_samples=20, smooth_noise=0.20)
		print(saliency_map.shape)
		np.savetxt('analysis/saliency.csv', saliency_map, fmt='%f')

	saliency_map = np.loadtxt('analysis/saliency.csv', dtype=float)

	# Used in baseline model
	baseline_saliency = np.zeros(saliency_map.shape[1])
	baseline_saliency[20:26] = 1
	baseline_saliency[60:68] = 1
	baseline_saliency[98:100] = 1
	baseline_saliency = baseline_saliency.reshape(1, (len(baseline_saliency)))

	entropy_vals = get_entropy(dev_X)

	# Can make plots in ranges of predicted splicing efficiency
	eff_ranges = [(0, 1)]#, (0.1, 0.5), (0.5, 0.8), (0.8, 1.0)]
	for (range_start, range_end) in eff_ranges:
		mask = (predictions >= range_start) & (predictions <= range_end)
		all_idxs = np.arange(len(predictions))
		idxs = all_idxs[mask]
		sample_size = min(num_plot, len(idxs))
		plot_size = min(num_avg, len(idxs))
		idx_sample = np.random.choice(idxs, size=plot_size, replace=False)
		idx_sample = np.random.choice(idxs, size=sample_size, replace=False)
		data = saliency_map[idx_sample,:]
		df = pd.DataFrame(data).melt()
		
		fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12,8), \
			gridspec_kw={'height_ratios':[1,5,5,10]})
		
		# Canonical splicing sequences
		sns.heatmap(baseline_saliency, cmap='Blues', ax=ax1, cbar=False)
		ax1.set_title("Canonical splicing sequences")
		ax1.axes.xaxis.set_visible(False)
		ax1.axes.yaxis.set_visible(False)
		
		# Per-position sequence entropy
		sns.lineplot(data=entropy_vals, ax=ax2)
		ax2.axes.xaxis.set_visible(False)
		ax2.set_ylabel("Sequence Entropy")
		ax2.set_title("Per-position sequence entropy")

		# Average saliency
		sns.lineplot(data=df, x="variable", y="value", ax=ax3)
		ax3.axes.xaxis.set_visible(False)
		ax3.set_ylabel("Saliency")
		ax3.set_title("Average per-position saliency")#: %f to %f" % (range_start, range_end))
		
		# Saliency heatmap for select examples
		sns.heatmap(data, cmap='Blues', ax=ax4, cbar=False)
		ax4.set_title("Saliency heatmap")#: %f to %f" % (range_start, range_end))
		ax4.set_xlabel("Position")
		ax4.set_ylabel("Example constructs")
		
		plt.show()

def plot_loss():
	df = pd.read_csv('analysis/wandb_best.csv')
	epochs = df['Step'].astype(int).tolist()
	training_loss = df['Training loss'].astype(float).tolist()
	val_loss = df['Validation loss'].astype(float).tolist()
	random_mse = [0.266] * len(epochs)
	all_zero_mse = [0.25] * len(epochs)
	baseline_mse = [0.188] * len(epochs)

	plt.plot(epochs, random_mse, color='#0072B2', label='Random loss', linestyle='dashed')
	plt.plot(epochs, all_zero_mse, color='#D55E00', label='All-zero loss', linestyle='dashed')
	plt.plot(epochs, baseline_mse, color='#CC79A7', label='Baseline loss', linestyle='dashed')
	plt.plot(epochs, training_loss, color='#009E73', label='Training loss')
	plt.plot(epochs, val_loss, color='#56B4E9', label='Validation loss')
	plt.title("Loss for best CNN model compared to baselines")
	plt.ylim((0, 0.3))
	plt.xticks(epochs)
	plt.xlabel("Epochs")
	plt.ylabel("Mean squared error loss")
	plt.legend()
	plt.show()

# Low-medium-high position weight matrix images

model = tf.keras.models.load_model("trained_models/" + model_file)
model.summary()

dev_df = pd.read_csv(dev_filename)
dev_X, dev_Y = get_X_Y_window(dev_df, window_size=20)

predictions = np.array(model.predict(dev_X)).flatten()
loss = compute_loss(dev_Y, predictions)
print("Overall mean squared error loss: %f" % loss)

# plot_splicing_eff_dist(predictions, dev_Y)
# plot_avg_mse_library(dev_df, predictions, dev_Y)
# plot_saliency_for_windows(dev_X, predictions)
# plot_loss()
