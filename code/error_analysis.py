"""
Example usage: python error_analysis.py cnn_model_1resblock_fancyleaf52.h5 ../data/train_dev_test_equal/dev.csv
"""
import sys
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd

import tensorflow as tf 

from utils import get_X_Y, get_X_Y_window, get_X_Y_intron

model_file = sys.argv[1]

model = tf.keras.models.load_model("trained_models/" + model_file)
model.summary()

dev_filename = sys.argv[2]

dev_df = pd.read_csv(dev_filename)
dev_X, dev_Y = get_X_Y_window(dev_df, window_size=20)

predictions = np.array(model.predict(dev_X)).flatten()
print(predictions.shape)
min_val = min(min(dev_Y), min(predictions))
max_val = max(max(dev_Y), max(predictions))
bins = np.linspace(min_val, max_val, 30)
plt.hist([predictions, dev_Y], bins, label=['pred', 'dev'])
plt.legend()
plt.show()