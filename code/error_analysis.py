"""
Example usage: python error_analysis.py ../data/train_dev_test/dev.csv
"""
import sys
import pandas as pd
import tensorflow as tf 

from utils import get_X_Y

model_file = sys.argv[1]

model = tf.keras.models.load_model("trained_models/" + model_file)
model.summary()

dev_filename = sys.argv[2]

dev_df = pd.read_csv(dev_filename)
dev_X, dev_Y = get_X_Y(dev_df)

predictions = model.predict(dev_X)
print(predictions.shape)
print(predictions[:5])
print(dev_Y[:5])