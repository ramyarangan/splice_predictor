"""
Example usage: python analyze_data_distribution.py ../data/train_dev_test/dev.csv
"""
import sys
import pandas as pd
import numpy as np
import tensorflow as tf 

from utils import get_X_Y
from matplotlib import pyplot as plt

dev_filename = sys.argv[1]
dev_df = pd.read_csv(dev_filename)
dev_X, dev_Y = get_X_Y(dev_df)

plt.hist(dev_Y)
plt.show() 

print(np.sum(dev_Y > 0))
print(np.sum(dev_Y > 0.05))
print(np.sum(dev_Y > 0.1))
print(np.sum(dev_Y > 0.5))
print(np.sum(dev_Y > -1))