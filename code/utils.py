from tensorflow import keras 
import numpy as np

# Regression loss 
def compute_loss(y_pred, y_true):
    loss = keras.losses.mean_squared_error(y_pred, y_true)
    return loss.numpy()

# One-hot encoding indices
nt_dict = {'A': 0, 'C': 1, 'T': 2, 'G': 3}

# X is a list of sequences, converted to a one-hot encoding
def one_hot_encoding(X):
    seq_len = len(X[0])
    num_nts = len(nt_dict.keys())

    one_hot_X = np.zeros((len(X), seq_len, num_nts))
    for ii, seq in enumerate(X):
        idxs = np.array([nt_dict[x] for x in list(seq)])
        one_hot_X[ii, :, :] = np.eye(num_nts)[idxs]

    return one_hot_X

# Assemble one-hot-encoded inputs from input dataframe
def get_X_Y(df):
    seqs = np.array(df['full_seq'].tolist())
    X = one_hot_encoding(seqs)
    Y = np.array(df['splicing_eff'].astype(float).tolist())
    
    return X, Y