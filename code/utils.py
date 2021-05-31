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

def get_X_Y_intron(df):
    seqs = []

    for __, row in df.iterrows():
        fiveprime_idx = row['fivess_idx']
        full_seq = row['full_seq']

        # 145 is the length of the longest intron in the library
        seq = full_seq[(fiveprime_idx):(fiveprime_idx+145)]
        seqs += [seq]

    seqs = np.array(seqs)
    X = one_hot_encoding(seqs)
    Y = np.array(df['splicing_eff'].astype(float).tolist())
  
    return X, Y
 

def get_X_Y_window(df_list, window_size=20):
    X_all, Y_all = get_X_Y_window(df_list[0], window_size=window_size)
    for df in df_list[1:]:
        X_all = np.append(X_all, X, axis=0)
        Y_all = np.append(Y_all, Y)
    return X_all, Y_all

# Returns modified sequences that have windows
# centered on the 5'SS, BP, and 3'SS
def get_X_Y_window(df, window_size=20):
    seqs = []

    for __, row in df.iterrows():
        fiveprime_idx = row['fivess_idx']
        bp_idx = row['bp_idx']
        threeprime_idx = row['threess_idx']
        full_seq = row['full_seq']

        seq = full_seq[(fiveprime_idx-window_size):(fiveprime_idx+window_size)]
        seq += full_seq[(bp_idx-window_size):(bp_idx+window_size)]
        seq += full_seq[(threeprime_idx-window_size):(threeprime_idx+window_size)]

        seqs += [seq]

    seqs = np.array(seqs)
    X = one_hot_encoding(seqs)
    Y = np.array(df['splicing_eff'].astype(float).tolist())
  
    return X, Y
