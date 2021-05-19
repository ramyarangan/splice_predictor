from tensorflow import keras 

# Regression loss 
def compute_loss(y_pred, y_true):
	loss = keras.losses.mean_squared_error(y_pred, y_true)
	return loss.numpy()