import flwr as fl
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Conv1DTranspose, Flatten, Dense, Reshape, Cropping1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten

# Define the model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(29, 37)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    # Instead of flattening, reshape the output to keep time sequence for LSTM
    # Assume each max pooling halves the sequence length, so from 29 it becomes approximately 7
    Reshape((-1, 128)),  # Adjust this depending on how your dimensions reduce
    LSTM(50, return_sequences=True),
    LSTM(50),
    Flatten(),  # Use Flatten here if needed, after LSTM which can handle sequences
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Adjust the number of output units and activation function based on your task
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Use 'categorical_crossentropy' for multi-class classification

data = np.load('E:/Federeted Learning/sample_data.npz')
X= data['X']
Y = data['Y']

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=.20)

# Load CIFAR-10 data
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Define the Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()  # Corrected method to get_weights
    
    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=100, batch_size=32)
        return model.get_weights(), len(x_train), {}
    
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)  # Corrected typo in 'evaluate'
        return loss, len(x_test), {'accuracy': accuracy}

# Start Flower client
fl.client.start_client(server_address = "127.0.0.1:8080", client=CifarClient().to_client())
