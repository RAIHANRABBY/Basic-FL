import flwr as fl
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Reshape
from datasegment import FederatedData
import numpy as np
from keras.layers import Input
from flwr.client import NumPyClient

def create_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Reshape((-1, 128)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class CustomClient(NumPyClient):
    def __init__(self, x_train, y_train, x_test, y_test, input_shape):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = create_model(input_shape)

    def get_parameters(self,config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        epochs = config.get("epochs", 2)
        batch_size = config.get("batch_size", 32)
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {'accuracy': accuracy}

def run_client(x_train, y_train, x_test, y_test, client_idx, input_shape):
    client = CustomClient(x_train, y_train, x_test, y_test, input_shape)
    fl.client.start_client(server_address= "127.0.0.1:8080", client=client.to_client())


def main(client_idx):
    data_path = 'E:/Federeted Learning/raw_data.npz'  # Adjust as needed
    num_clients = 10  # Adjust as needed
    federated_data = FederatedData(data_path, num_clients)
    federated_data.create_partitions()
    
    x_train, x_val, y_train, y_val = federated_data.get_training_and_validation_data(client_idx)
    x_test, y_test = federated_data.get_testing_data(client_idx)

    # Adjust input shape according to your data dimensions
    input_shape = (x_train.shape[1], x_train.shape[2])

    run_client(np.concatenate([x_train, x_val]), np.concatenate([y_train, y_val]), x_test, y_test, client_idx, input_shape)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python client.py <client_index>")
        sys.exit(1)
    
    client_idx = int(sys.argv[1])
    main(client_idx)
