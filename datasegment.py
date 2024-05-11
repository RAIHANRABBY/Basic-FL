import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit

class FederatedData:
    def __init__(self, data_path, num_clients):
        self.data_path = data_path
        self.num_clients = num_clients
        self.load_data()
        self.partitions = []

    def load_data(self):
        try:
            data = np.load(self.data_path)
            self.X = data['X']
            self.Y = data['Y']
            self.groups = data['Group']  # Assuming 'Group' contains patient identifiers
        except KeyError as e:
            raise ValueError(f"Missing expected data field: {e}")
        except FileNotFoundError as e:
            raise ValueError(f"Data file not found: {e}")

    def create_partitions(self):
        num_groups = len(set(self.groups))
        if self.num_clients > num_groups:
            raise ValueError("Number of clients exceeds the number of available patient groups.")
        
        gss = GroupShuffleSplit(n_splits=self.num_clients, test_size=0.2, random_state=42)
        for train_idx, test_idx in gss.split(self.X, self.Y, self.groups):
            partition_X_train, partition_X_test = self.X[train_idx], self.X[test_idx]
            partition_Y_train, partition_Y_test = self.Y[train_idx], self.Y[test_idx]
            self.partitions.append((partition_X_train, partition_Y_train, partition_X_test, partition_Y_test))

    def get_training_and_validation_data(self, client_idx):
        if client_idx < 0 or client_idx >= len(self.partitions):
            raise ValueError(f"Invalid client index. Must be between 0 and {len(self.partitions) - 1}.")
        
        partition_X_train, partition_Y_train, _, _ = self.partitions[client_idx]
        X_train, X_val, Y_train, Y_val = train_test_split(partition_X_train, partition_Y_train, test_size=0.2, random_state=42)
        return X_train, X_val, Y_train, Y_val

    def get_testing_data(self, client_idx):
        if client_idx < 0 or client_idx >= len(self.partitions):
            raise ValueError(f"Invalid client index. Must be between 0 and {len(self.partitions) - 1}.")
        
        _, _, partition_X_test, partition_Y_test = self.partitions[client_idx]
        return partition_X_test, partition_Y_test
