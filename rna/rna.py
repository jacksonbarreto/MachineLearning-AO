import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.metrics import AUC, MeanSquaredError, BinaryAccuracy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from helpers.validator import is_valid_target_variable_name, is_valid_test_size, is_valid_dataset


class RNA:
    def __init__(self, dataset_name, raw_dataset, target_variable_name, layers, neurons, activations,
                 test_size=0.2, random_state=42, epochs=100, batch_size=10):
        if is_valid_dataset(raw_dataset):
            if is_valid_target_variable_name(target_variable_name, raw_dataset):
                if is_valid_test_size(test_size, len(raw_dataset)):
                    self.__prepare_dataset__(raw_dataset, target_variable_name)
                    self.model = Sequential()
                    self.test_size = test_size
                    self.random_state = random_state
                    self.dataset_name = dataset_name
                    self.epochs = epochs
                    self.batch_size = batch_size
                    self.layers = layers
                    self.neurons = neurons
                    self.activations = activations
                    if not self.__is_valid_layer_configuration__():
                        raise Exception("Invalid layer configuration")
                    if not self.__is_valid_epochs__():
                        raise Exception("Invalid epochs")
                    if not self.__is_valid_batch_size__():
                        raise Exception("Invalid batch size")
                    self.__prepare_layers__()
                    self.__prepare_training_data__(test_size, random_state)
                    self.__fit__()
                else:
                    raise ValueError("test_size must be between 0.0 and 1.0 or between 1 and %i" % len(raw_dataset))
            else:
                raise ValueError("Invalid target variable name, the variable name must exist in the dataset")

        else:
            raise Exception(
                "Invalid dataset type, the dataset should be of the type: "
                "lists, numpy arrays, scipy-sparse matrices or pandas dataframes")

    def get_information_model(self):
        return {
            "algorithm": "RNA",
            "score": self.model.score(self.X_test, self.y_test),
            "parameters": {
                "layers": self.layers,
                "neurons": self.neurons,
                "activations": self.activations,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "model": self.model.to_json()
            },
            "dataset": self.dataset_name,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "metrics": self.__get_metrics__()
        }

    def __get_metrics__(self):
        return {
            "accuracy": self.__get_accuracy__(),
            "mean_squared_error": self.__get_mse__(),
            "binary_accuracy": self.__get_binary_accuracy__(),
            "confusion_matrix": self.__get_confusion_matrix__()
        }

    def __fit__(self):
        self.model.fit(self.X_train, self.y_train)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae', 'mape'])
        self.__predict__()

    def __predict__(self):
        self.y_pred = self.model.predict(self.X_test)

    def __prepare_dataset__(self, raw_dataset, target_variable_name):
        self.dataset = raw_dataset.drop(target_variable_name, axis=1)
        self.target_column = raw_dataset[target_variable_name]

    def __prepare_training_data__(self, test_size, random_state):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset, self.target_column,
                                                                                test_size=test_size,
                                                                                random_state=random_state)

    def __get_confusion_matrix__(self):
        return pd.DataFrame(confusion_matrix(self.y_test, np.where(self.y_pred > 0.5, 1, 0))).to_json()

    def __prepare_layers__(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                self.model.add(Dense(self.neurons[i], activation=self.activations[i], input_dim=self.dataset.columns))
            else:
                self.model.add(Dense(self.neurons[i], activation=self.activations[i]))

    def __get_accuracy__(self):
        auc = AUC()
        auc.update_state(self.y_test, self.y_pred)
        return auc.result().numpy()

    def __get_mse__(self):
        mse = MeanSquaredError()
        mse.update_state(self.y_test, self.y_pred)
        return mse.result().numpy()

    def __get_binary_accuracy__(self):
        ba = BinaryAccuracy()
        ba.update_state(self.y_test, self.y_pred)
        return ba.result().numpy()

    def __is_valid_layer_configuration__(self):
        if self.__is_valid_layer_size__() and \
                self.__is_valid_neurons__() \
                and self.__is_valid_layer__() and self.__is_valid_activations__():
            return True
        else:
            return False

    def __is_valid_epochs__(self):
        if self.epochs > 0:
            return True
        else:
            return False

    def __is_valid_batch_size__(self):
        if self.batch_size > 0:
            return True
        else:
            return False

    def __is_valid_layer_size__(self):
        if (len(self.layers) > 0 and len(self.neurons) > 0 and len(self.activations) > 0) and (
                len(self.layers) == len(self.neurons) == len(self.activations)):
            return True
        else:
            return False

    def __is_valid_neurons__(self):
        for neuron in self.neurons:
            if isinstance(neuron, int) and neuron > 0:
                return True
            else:
                return False

    def __is_valid_layer__(self):
        for layer in self.layers:
            if isinstance(layer, int) and layer > 0:
                return True
            else:
                return False

    def __is_valid_activations__(self):
        for activation in self.activations:
            if isinstance(activation, str) and activation in ["relu", "sigmoid", "tanh", "softmax"]:
                return True
            else:
                return False
