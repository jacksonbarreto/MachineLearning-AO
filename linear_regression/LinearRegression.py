import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as Lr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score, max_error, mean_absolute_error, \
    mean_squared_log_error, median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance, \
    mean_absolute_percentage_error, d2_absolute_error_score, d2_pinball_score, d2_tweedie_score, confusion_matrix
from sklearn.preprocessing import minmax_scale

from helpers.validator import is_valid_dataset, is_valid_target_variable_name, is_valid_test_size


class LinearRegression:
    def __init__(self, dataset_name, raw_dataset, target_variable_name, test_size=0.2, random_state=42):
        if is_valid_dataset(raw_dataset):
            if is_valid_target_variable_name(target_variable_name, raw_dataset):
                if is_valid_test_size(test_size, len(raw_dataset)):
                    self.__prepare_dataset__(raw_dataset, target_variable_name)
                    self.model = Lr()
                    self.test_size = test_size
                    self.random_state = random_state
                    self.dataset_name = dataset_name
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
            "algorithm": "Linear Regression",
            "score": self.model.score(self.X_test, self.y_test),
            "dataset": self.dataset_name,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "parameters": self.model.get_params(),
            "metrics": self.__get_metrics__()
        }

    def __fit__(self):
        self.model.fit(self.X_train, self.y_train)
        self.__predict__()

    def __get_metrics__(self):
        return {
            "explained_variance_score": explained_variance_score(self.y_test, self.y_pred),
            "max_error": max_error(self.y_test, self.y_pred),
            "mean_absolute_error": mean_absolute_error(self.y_test, self.y_pred),
            "mean_squared_error": mean_squared_error(self.y_test, self.y_pred),
            "mean_squared_log_error": mean_squared_log_error(self.y_test, minmax_scale(self.y_pred, feature_range=(0, 1))),
            "median_absolute_error": median_absolute_error(self.y_test, self.y_pred),
            "r2_score": r2_score(self.y_test, self.y_pred),
            # "mean_poisson_deviance": mean_poisson_deviance(self.y_test, minmax_scale(self.y_pred, feature_range=(0, 1))),
            # "mean_gamma_deviance": mean_gamma_deviance(self.y_test, minmax_scale(self.y_pred, feature_range=(0, 1))),
            "mean_absolute_percentage_error": mean_absolute_percentage_error(self.y_test, self.y_pred),
            "d2_absolute_error_score": d2_absolute_error_score(self.y_test, self.y_pred),
            "d2_pinball_score": d2_pinball_score(self.y_test, self.y_pred),
            "d2_tweedie_score": d2_tweedie_score(self.y_test, minmax_scale(self.y_pred, feature_range=(0, 1))),
            "confusion_matrix": self.__get_confusion_matrix__()
        }

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
