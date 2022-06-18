import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from helpers.validator import is_valid_dataset, is_valid_target_variable_name, is_valid_test_size
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, top_k_accuracy_score, \
    average_precision_score, brier_score_loss, roc_auc_score, f1_score, precision_score


class DecisionTree:
    def __init__(self, raw_dataset, target_variable_name, test_size=0.2, random_state=42):
        if is_valid_dataset(raw_dataset):
            if is_valid_target_variable_name(target_variable_name, raw_dataset):
                if is_valid_test_size(test_size, len(raw_dataset)):
                    self.__prepare_dataset__(raw_dataset, target_variable_name)
                    self.model = DecisionTreeClassifier(max_features="auto")
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
            "algorithm": "Decision Tree",
            "score": self.model.score(self.X_test, self.y_test),
            "parameters": self.model.get_params(),
            "metrics": self.__get_metrics__()
        }

    def __get_metrics__(self):
        return {
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "balanced_accuracy": balanced_accuracy_score(self.y_test, self.y_pred),
            "top_k_accuracy": top_k_accuracy_score(self.y_test, self.y_pred),
            "average_precision": average_precision_score(self.y_test, self.y_pred),
            "neg_brier_score": brier_score_loss(self.y_test, self.y_pred),
            "f1_score": f1_score(self.y_test, self.y_pred),
            "precision": precision_score(self.y_test, self.y_pred),
            "roc_auc": roc_auc_score(self.y_test, self.y_pred),
            "confusion_matrix": self.__get_confusion_matrix__()
        }

    def __fit__(self):
        self.model.fit(self.X_train, self.y_train)
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
        return pd.DataFrame(confusion_matrix(self.y_test, self.y_pred))
