import numpy as np
import pandas as pd
import pymongo
import warnings

import sklearn_json as skljson
from keras.saving.model_config import model_from_json
from sklearn.linear_model import LinearRegression


def mongo_database():
    client = pymongo.MongoClient(
        "mongodb+srv://jackson:V0BUiWpMm28k4mzU@cluster0.mhttx.mongodb.net/?retryWrites=true&w=majority")
    return client["ao"].get_collection("ia_models")


def deserialize_linear_regressor(model_dict):
    model = LinearRegression()
    model.set_params(**model_dict['params'])
    model.coef_ = np.array(model_dict['coef_'])
    model.intercept_ = np.array(model_dict['intercept_'])

    return model


if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    db = mongo_database()
    dataset_raw = pd.read_csv("heart.csv", header=0)
    dataset_cases = dataset_raw.drop(columns=['output'])

    best_linear_regression_model = db.find({"algorithm": "Linear Regression"}).sort("score", -1).limit(1)
    best_decision_tree_model = db.find({"algorithm": "Decision Tree"}).sort("score", -1).limit(1)
    best_rna_model = db.find({"algorithm": "RNA"}).sort("score", -1).limit(1)

    model_ln = deserialize_linear_regressor(best_linear_regression_model[0]["model"])
    model_tree = skljson.deserialize_model(best_decision_tree_model[0]["model"])
    model_rna = model_from_json(best_rna_model[0]["model"])

    msg_output_case_heart_attack = "Heart attack (1 = yes, 0 = no): " + str(int(dataset_raw.iloc[0]["output"]))
    case_for_prediction = dataset_cases[1:2]

    print("<<<<< Linear Regression >>>>>")
    print("Model Score: " + str(best_linear_regression_model[0]["score"]))
    print(msg_output_case_heart_attack)
    print("Prediction (between 0 and 1): " + str(model_ln.predict(case_for_prediction)[0]) + " %")
    print("<<<<< Decision Tree >>>>>")
    print("Model Score: " + str(best_decision_tree_model[0]["score"]))
    print(msg_output_case_heart_attack)
    print("Prediction (1 = yes, 0 = no): " + str(model_tree.predict(case_for_prediction)[0]))
    print("<<<<< RNA >>>>>")
    print("Model Score: " + str(best_rna_model[0]["score"]))
    print(msg_output_case_heart_attack)
    predict = np.argmax(model_rna.predict(case_for_prediction, verbose=0), axis=-1)
    print("Prediction: " + str((model_rna.predict(case_for_prediction, verbose=0)[0][0])))
