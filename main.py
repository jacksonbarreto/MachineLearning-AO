from decimal import Decimal, getcontext

import pandas as pd
import pymongo

from threading import Thread
from DecisionTree.DecisionTree import DecisionTree
from linear_regression.LinearRegression import LinearRegression
from rna.rna import RNA


def mongo_database():
    client = pymongo.MongoClient(
        "mongodb+srv://jackson:V0BUiWpMm28k4mzU@cluster0.mhttx.mongodb.net/?retryWrites=true&w=majority")
    return client["ao"].get_collection("ia_models")


def fit_linear_regression(dataset, database):
    getcontext().prec = 2
    current = Decimal(0.20)
    step = Decimal(0.01)
    end = Decimal(0.50)
    while current <= end:
        ln = LinearRegression("heart", dataset, "output", float(current), 42)
        database.insert_one(ln.get_information_model())
        current += step


def fit_decision_tree(dataset, database):
    getcontext().prec = 2
    current = Decimal(0.20)
    while current <= Decimal(0.50):
        dt = DecisionTree("heart", dataset, "output", float(current), 42)
        database.insert_one(dt.get_information_model())
        current += Decimal(0.01)


def fit_rna(dataset, database):
    getcontext().prec = 2
    current = Decimal(0.20)
    while current <= Decimal(0.50):
        for i in range(2, 112):
            rna = RNA("heart", dataset, "output", 3, [12, 32, 1],
                      ["relu", "relu", "sigmoid"], float(current), 42, 100, i)
            database.insert_one(rna.get_information_model())
    current += Decimal(0.01)


if __name__ == '__main__':
    db = mongo_database()
    dataset_raw = pd.read_csv("heart.csv", header=0)

    thread_linear_regression = Thread(target=fit_linear_regression, args=(dataset_raw, db))
    thread_linear_regression.start()

    thread_decision_tree = Thread(target=fit_decision_tree, args=(dataset_raw, db))
    thread_decision_tree.start()

    thread_rna = Thread(target=fit_rna, args=(dataset_raw, db))
    thread_rna.start()
