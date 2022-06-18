from decimal import Decimal, getcontext

import pandas as pd

from DecisionTree.DecisionTree import DecisionTree
from linear_regression.LinearRegression import LinearRegression
from rna.rna import RNA

if __name__ == '__main__':
    getcontext().prec = 2
    current = 0.20
    step = 0.01
    end = 0.50

    dataset_raw = pd.read_csv("heart.csv", header=0)
    while current <= end:
        ln = LinearRegression("heart", dataset_raw, "output", current, 42)
        # print(ln.get_information_model())
        tree = DecisionTree("heart", dataset_raw, "output", current, 42)
        # print(tree.get_information_model())
        rna = RNA("heart", dataset_raw, "output", 3, [12, 32, 1], ["relu", "relu", "sigmoid"], current, 42, 100, 10)
        print(rna.get_information_model())
        current += step

