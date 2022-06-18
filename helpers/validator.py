import numpy as np
import pandas as pd
import scipy as sp


def is_numpy_array(raw_dataset):
    if isinstance(raw_dataset, np.ndarray):
        return True
    else:
        return False


def is_pandas_dataframe(raw_dataset):
    if isinstance(raw_dataset, pd.DataFrame):
        return True
    else:
        return False


def is_scipy_sparse_matrix(raw_dataset):
    if isinstance(raw_dataset, sp.sparse.csr.csr_matrix):
        return True
    else:
        return False


def is_list(raw_dataset):
    if isinstance(raw_dataset, list):
        return True
    else:
        return False


def is_valid_dataset(raw_dataset):
    if is_numpy_array(raw_dataset) or is_pandas_dataframe(raw_dataset) or is_scipy_sparse_matrix(
            raw_dataset) or is_list(raw_dataset):
        return True
    else:
        return False


def is_valid_target_variable_name(target_variable_name, raw_dataset):
    if isinstance(target_variable_name, str) and len(
            target_variable_name) > 0 and target_variable_name in raw_dataset.columns:
        return True
    else:
        return False


def is_valid_test_size(test_size, dataset_size):
    if (isinstance(test_size, float) and 0.0 <= test_size <= 1.0) or (
            isinstance(test_size, int) and 0 < test_size <= dataset_size):
        return True
    else:
        return False
