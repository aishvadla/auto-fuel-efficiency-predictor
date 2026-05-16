from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessing
import numpy as np


def test_data_preprocessing():
    test_size = 0.3
    random_state = 24
    ingestion_obj = DataIngestion(test_size, random_state)
    preprocessing_obj = DataPreprocessing()
    train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()
    X_train_arr, X_test_arr, y_train_arr, y_test_arr = preprocessing_obj.initiate_data_preprocessing(train_data_path, test_data_path)
    
    # Row count matches
    assert len(X_train_arr) == len(y_train_arr), f"Training features and target row count don't match"
    assert len(X_test_arr) == len(y_test_arr), f"Test features and target row count don't match"

    # No missing values after preprocessing
    assert not np.isnan(X_train_arr).any(), f"Training featrues contain NaN after preprocessing"
    assert not np.isnan(X_test_arr).any(), f"Test featrues contain NaN after preprocessing"

    # Correct dtypes for PyTorch
    assert X_train_arr.dtype == np.float32, f"X_train should be float32"
    assert y_train_arr.dtype == np.float32, f"y_train should be float32"

    # Shape sanity check
    assert X_train_arr.shape[1] == X_test_arr.shape[1], f"Training and test data feature count don't match"