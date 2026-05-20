from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessing
from src.models.train import ModelTrainer


def test_model_trainer():
    ingestion_obj = DataIngestion()
    preprocessing_obj = DataPreprocessing()
    trainer = ModelTrainer()

    # Data ingestion
    train_data_path, val_data_path, test_data_path = (
        ingestion_obj.initiate_data_ingestion()
    )
    
    # Data preprocessing
    X_train_arr, X_val_arr, X_test_arr, y_train_arr, y_val_arr, y_test_arr = (
        preprocessing_obj.initiate_data_preprocessing(
            train_data_path, val_data_path, test_data_path
        )
    )
    # Model training
    trainer_output = trainer.initiate_model_training(
        X_train_arr, X_val_arr, y_train_arr, y_val_arr
    )
    loss_hist_train, r2_score_hist_train, loss_hist_val, r2_score_hist_val = (
        trainer_output
    )

    # Assert training actually happened
    assert (
        len(loss_hist_train) == trainer.trainer_config.epochs
    ), "Loss history length doesn't match epochs"

    # Assert loss decreased over training
    assert loss_hist_train[-1] < loss_hist_train[0], "Training loss did not decrease"

    # Assert model was saved
    from pathlib import Path

    assert Path(
        trainer.trainer_config.model_obj_file_path
    ).exists(), "Model file was not saved"

    # Assert R2 is reasonable — better than random
    assert (
        r2_score_hist_val[-1] > 0
    ), "Validation R2 is negative — model worse than baseline"
