from src.data.ingestion import DataIngestion


def test_data_ingestion():
    test_size = 0.3
    random_state = 24
    ingestion_obj = DataIngestion(test_size, random_state)
    train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()
