# Using the dataset from https://archive.ics.uci.edu/dataset/9/auto+mpg

import sys
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from src.utils.logger import logger
from src.utils.exception import CustomException
from sklearn.model_selection import train_test_split

DATA_SRC_URL = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)


@dataclass
class DataIngestionConfig:
    raw_data_path: str = Path("data") / "raw" / "auto-mpg.csv"
    train_data_path: str = Path("data") / "raw" / "train.csv"
    test_data_path: str = Path("data") / "raw" / "test.csv"


class DataIngestion:
    def __init__(self, test_size=0.2, random_state=23):
        self.ingestion_config = DataIngestionConfig()
        self.test_size = test_size
        self.random_state = random_state

    def initiate_data_ingestion(self):
        logger.info("Initiated data ingestion")
        try:
            column_names = [
                "MPG",
                "Cylinders",
                "Displacement",
                "Horsepower",
                "Weight",
                "Acceleration",
                "Model Year",
                "Origin",
            ]
            df = pd.read_csv(
                DATA_SRC_URL,
                names=column_names,
                sep=" ",
                na_values="?",
                comment="\t",
                skipinitialspace=True,
            )
            logger.info("Read the dataset as Pandas dataframe")
            raw_data_path = Path(self.ingestion_config.raw_data_path).parent
            raw_data_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, header=True)

            logger.info("Train Test split initiated")
            train_set, test_set = train_test_split(
                df, test_size=self.test_size, random_state=self.random_state
            )
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logger.info("Data Ingestion Completed. Files written to data/raw/*.csv")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
