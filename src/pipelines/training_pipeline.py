from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessing
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.utils.logger import logger


class TrainPipeline:
    def __init__(self):
        self.ingestion_obj = DataIngestion()
        self.preprocessor_obj = DataPreprocessing()
        self.trainer_obj = ModelTrainer()
        self.evaluator_obj = ModelEvaluator()

    def train(self):
        logger.info("Starting training pipeline")
    
        # Stage 1 — ingestion
        train_path, val_path, test_path = self.ingestion_obj.initiate_data_ingestion()
        
        # Stage 2 — preprocessing
        X_train, X_val, X_test, y_train, y_val, y_test = (
            self.preprocessor_obj.initiate_data_preprocessing(train_path, val_path, test_path)
        )
        
        # Stage 3 — training
        self.trainer_obj.initiate_model_training(X_train, X_val, y_train, y_val)
        
        # Stage 4 — evaluation
        metrics = self.evaluator_obj.initiate_model_evaluation(X_test, y_test)
        
        logger.info(f"Training pipeline complete. Metrics: {metrics}")
        return metrics