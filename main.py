"""Training entry point for the auto fuel efficiency predictor.

This script initializes the training pipeline and runs the end-to-end
training and evaluation workflow. It prints final metrics once the
training process completes.
"""

from src.pipelines.training_pipeline import TrainPipeline

if __name__ == "__main__":
    train_pipeline_obj = TrainPipeline()
    metrics = train_pipeline_obj.train()
    print(f"\nFinal metrics: {metrics}")
