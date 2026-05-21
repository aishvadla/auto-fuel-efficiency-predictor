from src.pipelines.training_pipeline import TrainPipeline

if __name__ == "__main__":
    train_pipeline_obj = TrainPipeline()
    metrics = train_pipeline_obj.train()
    print(f"\nFinal metrics: {metrics}")
