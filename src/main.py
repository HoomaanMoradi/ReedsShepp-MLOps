"""Main entry point for the ReedsShepp-MLOps pipeline.

This script orchestrates the complete machine learning pipeline including:
1. Data Ingestion: Loading and validating raw data
2. Data Processing: Feature engineering and data preparation
3. Model Training: Training and evaluating the MLP classifier

Example:
    python src/main.py
"""

from config_reader import read_config
from data_ingestion import DataIngestion
from data_processing import DataProcessing
from model_training import ModelTraining


def main() -> None:
    """Execute the complete ML pipeline from data ingestion to model training."""
    # Read configuration from YAML file
    config_path = "config/config.yaml"
    config = read_config(config_path)

    # 1. Data Ingestion Phase
    # ------------------------
    # Initialize and run data ingestion to load and validate raw data
    print("\n=== Starting Data Ingestion ===")
    data_ingestion = DataIngestion(config)
    data_ingestion.run()

    # 2. Data Processing Phase
    # ------------------------
    # Process the raw data into features suitable for model training
    print("\n=== Starting Data Processing ===")
    data_processing = DataProcessing(config)
    data_processing.run()

    # 3. Model Training Phase
    # -----------------------
    # Train and evaluate the MLP classifier
    print("\n=== Starting Model Training ===")
    model_training = ModelTraining(config)
    model_training.run()


if __name__ == "__main__":
    main()
