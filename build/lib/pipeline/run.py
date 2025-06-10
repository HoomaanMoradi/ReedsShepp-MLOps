"""
Orchestrates the end-to-end machine learning pipeline for ReedsShepp path planning.

This script serves as the main entry point for executing the complete ML pipeline.
It sequentially coordinates the following stages:
    1. Data Ingestion: Loads and validates input data from various sources
    2. Data Processing: Transforms and prepares the data for model training
    3. Model Training: Trains and validates the ReedsShepp path planning model

Example:
    To run the complete pipeline:
    $ python -m pipeline.run
"""

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.config_reader import read_config
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.model_training import ModelTraining

if __name__ == "__main__":
    # Load configuration from YAML file
    config_path = "config/config.yaml"
    config = read_config(config_path)

    # Execute data ingestion stage
    print("Starting data ingestion...")
    data_ingestion = DataIngestion(config)
    data_ingestion.run()

    # Execute data processing stage
    print("Starting data processing...")
    data_processing = DataProcessing(config)
    data_processing.run()

    # Execute model training stage
    print("Starting model training...")
    model_training = ModelTraining(config)
    model_training.run()

    print("Pipeline execution completed successfully!")
