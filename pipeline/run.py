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
    or:
    $ python -m pipeline.run --framework sklearn
"""

import os
import sys
import argparse

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.config_reader import read_config
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.model_training_sklearn import ModelTraining as SklearnModelTraining
from src.model_training_lightning import ModelTraining as LightningModelTraining

def parse_arguments():
    """Parse command-line arguments for the ReedsShepp path planning pipeline.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments with the following attributes:
            - framework (str): The ML framework to use for training. Either 'sklearn' or 'lightning'.
              Defaults to 'lightning' (PyTorch Lightning implementation) if not specified.
              
    Example:
        # PyTorch Lightning (default, no arguments needed):
        python -m pipeline.run
        
        # Scikit-learn (explicitly specified):
        python -m pipeline.run --framework sklearn
    """
    parser = argparse.ArgumentParser(
        description="ReedsShepp path planning ML pipeline with framework selection"
    )
    parser.add_argument(
        "--framework",
        type=str,
        choices=["sklearn", "lightning"],
        default="lightning",
        help="ML framework to use (sklearn or lightning, default: lightning)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

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
    if args.framework == "sklearn":
        model_training = SklearnModelTraining(config)
    else:
        model_training = LightningModelTraining(config)
    model_training.run()

    print("Pipeline execution completed successfully!")