"""Model Training Module for MLPClassifier.

This module provides a comprehensive pipeline for training and evaluating a Multi-layer 
Perceptron (MLP) classifier using scikit-learn. It handles the complete machine learning 
workflow including data loading, model configuration, training with early stopping, 
hyperparameter tuning, and performance evaluation with multiple metrics.

The module is designed to be configurable through YAML configuration files and includes 
MLflow integration for experiment tracking and model versioning.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer

from src.logger import get_logger

logger = get_logger(__name__)


class ModelTraining:
    """A class to manage the complete MLP model training pipeline.

    This class handles the entire workflow from data loading to model evaluation,
    with configurable parameters for model architecture and training process.
    It provides a clean interface for training MLP models with support for early
    stopping, learning rate adaptation, and comprehensive evaluation.

    Attributes:
        model_training_config (Dict[str, Any]): Configuration parameters for model training
        data_ingestion_config (Dict[str, Any]): Configuration for data loading
        processed_dir (Path): Directory containing processed data files
        model_output_dir (Path): Directory to save trained models and outputs
        model_name (str): Name of the model for saving and logging
        train_path (Path): Path to training data file
        val_path (Path): Path to validation data file
        test_path (Path): Path to test data file
        model_output_path (Path): Full path where the trained model will be saved
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the ModelTraining with configuration.

        Args:
            config: Configuration dictionary containing:
                - model_training: Parameters for model training
                - data_ingestion: Paths and settings for data loading

        Creates necessary directories for model outputs if they don't exist.
        """
        self.model_training_config = config["model_training"]
        self.data_ingestion_config = config["data_ingestion"]
        artifact_dir = Path(self.data_ingestion_config["artifact_dir"])
        self.processed_dir = artifact_dir / "processed"
        self.model_output_dir = artifact_dir / "models"
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = self.model_training_config["model_name"]

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load training, validation, and test data from processed CSV files.

        The method expects to find 'train.csv', 'validation.csv', and 'test.csv' in the
        processed directory. These files should contain both features and the target
        variable ('output' column).

        Returns:
            A tuple containing:
                - train_data: Training dataset with features and target
                - val_data: Validation dataset for early stopping
                - test_data: Test dataset for final evaluation

        Raises:
            FileNotFoundError: If any of the required CSV files are not found in processed_dir

        Note:
            The data is expected to be preprocessed and contain an 'output' column as the target.
        """
        self.train_path = self.processed_dir / "train.csv"
        self.val_path = self.processed_dir / "validation.csv"
        self.test_path = self.processed_dir / "test.csv"

        logger.debug(f"Loading training data from {self.train_path}")
        train_data = pd.read_csv(self.train_path)

        logger.debug(f"Loading validation data from {self.val_path}")
        val_data = pd.read_csv(self.val_path)

        logger.debug(f"Loading test data from {self.test_path}")
        test_data = pd.read_csv(self.test_path)

        logger.info(
            f"Loaded {len(train_data)} training samples, "
            f"{len(val_data)} validation samples and "
            f"{len(test_data)} test samples"
        )
        return train_data, val_data, test_data

    def build_model(self) -> MLPClassifier:
        """Build and configure an MLPClassifier based on the configuration.

        The model architecture and training parameters are read from the configuration.
        The MLP uses ReLU activation and Adam optimizer by default, with optional
        early stopping and learning rate adaptation.

        Returns:
            MLPClassifier: A configured but untrained MLPClassifier instance

        Configuration Parameters:
            hidden_layer_sizes: List of layer sizes (e.g., [100, 50] for two hidden layers)
            max_iter: Maximum number of iterations (epochs) for training
            random_state: Random seed for reproducibility
            early_stop_number: Number of epochs to wait before early stopping
            learning_rate_init: Initial learning rate for the optimizer

        Example:
            >>> config = {
            ...     "hidden_layer_sizes": [100, 50],
            ...     "max_iter": 100,
            ...     "random_state": 42,
            ...     "early_stop_number": 5,
            ...     "learning_rate_init": 0.001
            ... }
            >>> model = build_model()
            >>> type(model)
            <class 'sklearn.neural_network._multilayer_perceptron.MLPClassifier'>
        """
        max_iter = self.model_training_config["max_iter"]
        random_state = self.model_training_config["random_state"]
        hidden_layer_sizes = self.model_training_config["hidden_layer_sizes"]
        early_stop_number = self.model_training_config["early_stop_number"]
        learning_rate_init = self.model_training_config["learning_rate_init"]
        train_ratio = self.data_ingestion_config["train_ratio"]

        logger.info(
            f"Building MLPClassifier with hidden layers: {hidden_layer_sizes}, "
            f"max_iter: {max_iter}, random_state: {random_state}, "
            f"learning_rate: {learning_rate_init}"
        )

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=max_iter,
            early_stopping=True,  # Enable early stopping to prevent overfitting
            n_iter_no_change=early_stop_number,  # Stop if no improvement for N epochs
            validation_fraction=1
            - train_ratio,  # Use (1-train_ratio) of training data for validation
            nesterovs_momentum=True,  # Use Nesterov's momentum for better convergence
            learning_rate="adaptive",  # Automatically adjust learning rate
            learning_rate_init=learning_rate_init,  # Initial learning rate
        )

        return model

    def train(
        self, model: MLPClassifier, train_data: pd.DataFrame, val_data: pd.DataFrame
    ) -> MLPClassifier:
        """Train the model on the provided training data with validation monitoring.

        The method separates features from the target variable and fits the model.
        Combines training and validation data, letting scikit-learn handle the split
        based on the validation_fraction parameter set in build_model().

        Args:
            model: The MLPClassifier model to be trained
            train_data: DataFrame containing both features and target ('output' column)
            val_data: Validation dataset for monitoring training progress

        Returns:
            The trained model (same as input model, but fitted)

        Note:
            - The model is modified in-place, but we return it for method chaining
            - Training progress and final metrics are logged
            - Early stopping is applied if no improvement is seen
        """
        logger.info(f"Starting model training with {len(train_data)} samples")
        # Separate features and target
        X_train = train_data.drop(columns=["output"])
        y_train = train_data["output"]

        logger.debug(
            f"Training data shape - Features: {X_train.shape}, Target: {y_train.shape}"
        )

        # Prepare validation data
        X_val = val_data.drop(columns=["output"])
        y_val = val_data["output"]

        # Combine train and validation data since scikit-learn will handle the split
        X_combined = pd.concat([X_train, X_val])
        y_combined = pd.concat([y_train, y_val])

        logger.debug(
            f"Training on {len(X_train)} samples, "
            f"validating on {len(X_val)} samples"
        )

        # Train the model with validation
        # Note: validation_fraction is set in build_model() to use the correct ratio
        model.fit(X_combined, y_combined)

        # Log training completion with appropriate metrics
        if hasattr(model, "best_validation_score_"):
            logger.info(
                f"Training completed. Best validation score: "
                f"{model.best_validation_score_:.4f}"
            )
        elif hasattr(model, "best_loss_"):
            logger.info(f"Training completed. Best loss: {model.best_loss_:.4f}")
        else:
            logger.info("Training completed.")

        return model

    def evaluate(
        self, model: MLPClassifier, test_data: pd.DataFrame
    ) -> Tuple[float, float]:
        """Evaluate the model's performance on the test dataset.

        Computes and logs several evaluation metrics:
        - Standard accuracy (exact match)
        - Top-k accuracy (true class is in top-k predicted classes)
        - Detailed classification report with precision, recall, and f1-score

        Args:
            model: Trained MLPClassifier model to evaluate
            test_data: Test dataset containing features and target variable

        Returns:
            A tuple containing:
                - accuracy: Standard accuracy score (exact match)
                - top_k_accuracy: Top-k accuracy score

        Note:
            - The top-k value is read from the model configuration
            - Logs comprehensive evaluation metrics to the logger
            - Handles potential class imbalance through the classification report
        """
        logger.info("Evaluating model on test data...")

        # Get configuration
        top_k = self.model_training_config.get("top_k", 5)

        # Prepare features and target
        X_test = test_data.drop(columns=["output"])
        y_test = test_data["output"]

        logger.debug(f"Evaluating model on {len(X_test)} test samples")

        # Get model predictions and probabilities
        logger.debug("Generating predictions...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Calculate standard accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate top-k accuracy
        model_classes = model.classes_  # Classes the model was trained on

        # Convert true labels to indices in the model's classes
        y_true_indices = np.searchsorted(model_classes, y_test)

        # Get indices of top-k predicted classes for each sample
        # Using argpartition for better performance than full argsort
        top_k_pred_indices = np.argpartition(y_proba, -top_k, axis=1)[:, -top_k:]

        # Check if true class is in top-k predictions
        correct = np.any(top_k_pred_indices == y_true_indices.reshape(-1, 1), axis=1)
        top_k_accuracy = np.mean(correct)

        # Log metrics
        logger.info(f"\n=== Evaluation Results ===")
        logger.info(f"Standard Accuracy: {accuracy:.4f}")
        logger.info(f"Top-{top_k} Accuracy: {top_k_accuracy:.4f}")

        # Generate and log detailed classification report
        logger.info("\n=== Classification Report ===")
        logger.info(classification_report(y_test, y_pred, zero_division=0))

        return accuracy, top_k_accuracy

    def save(self, model: MLPClassifier) -> None:
        """Save the trained model to disk using joblib compression.

        The model is saved in the directory specified by model_output_dir with
        the name specified by model_name. Uses LZMA compression for efficient storage.

        Args:
            model: The trained MLPClassifier model to be saved

        Side Effects:
            - Creates or overwrites the model file at model_output_dir/model_name.joblib
            - Updates model_output_path with the full path to the saved model

        Note:
            The model is compressed using LZMA with compression level 3,
            which provides a good balance between compression ratio and speed.
        """
        logger.info("Saving the trained model...")
        self.model_output_path = self.model_output_dir / f"{self.model_name}.joblib"
        joblib.dump(model, self.model_output_path, compress=("lzma", 3))
        logger.info(f"Model saved successfully to {self.model_output_path}")

    def run(self) -> None:
        """Execute the complete model training and evaluation pipeline.

        The pipeline consists of the following steps:
        1. Load training, validation, and test data
        2. Initialize the model with configured parameters
        3. Train the model using training and validation data
        4. Evaluate the model on the test data
        5. Log all results and metrics

        The method handles the complete workflow and includes:
        - Progress logging at each major step
        - Error handling with detailed error messages
        - Resource cleanup in case of failures
        - Comprehensive logging of results and metrics

        Example:
            >>> from src.model_training import ModelTraining
            >>> from src.utils.config import read_config
            >>>
            >>> # Load configuration
            >>> config = read_config("config/config.yaml")
            >>>
            >>> # Initialize and run training
            >>> model_trainer = ModelTraining(config)
            >>> model_trainer.run()

        Note:
            - The method uses the module's logger for all output
            - All configuration is read from the config file
            - The pipeline is wrapped in try-except to ensure proper error handling
            - Final metrics are logged before completion
        """
        try:
            mlflow.set_experiment("reedsshepp_mlops")
            with mlflow.start_run():
                # Initialize MLflow tracking and log configuration
                logger.info("=== Starting Model Training Pipeline ===")
                logger.info("Initializing MLflow experiment tracking...")
                mlflow.set_tag("model_type", self.model_name)
                mlflow.log_params(self.model_training_config)
                logger.info(f"Tracking experiment with model: {self.model_name}")

                logger.info("Loading training, validation, and test data...")
                train_data, val_data, test_data = self.load_data()
                mlflow.log_artifact(self.train_path, "datasets")
                mlflow.log_artifact(self.val_path, "datasets")
                mlflow.log_artifact(self.test_path, "datasets")

                # Model Building
                logger.info("\nBuilding model...")
                model = self.build_model()

                # Model Training
                logger.info("\nTraining model...")
                model = self.train(model, train_data, val_data)

                # Model Evaluation
                logger.info("\nEvaluating model on test data...")
                accuracy, top_k_accuracy = self.evaluate(model, test_data)

                # Log final results and metrics
                logger.info("\n=== Training Summary ===")
                final_metrics = {
                    "final_accuracy": accuracy,
                    f"top_{self.model_training_config.get('top_k', 5)}_accuracy": top_k_accuracy,
                }

                for name, value in final_metrics.items():
                    logger.info(f"{name.replace('_', ' ').title()}: {value:.4f}")
                    mlflow.log_metric(name, value)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("top_k_accuracy", top_k_accuracy)

                self.save(model)

                mlflow.log_artifact(self.model_output_path, "models")
                params = model.get_params()
                mlflow.log_params(params)
                logger.info("MLflow completed successfully")

                logger.info("Model training and evaluation completed successfully!")

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            logger.exception("Detailed error:")
            raise

        finally:
            logger.info("=== Model Training Pipeline Completed ===\n")
