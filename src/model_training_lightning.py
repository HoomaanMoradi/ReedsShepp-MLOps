"""Model Training Module for MLP using PyTorch Lightning.

This module provides a comprehensive pipeline for training and evaluating a Multi-layer 
Perceptron (MLP) classifier using PyTorch Lightning, while maintaining the same
method structure as the original scikit-learn implementation. It handles the complete 
machine learning workflow including data loading, model configuration, training with 
early stopping, and performance evaluation with multiple metrics.

The module is designed to be configurable through YAML configuration files and includes 
MLflow integration for experiment tracking and model versioning.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy

from src.logger import get_logger

logger = get_logger(__name__)


# This is the PyTorch Lightning equivalent of sklearn's MLPClassifier
class MLP(pl.LightningModule):
    """PyTorch Lightning module for the Multi-Layer Perceptron."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_layer_sizes: List[int],
        learning_rate: float,
        top_k: int,
    ):
        super().__init__()
        # save_hyperparameters() allows us to access them later via self.hparams
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()

        # Define metrics
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.test_top_k_acc = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=top_k
        )

        # To generate a classification report at the end of testing
        self.test_step_outputs = []

        # Build the network layers dynamically
        layers = []
        in_features = self.hparams.input_size
        for size in self.hparams.hidden_layer_sizes:
            layers.append(nn.Linear(in_features, size))
            layers.append(nn.ReLU())
            in_features = size
        layers.append(nn.Linear(in_features, self.hparams.num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        # We monitor "val_accuracy" for early stopping and model checkpointing
        self.log("val_accuracy", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.test_acc.update(logits, y)
        self.test_top_k_acc.update(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.test_step_outputs.append({"preds": preds, "targets": y})

    def on_test_epoch_end(self):
        # Log final test metrics
        self.log("final_accuracy", self.test_acc.compute())
        self.log(
            f"top_{self.hparams.top_k}_accuracy", self.test_top_k_acc.compute()
        )

        # Generate and log classification report
        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs]).cpu()
        all_targets = torch.cat([x["targets"] for x in self.test_step_outputs]).cpu()
        report = classification_report(all_targets, all_preds, zero_division=0)
        logger.info("\n=== Classification Report ===")
        logger.info(f"\n{report}")
        self.test_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class ModelTraining:
    """A class to manage the complete MLP model training pipeline.

    This class orchestrates the training workflow using PyTorch Lightning,
    but exposes an interface identical to the original scikit-learn version.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the ModelTraining with configuration."""
        self.model_training_config = config["model_training_lightning"]
        self.data_ingestion_config = config["data_ingestion"]
        artifact_dir = Path(self.data_ingestion_config["artifact_dir"])
        self.processed_dir = artifact_dir / "processed"
        self.model_output_dir = artifact_dir / "models"
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = self.model_training_config["model_name"]
        
        self.train_path: Optional[Path] = None
        self.val_path: Optional[Path] = None
        self.test_path: Optional[Path] = None
        self.model_output_path: Optional[Path] = None
        self.best_model_checkpoint_path: Optional[str] = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load training, validation, and test data from processed CSV files."""
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

    def build_model(self, train_data: pd.DataFrame) -> MLP:
        """Build and configure a PyTorch Lightning MLP model based on the configuration."""
        # Infer input_size and num_classes from the training data
        input_size = len(train_data.drop(columns=["output"]).columns)
        num_classes = train_data["output"].nunique()

        logger.info(
            f"Building PyTorch Lightning MLP with input_size: {input_size}, "
            f"num_classes: {num_classes}, "
            f"hidden layers: {self.model_training_config['hidden_layer_sizes']}, "
            f"learning_rate: {self.model_training_config['learning_rate_init']}"
        )

        model = MLP(
            input_size=input_size,
            num_classes=num_classes,
            hidden_layer_sizes=self.model_training_config["hidden_layer_sizes"],
            learning_rate=self.model_training_config["learning_rate_init"],
            top_k=self.model_training_config.get("top_k", 5),
        )
        return model

    def train(self, model: MLP, train_data: pd.DataFrame, val_data: pd.DataFrame, mlf_logger: MLFlowLogger) -> MLP:
        """Train the model on the provided training data with validation monitoring."""
        logger.info(f"Starting model training with {len(train_data)} samples")

        # Prepare PyTorch Datasets and DataLoaders
        X_train = torch.tensor(train_data.drop("output", axis=1).values, dtype=torch.float32)
        y_train = torch.tensor(train_data["output"].values, dtype=torch.long)
        train_dataset = TensorDataset(X_train, y_train)

        X_val = torch.tensor(val_data.drop("output", axis=1).values, dtype=torch.float32)
        y_val = torch.tensor(val_data["output"].values, dtype=torch.long)
        val_dataset = TensorDataset(X_val, y_val)

        batch_size = self.model_training_config.get("batch_size", 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
        
        # Configure callbacks
        early_stop_callback = EarlyStopping(
            monitor="val_accuracy",
            patience=self.model_training_config["early_stop_number"],
            mode="max",
            verbose=True
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_output_dir,
            filename=f"{self.model_name}-best-checkpoint",
            save_top_k=1,
            monitor="val_accuracy",
            mode="max"
        )

        # Initialize the Trainer
        trainer = pl.Trainer(
            max_epochs=self.model_training_config["max_iter"],
            logger=mlf_logger,
            callbacks=[early_stop_callback, checkpoint_callback],
            accelerator="auto",
        )

        # Train the model
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        self.best_model_checkpoint_path = checkpoint_callback.best_model_path
        logger.info(f"Training completed. Best model saved at {self.best_model_checkpoint_path}")

        # Load the best model weights before returning
        trained_model = MLP.load_from_checkpoint(self.best_model_checkpoint_path)
        return trained_model

    def evaluate(self, model: MLP, test_data: pd.DataFrame, mlf_logger: MLFlowLogger) -> Tuple[float, float]:
        """Evaluate the model's performance on the test dataset."""
        logger.info("Evaluating model on test data...")

        # Prepare Test DataLoader
        X_test = torch.tensor(test_data.drop("output", axis=1).values, dtype=torch.float32)
        y_test = torch.tensor(test_data["output"].values, dtype=torch.long)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.model_training_config.get("batch_size", 32), num_workers=4)

        trainer = pl.Trainer(logger=mlf_logger, accelerator="auto")
        results = trainer.test(model, dataloaders=test_loader, verbose=False)

        accuracy = results[0]["final_accuracy"]
        top_k_accuracy = results[0][f"top_{model.hparams.top_k}_accuracy"]

        logger.info(f"\n=== Evaluation Results ===")
        logger.info(f"Standard Accuracy: {accuracy:.4f}")
        logger.info(f"Top-{model.hparams.top_k} Accuracy: {top_k_accuracy:.4f}")
        
        return accuracy, top_k_accuracy

    def save(self, model: MLP) -> None:
        """Save the trained model's state dictionary to disk."""
        logger.info("Saving the final model state...")
        self.model_output_path = self.model_output_dir / f"{self.model_name}.pth"
        torch.save(model.state_dict(), self.model_output_path)
        logger.info(f"Model state_dict saved successfully to {self.model_output_path}")

    def run(self) -> None:
        """Execute the complete model training and evaluation pipeline."""
        try:
            mlflow.set_experiment("reedsshepp_mlops_pytorch")
            mlf_logger = MLFlowLogger(
                experiment_name="reedsshepp_mlops_pytorch",
                tracking_uri=mlflow.get_tracking_uri(),
            )
            
            with mlflow.start_run(run_id=mlf_logger.run_id):
                logger.info("=== Starting Model Training Pipeline (PyTorch Lightning) ===")
                mlflow.set_tag("model_type", self.model_name)
                mlflow.log_params(self.model_training_config)

                pl.seed_everything(self.model_training_config["random_state"], workers=True)

                logger.info("Loading training, validation, and test data...")
                train_data, val_data, test_data = self.load_data()
                
                # Encode target labels to be 0-indexed for PyTorch ---
                # PyTorch's CrossEntropyLoss expects target labels to be in the range [0, num_classes-1].
                # Scikit-learn handles this automatically, but in PyTorch we must do it manually.
                logger.info("Encoding target labels to be 0-indexed for PyTorch...")
                label_encoder = LabelEncoder()
                
                # Fit on the training labels and transform them
                train_data['output'] = label_encoder.fit_transform(train_data['output'])
                
                # Use the SAME fitted encoder to transform validation and test labels
                val_data['output'] = label_encoder.transform(val_data['output'])
                test_data['output'] = label_encoder.transform(test_data['output'])
                
                logger.info(f"Label encoding complete. {len(label_encoder.classes_)} classes mapped.")
                # --- END FIX ---

                mlflow.log_artifact(str(self.train_path), "datasets")
                mlflow.log_artifact(str(self.val_path), "datasets")
                mlflow.log_artifact(str(self.test_path), "datasets")

                logger.info("\nBuilding model...")
                model = self.build_model(train_data)

                logger.info("\nTraining model...")
                model = self.train(model, train_data, val_data, mlf_logger)

                logger.info("\nEvaluating model on test data...")
                accuracy, top_k_accuracy = self.evaluate(model, test_data, mlf_logger)

                logger.info("\n=== Training Summary ===")
                logger.info(f"Final Accuracy: {accuracy:.4f}")
                logger.info(f"Top {self.model_training_config.get('top_k', 5)} Accuracy: {top_k_accuracy:.4f}")

                self.save(model)
                mlflow.log_artifact(str(self.model_output_path), "models")

                if self.best_model_checkpoint_path:
                    mlflow.log_artifact(self.best_model_checkpoint_path, "models")
                
                logger.info("MLflow logging completed successfully")
                logger.info("Model training and evaluation completed successfully!")

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            logger.exception("Detailed error:")
            raise

        finally:
            logger.info("=== Model Training Pipeline Completed ===\n")