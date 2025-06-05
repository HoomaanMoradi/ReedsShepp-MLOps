"""Data Processing Module for ReedsShepp-MLOps.

This module provides functionality for processing raw trajectory datasets into a format
suitable for machine learning model training. It handles data loading, cleaning,
and preparation while ensuring consistent processing across training, validation,
and test datasets to prevent data leakage.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from logger import get_logger

# Initialize module logger
logger = get_logger(__name__)


class DataProcessing:
    """Process and prepare trajectory datasets for machine learning tasks.

    This class provides a complete pipeline for processing raw trajectory data,
    including loading, cleaning, and preparing datasets for model training.
    It ensures consistent processing across training, validation, and test sets
    to prevent data leakage and maintain data integrity.

    Attributes:
        raw_dir (Path): Directory containing the raw input data files.
        processed_dir (Path): Directory where processed data will be saved.

    Example:
        >>> from config_reader import read_config
        >>> config = read_config("config/config.yaml")
        >>> processor = DataProcessing(config)
        >>> processor.run()  # Runs complete processing pipeline
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the DataProcessing pipeline with configuration.

        Sets up the necessary directory structure for raw and processed data.
        Creates the processed directory if it doesn't exist.

        Args:
            config: Configuration dictionary containing:
                - data_ingestion.artifact_dir: Base directory for data artifacts
                - data_ingestion.raw_data_dir: Relative path to raw data directory
                - data_ingestion.processed_data_dir: Relative path for processed data

        Raises:
            KeyError: If required configuration keys are missing
            OSError: If directories cannot be created
        """
        try:
            # Set up base directories from configuration
            artifact_dir = Path(config["data_ingestion"]["artifact_dir"])
            self.raw_dir = artifact_dir / "raw"
            self.processed_dir = artifact_dir / "processed"

            # Ensure output directory exists
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(
                f"Initialized DataProcessing with raw_dir={self.raw_dir}, "
                f"processed_dir={self.processed_dir}"
            )

        except KeyError as e:
            logger.error(f"Missing required configuration: {e}")
            raise
        except OSError as e:
            logger.error(f"Failed to create directory structure: {e}")
            raise

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw datasets from CSV files in the raw directory.

        Loads the training, validation, and test datasets from their respective
        CSV files. This is the first step in the data processing pipeline.

        Returns:
            A tuple containing three pandas DataFrames in order:
            - train_data: Training dataset with features and target
            - val_data: Validation dataset for model tuning
            - test_data: Test dataset for final evaluation

        Raises:
            FileNotFoundError: If any required input file is missing
            pd.errors.EmptyDataError: If any input file is empty

        Example:
            >>> train, val, test = processor.load_raw_data()
            >>> print(f"Loaded {len(train)} training samples")
        """
        try:
            logger.info(f"Loading raw datasets from {self.raw_dir}")

            # Load each dataset with basic validation
            train_data = pd.read_csv(self.raw_dir / "train.csv")
            val_data = pd.read_csv(self.raw_dir / "validation.csv")
            test_data = pd.read_csv(self.raw_dir / "test.csv")

            # Log basic statistics
            logger.debug(
                f"Loaded datasets - Train: {len(train_data)} rows, "
                f"Validation: {len(val_data)} rows, Test: {len(test_data)} rows"
            )

            return train_data, val_data, test_data

        except FileNotFoundError as e:
            logger.error(f"Missing input file: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty dataset encountered: {e}")
            raise

    def process_data(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Apply consistent processing steps to all dataset splits.

        Ensures identical transformations are applied to training, validation,
        and test datasets to prevent data leakage. This method serves as a wrapper
        around _process_single_dataset to maintain consistency across all splits.

        Args:
            train_data: Training dataset to be processed
            val_data: Validation dataset to be processed
            test_data: Test dataset to be processed

        Returns:
            A tuple containing the three processed datasets in order:
            (processed_train, processed_val, processed_test)

        Example:
            >>> train_processed, val_processed, test_processed = processor.process_data(
            ...     train_data, val_data, test_data
            ... )
        """
        logger.info("Processing dataset splits")

        # Process each dataset split individually
        train_processed = self._process_single_dataset(train_data)
        val_processed = self._process_single_dataset(val_data)
        test_processed = self._process_single_dataset(test_data)

        # Verify all datasets have the same columns
        self._verify_consistent_columns(train_processed, val_processed, test_processed)

        return train_processed, val_processed, test_processed

    def save_to_csv_files(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> None:
        """Save processed datasets to CSV files in the processed directory.

        Saves the processed training, validation, and test datasets to CSV files
        with the same base names as the input files but in the processed directory.

        Args:
            train_data: Processed training dataset to be saved
            val_data: Processed validation dataset to be saved
            test_data: Processed test dataset to be saved

        Raises:
            PermissionError: If there are permission issues when writing files
            OSError: If there are issues writing to the filesystem

        Example:
            >>> processor.save_to_csv_files(train_df, val_df, test_df)

        Note:
            Output files will be saved as:
            - {processed_dir}/train.csv
            - {processed_dir}/validation.csv
            - {processed_dir}/test.csv
        """
        try:
            # Define output file paths
            # Save each dataset with index=False to avoid adding an extra column
            train_data.to_csv(self.processed_dir / "train.csv", index=False)
            val_data.to_csv(self.processed_dir / "validation.csv", index=False)
            test_data.to_csv(self.processed_dir / "test.csv", index=False)

            logger.info(f"Saved processed datasets to {self.processed_dir}")

        except (PermissionError, OSError) as e:
            logger.error(f"Failed to save processed data: {e}")
            raise

    def _process_single_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process a single dataset by applying cleaning and transformation steps.

        This internal method handles the core data processing operations that should
        be applied consistently to all dataset splits. Currently, it removes the 'extra'
        column if it exists in the dataset.

        Args:
            data: Input DataFrame to be processed

        Returns:
            Processed DataFrame with specified transformations applied

        Example:
            >>> processed = processor._process_single_dataset(raw_data)
            >>> 'extra' in processed.columns
            False

        Note:
            This method should be extended with additional processing steps as needed.
            All transformations should be stateless and not depend on the full dataset
            to prevent data leakage.
        """

        # Remove the 'extra' column if present
        # This is a safety check to prevent KeyError if the column doesn't exist
        if "extra" in data.columns:
            data = data.drop(columns=["extra"])
            logger.debug("Removed 'extra' column from dataset")

        return data

    def _verify_consistent_columns(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> None:
        """Verify that all datasets have the same columns.

        This internal method checks for consistency in column names across all
        dataset splits. It raises a ValueError if any discrepancies are found.

        Args:
            train_data: Training dataset to verify
            val_data: Validation dataset to verify
            test_data: Test dataset to verify

        Raises:
            ValueError: If column names are inconsistent across datasets
        """
        # Get column names from each dataset
        train_cols = set(train_data.columns)
        val_cols = set(val_data.columns)
        test_cols = set(test_data.columns)

        # Check for consistency
        if train_cols != val_cols or train_cols != test_cols:
            raise ValueError("Inconsistent column names across datasets")

    def run(self) -> None:
        """Execute the complete data processing pipeline.

        This is the main entry point that orchestrates the entire data processing workflow:
        1. Loads the raw datasets from the raw directory
        2. Applies consistent processing to each dataset split
        3. Saves the processed data to the processed directory

        The method includes comprehensive error handling and logging to track
        the progress and status of the processing pipeline.

        Raises:
            RuntimeError: If any step in the pipeline fails

        Example:
            >>> from config_reader import read_config
            >>> config = read_config("config/config.yaml")
            >>> processor = DataProcessing(config)
            >>> processor.run()

        Note:
            This method should be called after initializing the DataProcessing class.
            It's designed to be idempotent, meaning it can be safely run multiple times.
        """
        try:
            logger.info("Starting data processing pipeline")

            # 1. Load raw datasets
            logger.debug("Loading raw datasets")
            train_data, val_data, test_data = self.load_raw_data()

            # 2. Process all datasets
            logger.info("Processing datasets")
            processed_train, processed_val, processed_test = self.process_data(
                train_data, val_data, test_data
            )

            # 3. Save processed data
            logger.info("Saving processed datasets")
            self.save_to_csv_files(processed_train, processed_val, processed_test)

            logger.info("Data processing pipeline completed successfully")

        except Exception as e:
            logger.error(f"Error in data processing pipeline: {e}")
            logger.exception("Stack trace:")
            raise RuntimeError(f"Data processing failed: {e}") from e
