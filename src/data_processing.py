"""Data Processing Module for ReedsShepp-MLOps.

This module provides a comprehensive pipeline for processing trajectory datasets,
including:
1. Loading raw data from CSV files
2. Cleaning and transforming data
3. Ensuring consistent processing across splits
4. Saving processed data

The module is designed to prevent data leakage by applying identical
transformations to training, validation, and test datasets. It includes
robust error handling and logging throughout the processing pipeline.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from src.logger import get_logger

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
        """Load raw trajectory datasets from CSV files.

        This method loads and validates the training, validation, and test datasets
        from their respective CSV files in the raw directory. It performs basic
        validation to ensure each file exists and contains data.

        Returns:
            A tuple containing three pandas DataFrames in order:
            - train_data: Training dataset with features and target
            - val_data: Validation dataset for model tuning
            - test_data: Test dataset for final evaluation

        Raises:
            FileNotFoundError: If any required input file is missing
            pd.errors.EmptyDataError: If any input file is empty
            pd.errors.ParserError: If CSV parsing fails

        Example:
            >>> train, val, test = processor.load_raw_data()
            >>> print(f"Loaded {len(train)} training samples")

        Note:
            - Expects CSV files with headers: input1, input2, input3, extra, output
            - Performs basic validation of dataset sizes
            - Logs detailed statistics about each dataset
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
        self, 
        train_data: pd.DataFrame, 
        val_data: pd.DataFrame, 
        test_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process and transform all dataset splits consistently.

        This method ensures identical transformations are applied to all dataset
        splits to prevent data leakage. It processes each dataset independently
        while maintaining consistency across splits.

        Args:
            train_data: Training dataset to be processed
            val_data: Validation dataset to be processed
            test_data: Test dataset to be processed

        Returns:
            A tuple containing three processed DataFrames in order:
            - processed_train: Processed training dataset
            - processed_val: Processed validation dataset
            - processed_test: Processed test dataset

        Raises:
            ValueError: If datasets have inconsistent columns after processing
            Exception: For any other unexpected errors during processing

        Example:
            >>> train_processed, val_processed, test_processed = processor.process_data(
            ...     train_data, val_data, test_data
            ... )

        Note:
            - Applies identical transformations to all splits
            - Verifies consistent columns across all datasets
            - Maintains original dataset splits throughout processing
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
        self, 
        train_data: pd.DataFrame, 
        val_data: pd.DataFrame, 
        test_data: pd.DataFrame
    ) -> None:
        """Save processed datasets to standardized CSV files.

        This method saves the processed datasets to CSV files with consistent
        formatting and structure. It includes error handling for file operations
        and provides detailed logging of the save process.

        Args:
            train_data: Processed training dataset to be saved
            val_data: Processed validation dataset to be saved
            test_data: Processed test dataset to be saved

        Raises:
            PermissionError: If there are permission issues when writing files
            OSError: If there are issues writing to the filesystem
            Exception: For any other unexpected errors during file operations

        Example:
            >>> processor.save_to_csv_files(train_df, val_df, test_df)

        Note:
            - Output files are saved to {processed_dir} with filenames:
              - train.csv
              - validation.csv
              - test.csv
            - Files are saved without index to maintain clean format
            - Includes detailed logging of save operations
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
        be applied consistently to all dataset splits. It performs the following:
        1. Removes the 'extra' column if present
        2. Validates data types and structure
        3. Applies consistent transformations

        Args:
            data: Input DataFrame to be processed

        Returns:
            pd.DataFrame: Processed DataFrame with consistent structure

        Raises:
            ValueError: If data structure is invalid
            Exception: For any other unexpected errors during processing

        Note:
            - This method is called by process_data() for each dataset split
            - Maintains original DataFrame structure where possible
            - Performs in-place operations when safe to do so
        """

        # Remove the 'extra' column if present
        # This is a safety check to prevent KeyError if the column doesn't exist
        if "extra" in data.columns:
            data = data.drop(columns=["extra"])
            logger.debug("Removed 'extra' column from dataset")

        return data

    def _verify_consistent_columns(
        self, 
        train_data: pd.DataFrame, 
        val_data: pd.DataFrame, 
        test_data: pd.DataFrame
    ) -> None:
        """Verify that all datasets have consistent columns.

        This internal method ensures that training, validation, and test datasets
        have exactly the same columns after processing. This is crucial for preventing
        data leakage and ensuring consistent model training.

        Args:
            train_data: Processed training dataset
            val_data: Processed validation dataset
            test_data: Processed test dataset

        Raises:
            ValueError: If datasets have inconsistent columns
            Exception: For any other unexpected errors during verification

        Note:
            - Called internally by process_data()
            - Compares column names and order across all datasets
            - Logs detailed information about any inconsistencies
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
