from pathlib import Path

import pandas as pd

from logger import get_logger

logger = get_logger(__name__)


class DataProcessing:
    """
    A class for processing raw datasets by removing unnecessary columns and preparing them for model training.

    The class handles loading raw data, processing it by removing specified columns, and saving
    the processed data to the appropriate directories.

    Attributes
    ----------
    raw_dir : Path
        Directory containing the raw input data files
    processed_dir : Path
        Directory where processed data will be saved
    """

    def __init__(self, config):
        """
        Initialize the DataProcessing class with configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing data paths and processing parameters.
            Expected to have 'data_ingestion' key with 'artifact_dir'.
        """
        # Set up directories for raw and processed data
        artifact_dir = Path(config["data_ingestion"]["artifact_dir"])
        self.raw_dir = artifact_dir / "raw"
        self.processed_dir = artifact_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self):
        """
        Load the raw datasets from CSV files.

        Returns
        -------
        tuple
            A tuple containing three pandas DataFrames in the following order:
            - train_data: Training dataset
            - val_data: Validation dataset
            - test_data: Test dataset

        Notes
        -----
        Assumes the following files exist in the raw directory:
        - train.csv
        - validation.csv
        - test.csv
        """
        train_data = pd.read_csv(self.raw_dir / "train.csv")
        val_data = pd.read_csv(self.raw_dir / "validation.csv")
        test_data = pd.read_csv(self.raw_dir / "test.csv")
        return train_data, val_data, test_data

    def process_data(self, train_data, val_data, test_data):
        """
        Apply consistent processing steps to all dataset splits.

        This ensures that the same transformations are applied to training,
        validation, and test datasets to prevent data leakage.

        Parameters
        ----------
        train_data : pd.DataFrame
            Training dataset to be processed
        val_data : pd.DataFrame
            Validation dataset to be processed
        test_data : pd.DataFrame
            Test dataset to be processed

        Returns
        -------
        tuple
            A tuple containing the three processed datasets in the same order:
            (processed_train, processed_val, processed_test)

        Parameters
        ----------
        train_data : pd.DataFrame
            Training dataset
        val_data : pd.DataFrame
            Validation dataset
        test_data : pd.DataFrame
            Test dataset

        Returns
        -------
        tuple
            Processed train, validation, and test dataframes
        """
        train_data = self._process_single_dataset(train_data)
        val_data = self._process_single_dataset(val_data)
        test_data = self._process_single_dataset(test_data)

        return train_data, val_data, test_data

    def save_to_csv_files(self, train_data, val_data, test_data):
        """
        Save the processed datasets to CSV files.

        The processed datasets are saved in the processed directory with the same
        base filenames as the input files.

        Parameters
        ----------
        train_data : pd.DataFrame
            Processed training dataset to be saved
        val_data : pd.DataFrame
            Processed validation dataset to be saved
        test_data : pd.DataFrame
            Processed test dataset to be saved

        Notes
        -----
        Output files will be saved as:
        - {processed_dir}/train.csv
        - {processed_dir}/validation.csv
        - {processed_dir}/test.csv
        """
        train_data.to_csv(self.processed_dir / "train.csv", index=False)
        val_data.to_csv(self.processed_dir / "validation.csv", index=False)
        test_data.to_csv(self.processed_dir / "test.csv", index=False)
        logger.info(f"Saved processed files to {self.processed_dir}")

    def _process_single_dataset(self, data):
        """
        Process a single dataset by removing unnecessary columns.

        This internal method handles the core data cleaning operations.
        Currently removes the 'extra' column if it exists in the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset to be processed

        Returns
        -------
        pd.DataFrame
            Processed dataset with specified columns removed

        Example
        -------
        >>> processor = DataProcessing(config)
        >>> processed_data = processor._process_single_dataset(raw_data)
        """
        # Remove the 'extra' column if it exists in the dataset
        # This is a safety check to prevent KeyError if the column doesn't exist
        if "extra" in data.columns:
            data = data.drop(columns=["extra"])

        return data

    def run(self):
        """Execute the complete data processing pipeline.

        This method coordinates the entire data processing workflow:
        1. Loads the raw datasets
        2. Applies consistent processing to each dataset
        3. Saves the processed data to disk

        The method logs the start and completion of the processing pipeline.

        Examples
        --------
        >>> from config import read_config
        >>> config = read_config("config/config.yaml")
        >>> processor = DataProcessing(config)
        >>> processor.run()

        Notes
        -----
        This is the main entry point for the data processing pipeline
        and should be called after initializing the DataProcessing class.
        """
        logger.info("Starting data processing pipeline")

        # Load raw datasets
        logger.debug("Loading raw datasets")
        train_data, val_data, test_data = self.load_raw_data()

        # Process all datasets
        logger.debug("Processing datasets")
        processed_train_data, processed_val_data, processed_test_data = (
            self.process_data(train_data, val_data, test_data)
        )

        # Save processed data
        logger.debug("Saving processed datasets")
        self.save_to_csv_files(
            processed_train_data, processed_val_data, processed_test_data
        )

        logger.info("Data processing completed successfully")
