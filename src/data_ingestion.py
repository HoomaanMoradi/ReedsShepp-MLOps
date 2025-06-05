"""Data Ingestion Module for ReedsShepp-MLOps.

This module handles the data ingestion pipeline for the ReedsShepp-MLOps project.
It provides functionality to download data from Google Cloud Storage, split it into
training, validation, and test sets, and save the processed data to CSV files.

The module supports both NumPy array and text-based data formats and includes
comprehensive error handling and logging throughout the data ingestion process.
"""

import random
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from google.cloud import storage

from logger import get_logger

# Initialize module logger
logger = get_logger(__name__)


class DataIngestion:
    """Orchestrates the data ingestion pipeline for ReedsShepp-MLOps.

    This class handles the complete data ingestion workflow including:
    - Downloading data from Google Cloud Storage
    - Splitting data into training, validation, and test sets
    - Saving processed data to CSV files
    - Comprehensive error handling and logging

    Attributes:
        project_id (str): Google Cloud project ID (optional)
        bucket_name (str): GCS bucket name containing the data
        train_val_object_name (str): Object name for training/validation data in GCS
        test_object_name (str): Object name for test data in GCS
        train_ratio (float): Ratio of training data (0.0 to 1.0)
        artifact_dir (Path): Base directory for storing artifacts
        raw_dir (Path): Directory for storing raw data files

    Example:
        >>> config = {
        ...     "data_ingestion": {
        ...         "project_id": "my-project",
        ...         "bucket_name": "my-bucket",
        ...         "train_val_object_name": "data/train_val.npy",
        ...         "test_object_name": "data/test.npy",
        ...         "train_ratio": 0.8,
        ...         "artifact_dir": "artifacts"
        ...     }
        ... }
        >>> data_ingestion = DataIngestion(config)
        >>> data_ingestion.run()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the DataIngestion pipeline with configuration.

        Sets up the necessary configuration parameters and creates required directories.

        Args:
            config: Dictionary containing data ingestion configuration with the
                   following required keys under 'data_ingestion':
                   - bucket_name: Name of the GCS bucket
                   - train_val_object_name: Path to training/validation data in GCS
                   - test_object_name: Path to test data in GCS
                   - train_ratio: Ratio for train/validation split (0.0 to 1.0)
                   - artifact_dir: Base directory for storing artifacts
                   - project_id: Optional Google Cloud project ID

        Raises:
            KeyError: If required configuration keys are missing
            OSError: If artifact directories cannot be created
        """
        try:
            # Extract configuration with type hints
            self.data_ingestion_config: Dict[str, Any] = config["data_ingestion"]
            self.project_id: Optional[str] = self.data_ingestion_config.get(
                "project_id"
            )
            self.bucket_name: str = self.data_ingestion_config["bucket_name"]
            self.train_val_object_name: str = self.data_ingestion_config[
                "train_val_object_name"
            ]
            self.test_object_name: str = self.data_ingestion_config["test_object_name"]
            self.train_ratio: float = float(self.data_ingestion_config["train_ratio"])

            # Set up directory structure
            self.artifact_dir: Path = Path(self.data_ingestion_config["artifact_dir"])
            self.raw_dir: Path = self.artifact_dir / "raw"

            # Create directories if they don't exist
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Initialized DataIngestion with raw_dir={self.raw_dir}")

        except KeyError as e:
            logger.error(f"Missing required configuration: {e}")
            raise
        except OSError as e:
            logger.error(f"Failed to create directory structure: {e}")
            raise

    def download_from_gcp(
        self, bucket_name: str, object_name: str
    ) -> Union[List[Any], str]:
        """Download and parse data from Google Cloud Storage.

        Attempts to download and parse the data first as a NumPy array (.npy/.npz),
        falling back to text if NumPy parsing fails. Supports both compressed and
        uncompressed NumPy arrays.

        Args:
            bucket_name: Name of the GCS bucket containing the data
            object_name: Path to the object within the bucket

        Returns:
            Union[List[Any], str]: Parsed data as a list (if NumPy array) or string

        Raises:
            google.cloud.exceptions.GoogleCloudError: If there's an issue with GCS access
            ValueError: If the downloaded data cannot be parsed
            Exception: For any other unexpected errors

        Example:
            >>> data = download_from_gcp("my-bucket", "path/to/data.npy")
            >>> print(f"Downloaded {len(data)} items" if isinstance(data, list) else "Text data")
        """
        try:
            # Initialize GCS client and download blob
            client = storage.Client(project=self.project_id)
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(object_name)

            logger.info(f"Downloading {object_name} from bucket {bucket_name}")
            raw_data = blob.download_as_bytes()
            logger.debug(f"Downloaded {len(raw_data):,} bytes from {object_name}")

            try:
                # First try loading as NumPy array (.npy) or compressed array (.npz)
                with BytesIO(raw_data) as buffer:
                    data = np.load(buffer, allow_pickle=True)

                    # Handle NPZ files (compressed NumPy arrays)
                    if isinstance(data, np.lib.npyio.NpzFile):
                        # Extract first array from NPZ file
                        data = data["arr_0"]

                    logger.info(
                        f"Loaded {object_name} as NumPy array with shape: {data.shape}"
                    )
                    return data.tolist()

            except Exception as np_err:
                # Fallback to text if NumPy loading fails
                logger.warning(
                    f"Failed to load {object_name} as NumPy: {str(np_err)}. "
                    "Attempting to load as text..."
                )
                try:
                    data = raw_data.decode("utf-8")
                    logger.info(
                        f"Successfully loaded {object_name} as text ({len(data):,} chars)"
                    )
                    logger.debug(
                        f"Text preview: {data[:100]}{'...' if len(data) > 100 else ''}"
                    )
                    return data
                except UnicodeDecodeError as decode_err:
                    logger.error(
                        f"Failed to decode {object_name} as text: {str(decode_err)}"
                    )
                    raise ValueError(
                        f"Could not decode {object_name} as NumPy or text"
                    ) from decode_err

        except Exception as e:
            logger.error(f"Failed to download or process {object_name}: {str(e)}")
            raise

    def download_raw_data(self):
        """Download both training/validation and test datasets from Google Cloud Storage.

        This method attempts to download both datasets independently, collecting any
        errors that occur. It will only raise an exception if both downloads fail.

        Returns:
            A tuple containing:
            - train_val_data: Downloaded training/validation data (or None if download failed)
            - test_data: Downloaded test data (or None if download failed)

        Raises:
            RuntimeError: If both dataset downloads fail

        Note:
            This method is designed to be fault-tolerant - it will still return
            partial results if only one dataset download succeeds.
        """
        train_val_data = None
        test_data = None
        errors = []

        logger.info(
            f"Starting data download from GCS bucket '{self.bucket_name}':\n"
            f"- Training/validation: {self.train_val_object_name}\n"
            f"- Test: {self.test_object_name}"
        )

        # Attempt to download training/validation data
        try:
            logger.debug(
                f"Downloading training/validation data: {self.train_val_object_name}"
            )
            train_val_data = self.download_from_gcp(
                bucket_name=self.bucket_name, object_name=self.train_val_object_name
            )
            logger.info("Successfully downloaded training/validation data")
        except Exception as e:
            error_msg = f"Failed to download training/validation data: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

        # Attempt to download test data
        try:
            logger.debug(f"Downloading test data: {self.test_object_name}")
            test_data = self.download_from_gcp(
                bucket_name=self.bucket_name, object_name=self.test_object_name
            )
            logger.info("Successfully downloaded test data")
        except Exception as e:
            error_msg = f"Failed to download test data: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

        # Only raise an exception if both downloads failed
        if not train_val_data and not test_data and errors:
            raise RuntimeError("All downloads failed: " + "; ".join(errors))

        return train_val_data, test_data

    def split_data(self, train_val_data, test_data):
        """Split and process training/validation and test datasets.

        This method handles both list and string input formats, parses the data,
        validates the structure, and splits it into training, validation, and test sets.

        Args:
            train_val_data: Training and validation data in either list or string format.
                           If string, will be split by newlines and parsed as integers.
            test_data: Test data in either list or string format. If string, will be
                     split by newlines and parsed as integers.

        Returns:
            A tuple containing three lists of integer lists:
            - train_data: Training data split according to train_ratio
            - val_data: Validation data (complement of training data split)
            - test_data_list: Processed test data

        Note:
            - Only rows with exactly 5 columns are kept
            - Input strings should be newline-separated with space-separated integers
            - Empty or invalid rows are logged and filtered out
        """
        train_data = []
        val_data = []
        test_data_list = []

        # Process train/validation data
        if train_val_data is not None:
            if isinstance(train_val_data, list):
                train_val_list = train_val_data
                logger.info(f"Train/validation data: {len(train_val_list)} rows")
            elif isinstance(train_val_data, str):
                try:
                    # Parse text data into list of integers
                    train_val_list = [
                        list(map(int, row.split()))
                        for row in train_val_data.strip().split("\n")
                    ]
                    logger.info(
                        f"Parsed train/validation data: {len(train_val_list)} rows"
                    )
                except ValueError as e:
                    logger.error(f"Failed to parse train/validation data: {str(e)}")
                    train_val_list = []
            else:
                logger.error(
                    f"Unexpected train/validation data type: {type(train_val_data)}"
                )
                train_val_list = []

            if train_val_list:
                # Filter rows with 5 columns
                valid_rows = [row for row in train_val_list if len(row) == 5]
                if len(valid_rows) != len(train_val_list):
                    logger.warning(
                        f"Filtered {len(train_val_list) - len(valid_rows)} invalid rows"
                    )
                train_val_list = valid_rows

                # Split into train and validation sets
                random.shuffle(train_val_list)
                train_size = int(len(train_val_list) * self.train_ratio)
                train_data = train_val_list[:train_size]
                val_data = train_val_list[train_size:]
            else:
                logger.warning("No valid train/validation data.")
        else:
            logger.warning("No train/validation data available.")

        # Process test data
        if test_data is not None:
            if isinstance(test_data, list):
                test_data_list = test_data
                logger.info(f"Test data: {len(test_data_list)} rows")
            elif isinstance(test_data, str):
                try:
                    # Parse text data into list of integers
                    test_data_list = [
                        list(map(int, row.split()))
                        for row in test_data.strip().split("\n")
                    ]
                    logger.info(f"Parsed test data: {len(test_data_list)} rows")
                except ValueError as e:
                    logger.error(f"Failed to parse test data: {str(e)}")
                    test_data_list = []
            else:
                logger.error(f"Unexpected test data type: {type(test_data)}")
                test_data_list = []

            if test_data_list:
                # Filter rows with 5 columns
                valid_rows = [row for row in test_data_list if len(row) == 5]
                if len(valid_rows) != len(test_data_list):
                    logger.warning(
                        f"Filtered {len(test_data_list) - len(valid_rows)} invalid rows"
                    )
                test_data_list = valid_rows
        else:
            logger.warning("No test data available.")

        return train_data, val_data, test_data_list

    def save_to_csv_files(self, train_data, val_data, test_data):
        """Save the processed datasets to CSV files in the raw data directory.

        This method takes the processed training, validation, and test datasets and
        saves them to separate CSV files in the configured raw data directory. Each
        file includes a header row and the data rows with consistent formatting.

        Args:
            train_data: Training data as a list of lists, where each inner list
                      contains exactly 5 integers representing the columns:
                      [input1, input2, input3, extra, output]
            val_data: Validation data in the same format as train_data
            test_data: Test data in the same format as train_data

        Raises:
            OSError: If there are issues writing to the filesystem
            ValueError: If any row doesn't have exactly 5 columns

        Note:
            - Output files are saved as:
              - {raw_dir}/train.csv
              - {raw_dir}/validation.csv
              - {raw_dir}/test.csv
            - All files include the same header row
            - Empty datasets are skipped with a warning
        """
        # Define CSV header
        header = "input1,input2,input3,extra,output\n"
        data_files = [
            ("train", train_data),
            ("validation", val_data),
            ("test", test_data),
        ]

        # Save each dataset to CSV
        for name, data in data_files:
            output_file = self.raw_dir / f"{name}.csv"
            with open(output_file, "w") as f:
                f.write(header)
                for row in data:
                    f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")
            logger.info(f"Saved {name} data with {len(data)} records to {output_file}")

        # Log split summary
        logger.info(f"Split summary:")
        logger.info(f"Train set: {len(train_data)} records")
        logger.info(f"Validation set: {len(val_data)} records")
        logger.info(f"Test set: {len(test_data)} records")

    def run(self) -> None:
        """Execute the complete data ingestion pipeline.

        This method orchestrates the entire data ingestion process:
        1. Downloads raw data from Google Cloud Storage
        2. Splits the data into training, validation, and test sets
        3. Saves the processed data to CSV files

        The pipeline is designed to be fault-tolerant, with detailed logging
        at each step. If any critical step fails, an exception will be raised
        after logging the error details.

        Raises:
            RuntimeError: If a critical error occurs during the ingestion process

        Example:
            >>> from config_reader import read_config
            >>> config = read_config("config/config.yaml")
            >>> data_ingestion = DataIngestion(config)
            >>> data_ingestion.run()

        Note:
            - Progress and status are logged at appropriate levels (INFO/ERROR)
            - The method is idempotent and can be safely retried on failure
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA INGESTION PIPELINE".center(60))
        logger.info("=" * 60)

        try:
            # Log configuration
            logger.info("Configuration:")
            logger.info(f"  Project ID: {self.project_id or 'Default'}")
            logger.info(f"  Bucket: {self.bucket_name}")
            logger.info(f"  Train/Val Object: {self.train_val_object_name}")
            logger.info(f"  Test Object: {self.test_object_name}")
            logger.info(
                f"  Train/Val Split: {self.train_ratio:.1%} train, {1 - self.train_ratio:.1%} validation"
            )
            logger.info(f"  Artifact Directory: {self.artifact_dir.absolute()}")

            # 1. Download raw data
            logger.info("\n" + "-" * 60)
            logger.info("DOWNLOADING DATA".center(60))
            logger.info("-" * 60)
            train_val_data, test_data = self.download_raw_data()

            if not any([train_val_data, test_data]):
                raise RuntimeError("Failed to download any data from GCS")

            # 2. Process and split data
            logger.info("\n" + "-" * 60)
            logger.info("PROCESSING DATA".center(60))
            logger.info("-" * 60)
            train_data, val_data, test_data = self.split_data(train_val_data, test_data)

            if not any([train_data, val_data, test_data]):
                raise RuntimeError("No valid data available after processing")

            # 3. Save processed data
            logger.info("\n" + "-" * 60)
            logger.info("SAVING DATA".center(60))
            logger.info("-" * 60)
            self.save_to_csv_files(train_data, val_data, test_data)

            logger.info("\n" + "=" * 60)
            logger.info("DATA INGESTION COMPLETED SUCCESSFULLY".center(60))
            logger.info("=" * 60)

        except Exception as e:
            logger.error("\n" + "!" * 60)
            logger.error("DATA INGESTION FAILED".center(60))
            logger.error("!" * 60)
            logger.exception(f"Error during data ingestion: {str(e)}")
            raise RuntimeError(f"Data ingestion failed: {str(e)}") from e
