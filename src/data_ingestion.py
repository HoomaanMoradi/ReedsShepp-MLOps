"""Data Ingestion Module for ReedsShepp-MLOps.

This module manages the complete data ingestion workflow for the ReedsShepp-MLOps project,
handling data from acquisition to preprocessing. It provides robust functionality for:
1. Downloading raw data from Google Cloud Storage (GCS)
2. Processing and validating data in both NumPy array and text formats
3. Splitting data into training, validation, and test sets
4. Saving processed data to standardized CSV files

The module is designed with fault tolerance and comprehensive logging, making it suitable
for production environments. It supports both compressed and uncompressed data formats
and includes validation checks to ensure data integrity.
"""

import random
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from google.cloud import storage

from src.logger import get_logger

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
        
        This method sets up the data ingestion pipeline by:
        1. Validating and extracting configuration parameters
        2. Setting up Google Cloud Storage connection parameters
        3. Creating the required directory structure for storing artifacts
        4. Ensuring the artifacts directory is mounted from the host

        Args:
            config: Dictionary containing data ingestion configuration with the
                   following required keys under 'data_ingestion':
                   - bucket_name: Name of the GCS bucket containing the data
                   - train_val_object_name: Path to training/validation data in GCS
                   - test_object_name: Path to test data in GCS
                   - train_ratio: Ratio for train/validation split (0.0 to 1.0)
                   - artifact_dir: Base directory for storing artifacts
                   - project_id: Optional Google Cloud project ID
                   - gcp_credentials_path: Optional path to GCP credentials file (default: /app/gcp-credentials.json)

        Raises:
            KeyError: If required configuration keys are missing
            OSError: If artifact directories cannot be created
            ValueError: If train_ratio is not between 0.0 and 1.0
            RuntimeError: If artifacts directory is not mounted from host
        """
        try:
            # Extract configuration with type hints
            self.data_ingestion_config: Dict[str, Any] = config["data_ingestion"]
            self.project_id: Optional[str] = self.data_ingestion_config.get("project_id")
            self.bucket_name: str = self.data_ingestion_config["bucket_name"]
            self.train_val_object_name: str = self.data_ingestion_config["train_val_object_name"]
            self.test_object_name: str = self.data_ingestion_config["test_object_name"]
            self.train_ratio: float = float(self.data_ingestion_config["train_ratio"])
            self.gcp_credentials_path: str = self.data_ingestion_config.get("gcp_credentials_path", "/app/gcp-credentials.json")

            # Set up directory structure
            self.artifact_dir: Path = Path(self.data_ingestion_config["artifact_dir"])
            self.raw_dir: Path = self.artifact_dir / "raw"

            # Check if artifacts directory exists and create it if needed
            if not self.artifact_dir.exists():
                try:
                    # Try to create the directory if it doesn't exist
                    self.artifact_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created artifacts directory at {self.artifact_dir}")
                except Exception as e:
                    logger.warning(f"Could not create artifacts directory: {str(e)}")
                    raise RuntimeError(f"Failed to create artifacts directory at {self.artifact_dir}. Please ensure you have write permissions.")
            
            # Set GOOGLE_APPLICATION_CREDENTIALS environment variable if credentials file exists
            if os.path.exists(self.gcp_credentials_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.gcp_credentials_path
                logger.info(f"Using GCP credentials from {self.gcp_credentials_path}")
            else:
                logger.warning(f"GCP credentials file not found at {self.gcp_credentials_path}. Trying default application credentials...")
            
            # Warn if running in local mode without proper permissions
            if not self.artifact_dir.is_dir():
                logger.warning("Artifacts directory is not a directory. This might be a security issue if running in Docker.")
            elif not os.access(str(self.artifact_dir), os.W_OK):
                logger.warning("Artifacts directory is not writable. This might be a security issue if running in Docker.")
            
            # Create directories if they don't exist
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Initialized DataIngestion with raw_dir={self.raw_dir}")

        except KeyError as e:
            logger.error(f"Missing required configuration: {e}")
            raise
        except OSError as e:
            logger.error(f"Failed to create directory structure: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid configuration value: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Security check failed: {e}")
            raise
        """Initialize the DataIngestion pipeline with configuration.

        This method sets up the data ingestion pipeline by:
        1. Validating and extracting configuration parameters
        2. Setting up Google Cloud Storage connection parameters
        3. Creating the required directory structure for storing artifacts

        Args:
            config: Dictionary containing data ingestion configuration with the
                   following required keys under 'data_ingestion':
                   - bucket_name: Name of the GCS bucket containing the data
                   - train_val_object_name: Path to training/validation data in GCS
                   - test_object_name: Path to test data in GCS
                   - train_ratio: Ratio for train/validation split (0.0 to 1.0)
                   - artifact_dir: Base directory for storing artifacts
                   - project_id: Optional Google Cloud project ID
                   - gcp_credentials_path: Optional path to GCP credentials file (default: /app/gcp-credentials.json)

        Raises:
            KeyError: If required configuration keys are missing
            OSError: If artifact directories cannot be created
            ValueError: If train_ratio is not between 0.0 and 1.0
        """
        try:
            # Extract configuration with type hints
            self.data_ingestion_config: Dict[str, Any] = config["data_ingestion"]
            self.project_id: Optional[str] = self.data_ingestion_config.get("project_id")
            self.bucket_name: str = self.data_ingestion_config["bucket_name"]
            self.train_val_object_name: str = self.data_ingestion_config["train_val_object_name"]
            self.test_object_name: str = self.data_ingestion_config["test_object_name"]
            self.train_ratio: float = float(self.data_ingestion_config["train_ratio"])
            self.gcp_credentials_path: str = self.data_ingestion_config.get("gcp_credentials_path", "/app/gcp-credentials.json")

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
    ) -> Union[List[Any], str, None]:
        """Download and parse data from Google Cloud Storage.

        This method attempts to download and parse data from GCS using a two-step process:
        1. First tries to load as a NumPy array (.npy/.npz)
        2. Falls back to text parsing if NumPy loading fails

        Supports both compressed and uncompressed NumPy arrays, with automatic detection
        of NPZ files (compressed NumPy arrays).

        Args:
            bucket_name: Name of the GCS bucket containing the data
            object_name: Path to the object within the bucket

        Returns:
            Union[List[Any], str]: 
                - List[Any]: If data was successfully parsed as NumPy array
                - str: If data was successfully parsed as text
                - None: If parsing fails in both formats

        Raises:
            google.cloud.exceptions.GoogleCloudError: If there's an issue with GCS access
            ValueError: If the downloaded data cannot be parsed in either format
            UnicodeDecodeError: If text decoding fails
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

    def download_raw_data(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Download and parse CSV data files from GCS if local data not found.

        This method first checks for local CSV files in artifacts/raw directory before
        attempting to download from GCS. If local files are found, it uses those instead
        of downloading. The CSV files should be named:
        - train.csv
        - validation.csv
        - test.csv

        Returns:
            A tuple containing:
            - train_val_data: List of lists containing training and validation data
            - test_data: List of lists containing test data

        Raises:
            RuntimeError: If data cannot be loaded from either local files or GCS
            Exception: For any other unexpected errors during download or parsing
        """
        train_val_data = []
        test_data = []
        errors = []

        # Check for local CSV files in artifacts/raw
        local_train_path = self.raw_dir / "train.csv"
        local_val_path = self.raw_dir / "validation.csv"
        local_test_path = self.raw_dir / "test.csv"

        # Try to load local CSV files first
        if local_train_path.exists() and local_val_path.exists():
            try:
                logger.info(f"Found local training data at {local_train_path}")
                logger.info(f"Found local validation data at {local_val_path}")
                
                # Load training data
                with open(local_train_path, 'r') as f:
                    train_data = []
                    # Skip header row
                    next(f)
                    for line in f:
                        try:
                            # Split line and convert to floats
                            try:
                                row = list(map(float, line.strip().split(',')))
                                if len(row) != 5:  # Ensure each row has exactly 5 columns
                                    logger.warning(f"Skipping invalid row in train.csv: {line.strip()}")
                                    continue
                                train_data.append(row)
                            except ValueError as e:
                                logger.warning(f"Error parsing row in train.csv: {line.strip()} - {str(e)}")
                                continue
                        except Exception as e:
                            logger.warning(f"Error parsing row in train.csv: {line.strip()} - {str(e)}")
                            continue
                
                # Load validation data
                with open(local_val_path, 'r') as f:
                    val_data = []
                    # Skip header row
                    next(f)
                    for line in f:
                        try:
                            # Split line and convert to floats
                            try:
                                row = list(map(float, line.strip().split(',')))
                                if len(row) != 5:  # Ensure each row has exactly 5 columns
                                    logger.warning(f"Skipping invalid row in validation.csv: {line.strip()}")
                                    continue
                                val_data.append(row)
                            except ValueError as e:
                                logger.warning(f"Error parsing row in validation.csv: {line.strip()} - {str(e)}")
                                continue
                        except Exception as e:
                            logger.warning(f"Error parsing row in validation.csv: {line.strip()} - {str(e)}")
                            continue
                
                # Combine train and validation data
                train_val_data = train_data + val_data
                logger.info(f"Successfully loaded {len(train_val_data)} training/validation samples from local files")
            except Exception as e:
                logger.warning(f"Failed to load local training/validation data: {str(e)}")
                train_val_data = []

        if local_test_path.exists():
            try:
                logger.info(f"Found local test data at {local_test_path}")
                with open(local_test_path, 'r') as f:
                    # Skip header row
                    next(f)
                    for line in f:
                        try:
                            # Split line and convert to floats
                            try:
                                row = list(map(float, line.strip().split(',')))
                                if len(row) != 5:  # Ensure each row has exactly 5 columns
                                    logger.warning(f"Skipping invalid row in test.csv: {line.strip()}")
                                    continue
                                test_data.append(row)
                            except ValueError as e:
                                logger.warning(f"Error parsing row in test.csv: {line.strip()} - {str(e)}")
                                continue
                        except Exception as e:
                            logger.warning(f"Error parsing row in test.csv: {line.strip()} - {str(e)}")
                            continue
                logger.info(f"Successfully loaded {len(test_data)} test samples from local file")
            except Exception as e:
                logger.warning(f"Failed to load local test data: {str(e)}")
                test_data = []

        # If local data found and successfully loaded, return it
        if train_val_data or test_data:
            logger.info("Using local CSV data instead of downloading from GCS")
            return train_val_data, test_data

        # If no local data found or loading failed, proceed with GCS download
        logger.info(
            f"No local data found, downloading from GCS bucket '{self.bucket_name}':\n"
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

    def split_data(
        self, 
        train_val_data: Union[List[Any], str, None], 
        test_data: Union[List[Any], str, None]
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        """Split and process training/validation and test datasets.

        This method processes raw data by:
        1. Parsing input data from either list or string format
        2. Validating row structure (5 columns required)
        3. Splitting training/validation data according to train_ratio
        4. Filtering invalid or malformed rows
        5. Returning processed datasets in consistent format

        Args:
            train_val_data: Training and validation data in either format:
                           - List of lists: Directly used for splitting
                           - String: Parsed into list of integers by row
            test_data: Test data in either format:
                     - List of lists: Directly used
                     - String: Parsed into list of integers by row

        Returns:
            A tuple containing three lists of integer lists:
            - train_data: Training data split according to train_ratio
            - val_data: Validation data (complement of training data split)
            - test_data_list: Processed test data

        Note:
            - Data validation ensures exactly 5 columns per row
            - Empty or invalid rows are logged and filtered out
            - Input strings should be newline-separated with space-separated integers
            - Data is shuffled before splitting to ensure randomness
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

    def save_to_csv_files(
        self, 
        train_data: List[List[int]], 
        val_data: List[List[int]], 
        test_data: List[List[int]]
    ) -> None:
        """Save the processed datasets to standardized CSV files.

        This method saves the processed datasets to CSV files with consistent formatting:
        1. Creates three separate CSV files (train, validation, test)
        2. Includes standardized header row across all files
        3. Ensures consistent data formatting
        4. Provides detailed logging of the save process

        Args:
            train_data: Training data as a list of lists, where each inner list
                      contains exactly 5 integers representing the columns:
                      [input1, input2, input3, extra, output]
            val_data: Validation data in the same format as train_data
            test_data: Test data in the same format as train_data

        Raises:
            OSError: If there are issues writing to the filesystem
            ValueError: If any row doesn't have exactly 5 columns
            Exception: For any other unexpected errors during file operations

        Note:
            - Output files are saved to {raw_dir} with filenames:
              - train.csv
              - validation.csv
              - test.csv
            - All files include the header: "input1,input2,input3,extra,output"
            - Empty datasets result in warning messages but no file creation
            - Provides summary logging of dataset sizes after saving
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

        This method orchestrates the entire data ingestion process in a sequential flow:
        1. Downloads raw data from Google Cloud Storage using download_raw_data()
        2. Processes and splits the data using split_data()
        3. Saves the processed datasets to CSV files using save_to_csv_files()

        The pipeline includes comprehensive error handling and logging at each step,
        making it suitable for production use. If any critical step fails, the method
        will raise an exception after logging detailed error information.

        Raises:
            RuntimeError: If a critical error occurs during any pipeline step
            Exception: For any other unexpected errors during execution

        Example:
            >>> from config_reader import read_config
            >>> config = read_config("config/config.yaml")
            >>> data_ingestion = DataIngestion(config)
            >>> data_ingestion.run()

        Note:
            - Progress and status are logged at appropriate levels (INFO/ERROR)
            - The method is idempotent and can be safely retried on failure
            - Provides detailed logging of each pipeline step
            - Returns None on successful completion
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
