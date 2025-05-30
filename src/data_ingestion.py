# Import required libraries
import random
from io import BytesIO
from pathlib import Path

import numpy as np
from google.cloud import storage

# Import logger function
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class DataIngestion:
    """Handle data ingestion from Google Cloud Storage, splitting, and saving to CSV."""

    def __init__(self, config):
        """Initialize with config and setup directories.

        Args:
            config (dict): Data ingestion settings.
        """
        # Extract and set configuration
        self.data_ingestion_config = config["data_ingestion"]
        self.project_id = self.data_ingestion_config.get("project_id", None)
        self.bucket_name = self.data_ingestion_config["bucket_name"]
        self.train_val_object_name = self.data_ingestion_config["train_val_object_name"]
        self.test_object_name = self.data_ingestion_config["test_object_name"]
        self.train_ratio = self.data_ingestion_config["train_ratio"]
        self.artifact_dir = Path(self.data_ingestion_config["artifact_dir"])
        # Create raw data directory
        self.raw_dir = self.artifact_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_from_gcp(self, bucket_name, object_name):
        """Download data from GCS and load as NumPy or text.

        Args:
            bucket_name (str): GCS bucket name.
            object_name (str): Object name to download.

        Returns:
            list or str: Downloaded data as list or string.

        Raises:
            Exception: If download or loading fails.
        """
        try:
            # Initialize GCS client and download blob
            client = storage.Client(project=self.project_id)
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(object_name)
            raw_data = blob.download_as_bytes()
            logger.info(f"Downloaded {object_name} from GCS ({len(raw_data)} bytes)")

            try:
                # Try loading as NumPy array or NPZ file
                data = np.load(BytesIO(raw_data), allow_pickle=True)
                if isinstance(data, np.lib.npyio.NpzFile):
                    data = data["arr_0"]
                logger.info(
                    f"Loaded {object_name} as .npy/.npz with shape: {data.shape}"
                )
                return data.tolist()
            except Exception as np_err:
                # Fallback to text if NumPy fails
                logger.warning(
                    f"Failed to load {object_name} as .npy/.npz: {str(np_err)}. Trying as text..."
                )
                data = raw_data.decode("utf-8")
                logger.info(f"Loaded {object_name} as text: {data[:100]}...")
                return data
        except Exception as e:
            logger.error(f"Failed to download {object_name}: {str(e)}")
            raise

    def download_raw_data(self):
        """Download train/validation and test data from GCS.

        Returns:
            tuple: (train_val_data, test_data).

        Raises:
            Exception: If both downloads fail.
        """
        train_val_data = None
        test_data = None
        errors = []

        logger.info(
            f"Downloading {self.train_val_object_name} and {self.test_object_name} from GCS"
        )

        # Attempt to download train/validation and test data
        try:
            train_val_data = self.download_from_gcp(
                self.bucket_name, self.train_val_object_name
            )
        except Exception as e:
            errors.append(f"train_val_object: {str(e)}")

        try:
            test_data = self.download_from_gcp(self.bucket_name, self.test_object_name)
        except Exception as e:
            errors.append(f"test_object: {str(e)}")

        # Raise exception if both downloads fail
        if errors and train_val_data is None and test_data is None:
            raise Exception(f"Failed to download data: {'; '.join(errors)}")

        return train_val_data, test_data

    def split_data(self, train_val_data, test_data):
        """Split train/validation data and process test data.

        Args:
            train_val_data (list or str or None): Train/validation data.
            test_data (list or str or None): Test data.

        Returns:
            tuple: (train_data, val_data, test_data_list).
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
        """Save train, validation, and test data to CSV files.

        Args:
            train_data (list): Training data.
            val_data (list): Validation data.
            test_data (list): Test data.
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

    def run(self):
        """Execute data ingestion pipeline."""
        logger.info(
            f"Data Ingestion started for {self.train_val_object_name} and {self.test_object_name}"
        )
        # Download, split, and save data
        train_val_data, test_data = self.download_raw_data()
        train_data, val_data, test_data = self.split_data(train_val_data, test_data)
        self.save_to_csv_files(train_data, val_data, test_data)
        logger.info(f"Data Ingestion completed successfully")
