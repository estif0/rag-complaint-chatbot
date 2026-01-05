"""
Data Loader Module for CFPB Complaints Dataset

This module provides functionality to load and validate the Consumer Financial
Protection Bureau (CFPB) complaints dataset.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CFPBDataLoader:
    """
    Loader for CFPB Complaints Dataset.

    This class handles loading and basic validation of the CFPB complaints
    dataset, ensuring data integrity and proper structure.

    Attributes:
        data_path (Path): Path to the CSV file containing complaints data.
        df (pd.DataFrame): Loaded dataframe (None until load_raw_data is called).
    """

    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize the CFPBDataLoader.

        Args:
            data_path: Path to the complaints CSV file.
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        logger.info(f"Initialized CFPBDataLoader with data path: {self.data_path}")

    def load_raw_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load the raw CFPB complaints dataset from CSV.

        Args:
            nrows: Optional number of rows to read. If None, reads entire file.

        Returns:
            pd.DataFrame: Loaded complaints dataframe.

        Raises:
            FileNotFoundError: If the data file doesn't exist.
            pd.errors.EmptyDataError: If the file is empty.
        """
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found at: {self.data_path}")

            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path, nrows=nrows)
            logger.info(f"Successfully loaded {len(self.df)} records")

            return self.df

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty data file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def validate_data(self) -> dict:
        """
        Validate the loaded data for completeness and structure.

        Checks for required columns, data shape, and missing values.

        Returns:
            dict: Validation report containing:
                - has_data: Whether data is loaded
                - shape: Tuple of (rows, columns)
                - columns: List of column names
                - required_columns_present: Whether key columns exist
                - missing_values: Dict of columns with missing value counts

        Raises:
            ValueError: If no data has been loaded yet.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")

        # Required columns for the RAG complaint chatbot
        required_columns = ["Product", "Consumer complaint narrative", "Complaint ID"]

        validation_report = {
            "has_data": True,
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "required_columns_present": all(
                col in self.df.columns for col in required_columns
            ),
            "missing_required_columns": [
                col for col in required_columns if col not in self.df.columns
            ],
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_complaint_ids": (
                self.df["Complaint ID"].duplicated().sum()
                if "Complaint ID" in self.df.columns
                else 0
            ),
        }

        # Log validation results
        logger.info("Data Validation Report:")
        logger.info(f"  - Shape: {validation_report['shape']}")
        logger.info(
            f"  - Required columns present: {validation_report['required_columns_present']}"
        )

        if not validation_report["required_columns_present"]:
            logger.warning(
                f"  - Missing required columns: {validation_report['missing_required_columns']}"
            )

        if validation_report["duplicate_complaint_ids"] > 0:
            logger.warning(
                f"  - Found {validation_report['duplicate_complaint_ids']} duplicate complaint IDs"
            )

        return validation_report

    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded dataframe.

        Returns:
            pd.DataFrame: The loaded complaints dataframe.

        Raises:
            ValueError: If no data has been loaded yet.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")
        return self.df

    def get_info(self) -> None:
        """
        Print detailed information about the loaded dataset.

        Raises:
            ValueError: If no data has been loaded yet.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")

        print("=" * 60)
        print("CFPB Complaints Dataset Information")
        print("=" * 60)
        self.df.info()
        print("\n" + "=" * 60)
        print("First few rows:")
        print("=" * 60)
        print(self.df.head())
