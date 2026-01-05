"""
Data Preprocessing Module for CFPB Complaints

This module provides functionality to filter, clean, and preprocess complaint
data for the RAG pipeline.
"""

import pandas as pd
import re
import logging
from typing import List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data Preprocessor for CFPB Complaints Dataset.

    This class provides methods to filter products, remove missing narratives,
    and clean text data for optimal embedding and retrieval.

    Attributes:
        df (pd.DataFrame): The complaints dataframe to preprocess.
        target_products (List[str]): List of product categories to filter.
    """

    # Target products for CrediTrust Financial
    DEFAULT_PRODUCTS = [
        "Credit card",
        "Credit card or prepaid card",
        "Prepaid card",
        "Personal loan",
        "Money transfer",
        "Money transfer, virtual currency, or money service",
        "Virtual currency",
        "Savings account",
        "Checking or savings account",
    ]

    def __init__(self, df: pd.DataFrame, target_products: Optional[List[str]] = None):
        """
        Initialize the Data Preprocessor.

        Args:
            df: Pandas DataFrame containing complaints data.
            target_products: Optional list of product categories to filter.
                           If None, uses DEFAULT_PRODUCTS.

        Raises:
            ValueError: If df is None or empty.
        """
        if df is None or len(df) == 0:
            raise ValueError("DataFrame cannot be None or empty")

        self.df = df.copy()
        self.target_products = (
            target_products if target_products else self.DEFAULT_PRODUCTS
        )
        logger.info(f"Initialized DataPreprocessor with {len(self.df)} records")
        logger.info(f"Target products: {self.target_products}")

    def filter_by_products(self, products: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter complaints to include only specified product categories.

        Args:
            products: Optional list of products to filter. If None, uses
                     target_products from initialization.

        Returns:
            pd.DataFrame: Filtered dataframe with only specified products.
        """
        if "Product" not in self.df.columns:
            logger.warning("'Product' column not found in dataframe")
            return self.df

        products_to_filter = products if products else self.target_products

        original_count = len(self.df)
        self.df = self.df[self.df["Product"].isin(products_to_filter)]
        filtered_count = len(self.df)

        logger.info(
            f"Filtered by products: {original_count} -> {filtered_count} records "
            f"({filtered_count/original_count*100:.1f}% retained)"
        )

        return self.df

    def remove_empty_narratives(self) -> pd.DataFrame:
        """
        Remove complaints with empty or missing consumer complaint narratives.

        Returns:
            pd.DataFrame: Dataframe with empty narratives removed.
        """
        if "Consumer complaint narrative" not in self.df.columns:
            logger.warning("'Consumer complaint narrative' column not found")
            return self.df

        original_count = len(self.df)

        # Remove null and empty string narratives
        self.df = self.df[self.df["Consumer complaint narrative"].notna()]
        self.df = self.df[self.df["Consumer complaint narrative"].str.strip() != ""]

        filtered_count = len(self.df)
        removed = original_count - filtered_count

        logger.info(
            f"Removed empty narratives: {original_count} -> {filtered_count} records "
            f"({removed} removed, {filtered_count/original_count*100:.1f}% retained)"
        )

        return self.df

    def clean_text(
        self,
        text: str,
        lowercase: bool = True,
        remove_special_chars: bool = True,
        remove_extra_spaces: bool = True,
        remove_boilerplate: bool = True,
    ) -> str:
        """
        Clean individual text string for improved embedding quality.

        Args:
            text: Text string to clean.
            lowercase: Convert text to lowercase.
            remove_special_chars: Remove special characters except basic punctuation.
            remove_extra_spaces: Remove extra whitespace.
            remove_boilerplate: Remove common boilerplate phrases.

        Returns:
            str: Cleaned text string.
        """
        if not isinstance(text, str):
            return ""

        cleaned = text

        # Remove boilerplate phrases
        if remove_boilerplate:
            boilerplate_patterns = [
                r"I am writing to file a complaint( about)?",
                r"This is a complaint regarding",
                r"Dear Sir or Madam,",
                r"To whom it may concern,",
                r"I wish to file a complaint",
                r"I would like to file a complaint",
            ]
            for pattern in boilerplate_patterns:
                cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Remove URLs
        cleaned = re.sub(r"http\S+|www\.\S+", "", cleaned)

        # Remove email addresses
        cleaned = re.sub(r"\S+@\S+", "", cleaned)

        # Remove phone numbers (various formats)
        cleaned = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "", cleaned)
        cleaned = re.sub(r"\(\d{3}\)\s*\d{3}[-.]?\d{4}", "", cleaned)

        # Remove special characters but keep basic punctuation
        if remove_special_chars:
            # Keep letters, numbers, spaces, and basic punctuation (. , ! ? ' -)
            cleaned = re.sub(r"[^a-zA-Z0-9\s.,!?\'-]", " ", cleaned)

        # Remove extra spaces
        if remove_extra_spaces:
            cleaned = re.sub(r"\s+", " ", cleaned)
            cleaned = cleaned.strip()

        # Convert to lowercase
        if lowercase:
            cleaned = cleaned.lower()

        return cleaned

    def clean_narratives(
        self, column: str = "Consumer complaint narrative", **clean_kwargs
    ) -> pd.DataFrame:
        """
        Clean all narratives in the dataframe.

        Args:
            column: Name of column containing narratives.
            **clean_kwargs: Additional arguments to pass to clean_text().

        Returns:
            pd.DataFrame: Dataframe with cleaned narratives.
        """
        if column not in self.df.columns:
            logger.warning(f"'{column}' column not found in dataframe")
            return self.df

        logger.info(f"Cleaning narratives in column: {column}")

        # Apply cleaning to each narrative
        self.df[column] = self.df[column].apply(
            lambda x: self.clean_text(x, **clean_kwargs) if pd.notna(x) else x
        )

        # Remove any that became empty after cleaning
        original_count = len(self.df)
        self.df = self.df[self.df[column].str.strip() != ""]
        filtered_count = len(self.df)

        if filtered_count < original_count:
            logger.info(
                f"Removed {original_count - filtered_count} records that became empty after cleaning"
            )

        logger.info("Text cleaning completed")

        return self.df

    def preprocess_pipeline(
        self,
        filter_products: bool = True,
        remove_empty: bool = True,
        clean: bool = True,
        **clean_kwargs,
    ) -> pd.DataFrame:
        """
        Run complete preprocessing pipeline.

        Args:
            filter_products: Whether to filter by target products.
            remove_empty: Whether to remove empty narratives.
            clean: Whether to clean text narratives.
            **clean_kwargs: Additional arguments for text cleaning.

        Returns:
            pd.DataFrame: Fully preprocessed dataframe.
        """
        logger.info("=" * 60)
        logger.info("Starting preprocessing pipeline")
        logger.info("=" * 60)
        logger.info(f"Initial records: {len(self.df)}")

        # Step 1: Filter products
        if filter_products:
            logger.info("\nStep 1: Filtering products...")
            self.filter_by_products()

        # Step 2: Remove empty narratives
        if remove_empty:
            logger.info("\nStep 2: Removing empty narratives...")
            self.remove_empty_narratives()

        # Step 3: Clean text
        if clean:
            logger.info("\nStep 3: Cleaning text...")
            self.clean_narratives(**clean_kwargs)

        logger.info("=" * 60)
        logger.info(f"Preprocessing complete! Final records: {len(self.df)}")
        logger.info("=" * 60)

        return self.df

    def get_data(self) -> pd.DataFrame:
        """
        Get the preprocessed dataframe.

        Returns:
            pd.DataFrame: The preprocessed complaints dataframe.
        """
        return self.df

    def save_data(self, output_path: str) -> None:
        """
        Save the preprocessed dataframe to CSV.

        Args:
            output_path: Path to save the CSV file.
        """
        self.df.to_csv(output_path, index=False)
        logger.info(f"Saved preprocessed data to {output_path}")
        logger.info(f"Saved {len(self.df)} records")
