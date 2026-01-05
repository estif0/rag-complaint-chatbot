"""
Unit Tests for DataPreprocessor

Tests the data preprocessing functionality.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.append(str(Path.cwd().parent))

from src.data_preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing."""
        data = {
            "Product": [
                "Credit card",
                "Mortgage",
                "Personal loan",
                "Savings account",
                "Credit card",
                "Debt collection",
            ],
            "Consumer complaint narrative": [
                "I am writing to file a complaint about my credit card billing.",
                "Issue with mortgage payment",
                None,  # Missing narrative
                "Problem with savings account",
                "   ",  # Empty string
                "Normal complaint text here",
            ],
            "Complaint ID": [1001, 1002, 1003, 1004, 1005, 1006],
            "Company": ["Bank A", "Bank B", "Bank C", "Bank D", "Bank E", "Bank F"],
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def empty_df(self):
        """Create an empty dataframe."""
        return pd.DataFrame()

    def test_initialization_success(self, sample_df):
        """Test successful initialization."""
        preprocessor = DataPreprocessor(sample_df)
        assert preprocessor.df is not None
        assert len(preprocessor.df) == 6
        assert preprocessor.target_products is not None

    def test_initialization_custom_products(self, sample_df):
        """Test initialization with custom target products."""
        custom_products = ["Credit card", "Personal loan"]
        preprocessor = DataPreprocessor(sample_df, target_products=custom_products)
        assert preprocessor.target_products == custom_products

    def test_initialization_empty_df(self, empty_df):
        """Test that initialization fails with empty dataframe."""
        with pytest.raises(ValueError, match="DataFrame cannot be None or empty"):
            DataPreprocessor(empty_df)

    def test_initialization_none_df(self):
        """Test that initialization fails with None dataframe."""
        with pytest.raises(ValueError, match="DataFrame cannot be None or empty"):
            DataPreprocessor(None)  # type: ignore

    def test_filter_by_products_default(self, sample_df):
        """Test filtering by default target products."""
        preprocessor = DataPreprocessor(
            sample_df,
            target_products=["Credit card", "Personal loan", "Savings account"],
        )
        filtered_df = preprocessor.filter_by_products()

        # Should keep Credit card (2), Personal loan (1), Savings account (1) = 4 total
        assert len(filtered_df) == 4
        assert set(filtered_df["Product"].unique()) <= {
            "Credit card",
            "Personal loan",
            "Savings account",
        }

    def test_filter_by_products_custom(self, sample_df):
        """Test filtering by custom products list."""
        preprocessor = DataPreprocessor(sample_df)
        filtered_df = preprocessor.filter_by_products(products=["Credit card"])

        # Should keep only Credit card (2)
        assert len(filtered_df) == 2
        assert filtered_df["Product"].unique()[0] == "Credit card"

    def test_filter_by_products_missing_column(self):
        """Test filtering when Product column is missing."""
        df = pd.DataFrame({"Some Column": [1, 2, 3]})
        preprocessor = DataPreprocessor(df)
        result = preprocessor.filter_by_products()

        # Should return dataframe unchanged
        assert len(result) == 3

    def test_remove_empty_narratives(self, sample_df):
        """Test removal of empty narratives."""
        preprocessor = DataPreprocessor(sample_df)
        cleaned_df = preprocessor.remove_empty_narratives()

        # Should remove None and empty string narratives (2 removed: indices 2 and 4)
        assert len(cleaned_df) == 4
        assert cleaned_df["Consumer complaint narrative"].notna().all()
        assert (cleaned_df["Consumer complaint narrative"].str.strip() != "").all()

    def test_remove_empty_narratives_missing_column(self):
        """Test removing empty narratives when column is missing."""
        df = pd.DataFrame({"Some Column": [1, 2, 3]})
        preprocessor = DataPreprocessor(df)
        result = preprocessor.remove_empty_narratives()

        # Should return dataframe unchanged
        assert len(result) == 3

    def test_clean_text_basic(self, sample_df):
        """Test basic text cleaning."""
        preprocessor = DataPreprocessor(sample_df)

        text = "I am writing to file a complaint about my CREDIT CARD!"
        cleaned = preprocessor.clean_text(text)

        assert cleaned.islower()
        assert (
            "writing to file a complaint" not in cleaned.lower()
        )  # Boilerplate removed
        assert "credit card" in cleaned

    def test_clean_text_urls(self, sample_df):
        """Test removal of URLs."""
        preprocessor = DataPreprocessor(sample_df)

        text = "Visit http://example.com or www.test.com for more info"
        cleaned = preprocessor.clean_text(text)

        assert "http://example.com" not in cleaned
        assert "www.test.com" not in cleaned
        assert "visit" in cleaned
        assert "for more info" in cleaned

    def test_clean_text_emails(self, sample_df):
        """Test removal of email addresses."""
        preprocessor = DataPreprocessor(sample_df)

        text = "Contact me at test@example.com for details"
        cleaned = preprocessor.clean_text(text)

        assert "test@example.com" not in cleaned
        assert "contact me at" in cleaned

    def test_clean_text_phone_numbers(self, sample_df):
        """Test removal of phone numbers."""
        preprocessor = DataPreprocessor(sample_df)

        text = "Call me at 555-123-4567 or (555) 987-6543"
        cleaned = preprocessor.clean_text(text)

        assert "555-123-4567" not in cleaned
        assert "(555) 987-6543" not in cleaned
        assert "call me at" in cleaned

    def test_clean_text_special_characters(self, sample_df):
        """Test removal of special characters."""
        preprocessor = DataPreprocessor(sample_df)

        text = "This has special chars: @#$%^&*() but keeps basic punctuation."
        cleaned = preprocessor.clean_text(text)

        assert "@#$%^&*()" not in cleaned
        assert "." in cleaned  # Basic punctuation kept

    def test_clean_text_extra_spaces(self, sample_df):
        """Test removal of extra spaces."""
        preprocessor = DataPreprocessor(sample_df)

        text = "Too    many      spaces    here"
        cleaned = preprocessor.clean_text(text)

        assert "  " not in cleaned  # No double spaces
        assert cleaned.count(" ") == 3  # Only single spaces between words

    def test_clean_text_no_lowercase(self, sample_df):
        """Test text cleaning without lowercasing."""
        preprocessor = DataPreprocessor(sample_df)

        text = "KEEP THIS UPPERCASE"
        cleaned = preprocessor.clean_text(text, lowercase=False)

        assert cleaned.isupper()

    def test_clean_text_empty_string(self, sample_df):
        """Test cleaning empty string."""
        preprocessor = DataPreprocessor(sample_df)

        cleaned = preprocessor.clean_text("")
        assert cleaned == ""

    def test_clean_text_non_string(self, sample_df):
        """Test cleaning non-string input."""
        preprocessor = DataPreprocessor(sample_df)

        cleaned = preprocessor.clean_text(None)  # type: ignore
        assert cleaned == ""

    def test_clean_narratives(self, sample_df):
        """Test cleaning all narratives in dataframe."""
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.remove_empty_narratives()  # Remove empty first
        cleaned_df = preprocessor.clean_narratives()

        # Check all narratives are lowercase
        assert all(cleaned_df["Consumer complaint narrative"].str.islower())

        # Check that at least one had boilerplate removed
        boilerplate_present = any(
            "writing to file a complaint" in str(text)
            for text in cleaned_df["Consumer complaint narrative"]
        )
        assert not boilerplate_present

    def test_clean_narratives_missing_column(self):
        """Test cleaning narratives when column is missing."""
        df = pd.DataFrame({"Some Column": [1, 2, 3]})
        preprocessor = DataPreprocessor(df)
        result = preprocessor.clean_narratives()

        # Should return dataframe unchanged
        assert len(result) == 3

    def test_preprocess_pipeline_full(self, sample_df):
        """Test complete preprocessing pipeline."""
        preprocessor = DataPreprocessor(
            sample_df,
            target_products=["Credit card", "Personal loan", "Savings account"],
        )

        result = preprocessor.preprocess_pipeline(
            filter_products=True, remove_empty=True, clean=True
        )

        # Should filter to target products, remove empty, and clean
        assert len(result) > 0
        assert len(result) < len(sample_df)  # Some should be filtered

        # All remaining should be target products
        assert set(result["Product"].unique()) <= {
            "Credit card",
            "Personal loan",
            "Savings account",
        }

        # All narratives should be cleaned
        assert all(result["Consumer complaint narrative"].str.islower())

    def test_preprocess_pipeline_selective(self, sample_df):
        """Test selective preprocessing."""
        preprocessor = DataPreprocessor(sample_df)

        result = preprocessor.preprocess_pipeline(
            filter_products=False, remove_empty=True, clean=False
        )

        # Should only remove empty narratives
        assert len(result) == 4  # 2 empty removed from 6

    def test_get_data(self, sample_df):
        """Test getting preprocessed data."""
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.remove_empty_narratives()

        data = preprocessor.get_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 4

    def test_save_data(self, sample_df):
        """Test saving preprocessed data to CSV."""
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.remove_empty_narratives()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            preprocessor.save_data(temp_path)

            # Check file exists
            assert os.path.exists(temp_path)

            # Check file content
            saved_df = pd.read_csv(temp_path)
            assert len(saved_df) == 4
            assert "Consumer complaint narrative" in saved_df.columns
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_dataframe_immutability_init(self, sample_df):
        """Test that original dataframe is not modified during initialization."""
        original_df = sample_df.copy()
        preprocessor = DataPreprocessor(sample_df)

        # Original should be unchanged
        pd.testing.assert_frame_equal(sample_df, original_df)
