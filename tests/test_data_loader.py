"""
Unit Tests for CFPBDataLoader

Tests the data loading and validation functionality.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.append(str(Path.cwd().parent))

from src.data_loader import CFPBDataLoader


class TestCFPBDataLoader:
    """Test suite for CFPBDataLoader class."""

    @pytest.fixture
    def sample_csv_file(self):
        """Create a temporary CSV file with sample complaint data."""
        data = {
            "Date received": ["2025-01-01", "2025-01-02"],
            "Product": ["Credit card", "Personal loan"],
            "Consumer complaint narrative": ["Issue with billing", "Late payment fee"],
            "Complaint ID": [1001, 1002],
            "Company": ["Bank A", "Bank B"],
            "State": ["CA", "NY"],
        }
        df = pd.DataFrame(data)

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    @pytest.fixture
    def empty_csv_file(self):
        """Create an empty CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_initialization(self, sample_csv_file):
        """Test that loader initializes correctly."""
        loader = CFPBDataLoader(sample_csv_file)
        assert loader.data_path == Path(sample_csv_file)
        assert loader.df is None

    def test_load_raw_data_success(self, sample_csv_file):
        """Test successful data loading."""
        loader = CFPBDataLoader(sample_csv_file)
        df = loader.load_raw_data()

        assert df is not None
        assert len(df) == 2
        assert "Product" in df.columns
        assert "Consumer complaint narrative" in df.columns

    def test_load_raw_data_with_nrows(self, sample_csv_file):
        """Test loading limited number of rows."""
        loader = CFPBDataLoader(sample_csv_file)
        df = loader.load_raw_data(nrows=1)

        assert len(df) == 1

    def test_load_raw_data_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        loader = CFPBDataLoader("nonexistent_file.csv")

        with pytest.raises(FileNotFoundError):
            loader.load_raw_data()

    def test_validate_data_before_loading(self, sample_csv_file):
        """Test that validation fails if data not loaded."""
        loader = CFPBDataLoader(sample_csv_file)

        with pytest.raises(ValueError, match="No data loaded"):
            loader.validate_data()

    def test_validate_data_success(self, sample_csv_file):
        """Test successful data validation."""
        loader = CFPBDataLoader(sample_csv_file)
        loader.load_raw_data()

        report = loader.validate_data()

        assert report["has_data"] is True
        assert report["shape"] == (2, 6)
        assert report["required_columns_present"] is True
        assert len(report["missing_required_columns"]) == 0

    def test_validate_data_missing_columns(self):
        """Test validation with missing required columns."""
        data = {"Date received": ["2025-01-01"], "Some Column": ["Some Value"]}
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            loader = CFPBDataLoader(temp_path)
            loader.load_raw_data()
            report = loader.validate_data()

            assert report["required_columns_present"] is False
            assert "Product" in report["missing_required_columns"]
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_get_data_success(self, sample_csv_file):
        """Test getting loaded data."""
        loader = CFPBDataLoader(sample_csv_file)
        loader.load_raw_data()
        df = loader.get_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_get_data_before_loading(self, sample_csv_file):
        """Test that get_data raises error if data not loaded."""
        loader = CFPBDataLoader(sample_csv_file)

        with pytest.raises(ValueError, match="No data loaded"):
            loader.get_data()

    def test_get_info_success(self, sample_csv_file, capsys):
        """Test get_info prints dataset information."""
        loader = CFPBDataLoader(sample_csv_file)
        loader.load_raw_data()
        loader.get_info()

        captured = capsys.readouterr()
        assert "CFPB Complaints Dataset Information" in captured.out

    def test_get_info_before_loading(self, sample_csv_file):
        """Test that get_info raises error if data not loaded."""
        loader = CFPBDataLoader(sample_csv_file)

        with pytest.raises(ValueError, match="No data loaded"):
            loader.get_info()
