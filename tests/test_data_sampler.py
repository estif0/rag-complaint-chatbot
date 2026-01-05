"""
Unit Tests for StratifiedSampler

Tests the stratified sampling functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_sampler import StratifiedSampler


class TestStratifiedSampler:
    """Test suite for StratifiedSampler class."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe with known distribution."""
        data = {
            "Product": ["Credit card"] * 50
            + ["Personal loan"] * 30
            + ["Savings account"] * 15
            + ["Money transfer"] * 5,
            "Consumer complaint narrative": [f"Complaint {i}" for i in range(100)],
            "Complaint ID": range(1000, 1100),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def empty_df(self):
        """Create an empty dataframe."""
        return pd.DataFrame()

    def test_initialization_success(self, sample_df):
        """Test successful initialization."""
        sampler = StratifiedSampler(sample_df)
        assert sampler.df is not None
        assert len(sampler.df) == 100
        assert sampler.product_column == "Product"

    def test_initialization_custom_column(self, sample_df):
        """Test initialization with custom product column."""
        df = sample_df.rename(columns={"Product": "ProductType"})
        sampler = StratifiedSampler(df, product_column="ProductType")
        assert sampler.product_column == "ProductType"

    def test_initialization_empty_df(self, empty_df):
        """Test that initialization fails with empty dataframe."""
        with pytest.raises(ValueError, match="DataFrame cannot be None or empty"):
            StratifiedSampler(empty_df)

    def test_initialization_none_df(self):
        """Test that initialization fails with None dataframe."""
        with pytest.raises(ValueError, match="DataFrame cannot be None or empty"):
            StratifiedSampler(None)

    def test_initialization_missing_column(self, sample_df):
        """Test that initialization fails when column doesn't exist."""
        with pytest.raises(ValueError, match="Column 'NonExistent' not found"):
            StratifiedSampler(sample_df, product_column="NonExistent")

    def test_create_stratified_sample_basic(self, sample_df):
        """Test basic stratified sampling."""
        sampler = StratifiedSampler(sample_df)
        sample = sampler.create_stratified_sample(sample_size=50, random_state=42)

        assert len(sample) == 50
        assert isinstance(sample, pd.DataFrame)
        assert "Product" in sample.columns

    def test_create_stratified_sample_maintains_distribution(self, sample_df):
        """Test that sampling maintains approximate distribution."""
        sampler = StratifiedSampler(sample_df)
        sample = sampler.create_stratified_sample(sample_size=50, random_state=42)

        # Original: 50% Credit card, 30% Personal loan, 15% Savings, 5% Money transfer
        sample_dist = sample["Product"].value_counts(normalize=True)

        # Check proportions are roughly maintained (within 10% tolerance)
        assert sample_dist["Credit card"] > 0.4  # Should be around 50%
        assert sample_dist["Personal loan"] > 0.2  # Should be around 30%

    def test_create_stratified_sample_reproducibility(self, sample_df):
        """Test that same random_state produces same sample."""
        sampler = StratifiedSampler(sample_df)
        sample1 = sampler.create_stratified_sample(sample_size=50, random_state=42)
        sample2 = sampler.create_stratified_sample(sample_size=50, random_state=42)

        pd.testing.assert_frame_equal(sample1.sort_index(), sample2.sort_index())

    def test_create_stratified_sample_invalid_size(self, sample_df):
        """Test that invalid sample sizes raise errors."""
        sampler = StratifiedSampler(sample_df)

        with pytest.raises(ValueError, match="Sample size must be positive"):
            sampler.create_stratified_sample(sample_size=0)

        with pytest.raises(ValueError, match="Sample size must be positive"):
            sampler.create_stratified_sample(sample_size=-10)

    def test_create_stratified_sample_exceeds_data(self, sample_df):
        """Test that sample size exceeding data size raises error."""
        sampler = StratifiedSampler(sample_df)

        with pytest.raises(ValueError, match="exceeds dataset size"):
            sampler.create_stratified_sample(sample_size=200)

    def test_create_stratified_sample_exact_size(self, sample_df):
        """Test that exact requested size is returned."""
        sampler = StratifiedSampler(sample_df)

        for size in [10, 25, 50, 75]:
            sample = sampler.create_stratified_sample(sample_size=size, random_state=42)
            assert len(sample) == size

    def test_validate_sample_distribution_success(self, sample_df):
        """Test validation of a good sample."""
        sampler = StratifiedSampler(sample_df)
        sample = sampler.create_stratified_sample(sample_size=50, random_state=42)

        report = sampler.validate_sample_distribution(sample, tolerance=0.1)

        assert "original_distribution" in report
        assert "sample_distribution" in report
        assert "deviations" in report
        assert "within_tolerance" in report
        assert "max_deviation" in report
        assert isinstance(report["within_tolerance"], (bool, np.bool_))

    def test_validate_sample_distribution_empty_sample(self, sample_df):
        """Test that validation fails with empty sample."""
        sampler = StratifiedSampler(sample_df)

        with pytest.raises(
            ValueError, match="Sample DataFrame cannot be None or empty"
        ):
            sampler.validate_sample_distribution(pd.DataFrame())

    def test_validate_sample_distribution_strict_tolerance(self, sample_df):
        """Test validation with very strict tolerance."""
        sampler = StratifiedSampler(sample_df)
        sample = sampler.create_stratified_sample(sample_size=10, random_state=42)

        # With very strict tolerance, might not pass
        report = sampler.validate_sample_distribution(sample, tolerance=0.001)

        assert "within_tolerance" in report
        # Small samples might not meet strict tolerance

    def test_validate_sample_distribution_deviations(self, sample_df):
        """Test that deviations are calculated correctly."""
        sampler = StratifiedSampler(sample_df)
        sample = sampler.create_stratified_sample(sample_size=50, random_state=42)

        report = sampler.validate_sample_distribution(sample, tolerance=0.1)

        # Check that all products have deviation values
        for product in sample_df["Product"].unique():
            assert product in report["deviations"]
            assert isinstance(report["deviations"][product], (int, float))

    def test_get_sample_summary(self, sample_df):
        """Test sample summary generation."""
        sampler = StratifiedSampler(sample_df)
        sample = sampler.create_stratified_sample(sample_size=50, random_state=42)

        summary = sampler.get_sample_summary(sample)

        assert isinstance(summary, pd.DataFrame)
        assert "Original_Count" in summary.columns
        assert "Original_Pct" in summary.columns
        assert "Sample_Count" in summary.columns
        assert "Sample_Pct" in summary.columns
        assert "Difference_Pct" in summary.columns

        # Check that percentages sum to ~100
        assert abs(summary["Original_Pct"].sum() - 100.0) < 0.01
        assert abs(summary["Sample_Pct"].sum() - 100.0) < 0.01

    def test_get_sample_summary_counts(self, sample_df):
        """Test that sample summary counts are correct."""
        sampler = StratifiedSampler(sample_df)
        sample = sampler.create_stratified_sample(sample_size=50, random_state=42)

        summary = sampler.get_sample_summary(sample)

        # Original counts should match
        assert summary["Original_Count"].sum() == 100
        assert summary["Sample_Count"].sum() == 50

    def test_dataframe_immutability(self, sample_df):
        """Test that original dataframe is not modified."""
        original_df = sample_df.copy()
        sampler = StratifiedSampler(sample_df)

        _ = sampler.create_stratified_sample(sample_size=50, random_state=42)

        # Original should be unchanged
        pd.testing.assert_frame_equal(sample_df, original_df)

    def test_large_sample(self, sample_df):
        """Test sampling close to original size."""
        sampler = StratifiedSampler(sample_df)
        sample = sampler.create_stratified_sample(sample_size=95, random_state=42)

        assert len(sample) == 95

        # Should still maintain reasonable distribution
        report = sampler.validate_sample_distribution(sample, tolerance=0.05)
        assert report["within_tolerance"]
