"""
Unit Tests for EDAAnalyzer

Tests the exploratory data analysis functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.append(str(Path.cwd().parent))

from src.eda_analyzer import EDAAnalyzer


class TestEDAAnalyzer:
    """Test suite for EDAAnalyzer class."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing."""
        data = {
            "Product": [
                "Credit card",
                "Credit card",
                "Personal loan",
                "Savings account",
                "Money transfers",
            ],
            "Consumer complaint narrative": [
                "This is a short complaint",
                "This is a much longer complaint with many more words that should be counted properly",
                "Another complaint here",
                None,  # Missing narrative
                "A very long narrative " * 100,  # Very long narrative (>500 words)
            ],
            "Complaint ID": [1001, 1002, 1003, 1004, 1005],
            "Company": ["Bank A", "Bank B", "Bank C", "Bank D", "Bank E"],
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def empty_df(self):
        """Create an empty dataframe."""
        return pd.DataFrame()

    def test_initialization_success(self, sample_df):
        """Test successful initialization with valid dataframe."""
        analyzer = EDAAnalyzer(sample_df)
        assert analyzer.df is not None
        assert len(analyzer.df) == 5

    def test_initialization_empty_df(self, empty_df):
        """Test that initialization fails with empty dataframe."""
        with pytest.raises(ValueError, match="DataFrame cannot be None or empty"):
            EDAAnalyzer(empty_df)

    def test_initialization_none_df(self):
        """Test that initialization fails with None dataframe."""
        with pytest.raises(ValueError, match="DataFrame cannot be None or empty"):
            EDAAnalyzer(None)  # type: ignore

    def test_analyze_product_distribution_no_plot(self, sample_df):
        """Test product distribution analysis without plotting."""
        analyzer = EDAAnalyzer(sample_df)
        distribution = analyzer.analyze_product_distribution(plot=False)

        assert isinstance(distribution, pd.DataFrame)
        assert "Count" in distribution.columns
        assert "Percentage" in distribution.columns
        assert len(distribution) > 0
        assert distribution["Count"].sum() == 5
        assert abs(distribution["Percentage"].sum() - 100.0) < 0.01

    def test_analyze_product_distribution_missing_column(self):
        """Test product distribution when Product column is missing."""
        df = pd.DataFrame({"Some Column": [1, 2, 3]})
        analyzer = EDAAnalyzer(df)
        distribution = analyzer.analyze_product_distribution(plot=False)

        assert distribution.empty

    def test_analyze_narrative_length_no_plot(self, sample_df):
        """Test narrative length analysis without plotting."""
        analyzer = EDAAnalyzer(sample_df)
        stats = analyzer.analyze_narrative_length(plot=False)

        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "very_short_count" in stats
        assert "very_long_count" in stats

        # Check that we analyzed 4 narratives (1 is missing)
        assert stats["count"] == 4

    def test_analyze_narrative_length_very_short(self):
        """Test identification of very short narratives."""
        df = pd.DataFrame(
            {
                "Consumer complaint narrative": [
                    "Short",  # 1 word - very short
                    "This is longer complaint text here",  # 6 words - still very short
                    "Hi",  # 1 word - very short
                ]
            }
        )
        analyzer = EDAAnalyzer(df)
        stats = analyzer.analyze_narrative_length(plot=False)

        # All 3 are less than 10 words
        assert stats["very_short_count"] == 3

    def test_analyze_narrative_length_very_long(self):
        """Test identification of very long narratives."""
        df = pd.DataFrame(
            {
                "Consumer complaint narrative": [
                    "Normal complaint",
                    "word " * 600,  # 600 words - very long
                ]
            }
        )
        analyzer = EDAAnalyzer(df)
        stats = analyzer.analyze_narrative_length(plot=False)

        assert stats["very_long_count"] == 1

    def test_analyze_narrative_length_missing_column(self):
        """Test narrative length when column is missing."""
        df = pd.DataFrame({"Some Column": [1, 2, 3]})
        analyzer = EDAAnalyzer(df)
        stats = analyzer.analyze_narrative_length(plot=False)

        assert stats == {}

    def test_identify_missing_narratives(self, sample_df):
        """Test identification of missing narratives."""
        analyzer = EDAAnalyzer(sample_df)
        summary = analyzer.identify_missing_narratives()

        assert isinstance(summary, dict)
        assert summary["total_complaints"] == 5
        assert summary["with_narrative"] == 4
        assert summary["without_narrative"] == 1
        assert summary["without_narrative_pct"] == 20.0
        assert len(summary["narrative_indices"]) == 4

    def test_identify_missing_narratives_all_present(self):
        """Test when all narratives are present."""
        df = pd.DataFrame(
            {"Consumer complaint narrative": ["Text 1", "Text 2", "Text 3"]}
        )
        analyzer = EDAAnalyzer(df)
        summary = analyzer.identify_missing_narratives()

        assert summary["without_narrative"] == 0
        assert summary["with_narrative"] == 3

    def test_identify_missing_narratives_all_missing(self):
        """Test when all narratives are missing."""
        df = pd.DataFrame({"Consumer complaint narrative": [None, None, None]})
        analyzer = EDAAnalyzer(df)
        summary = analyzer.identify_missing_narratives()

        assert summary["without_narrative"] == 3
        assert summary["with_narrative"] == 0

    def test_identify_missing_narratives_missing_column(self):
        """Test missing narratives when column doesn't exist."""
        df = pd.DataFrame({"Some Column": [1, 2, 3]})
        analyzer = EDAAnalyzer(df)
        summary = analyzer.identify_missing_narratives()

        assert summary == {}

    def test_generate_eda_report(self, sample_df):
        """Test EDA report generation."""
        analyzer = EDAAnalyzer(sample_df)
        report = analyzer.generate_eda_report()

        assert isinstance(report, str)
        assert "EXPLORATORY DATA ANALYSIS REPORT" in report
        assert "DATASET OVERVIEW" in report
        assert "PRODUCT DISTRIBUTION" in report
        assert "NARRATIVE LENGTH" in report
        assert "MISSING NARRATIVE" in report

    def test_generate_eda_report_save_to_file(self, sample_df):
        """Test saving EDA report to file."""
        analyzer = EDAAnalyzer(sample_df)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            report = analyzer.generate_eda_report(output_path=temp_path)

            # Check file was created
            assert os.path.exists(temp_path)

            # Check file content matches report
            with open(temp_path, "r") as file:
                saved_content = file.read()
            assert saved_content == report
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_dataframe_immutability(self, sample_df):
        """Test that the analyzer doesn't modify the original dataframe."""
        original_df = sample_df.copy()
        analyzer = EDAAnalyzer(sample_df)

        # Run various analyses
        analyzer.analyze_product_distribution(plot=False)
        analyzer.analyze_narrative_length(plot=False)
        analyzer.identify_missing_narratives()

        # Original dataframe should be unchanged
        pd.testing.assert_frame_equal(sample_df, original_df)
