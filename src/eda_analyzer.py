"""
Exploratory Data Analysis (EDA) Module for CFPB Complaints

This module provides functionality to analyze complaint data, including
product distribution, narrative length analysis, and missing data identification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Optional, Tuple, List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """
    Exploratory Data Analysis for CFPB Complaints Dataset.

    This class provides methods to analyze complaint data patterns,
    distributions, and quality issues.

    Attributes:
        df (pd.DataFrame): The complaints dataframe to analyze.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the EDA Analyzer.

        Args:
            df: Pandas DataFrame containing complaints data.

        Raises:
            ValueError: If df is None or empty.
        """
        if df is None or len(df) == 0:
            raise ValueError("DataFrame cannot be None or empty")

        self.df = df.copy()
        logger.info(f"Initialized EDA Analyzer with {len(self.df)} records")

    def analyze_product_distribution(
        self, plot: bool = True, save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyze the distribution of complaints across product categories.

        Args:
            plot: Whether to create visualization of distribution.
            save_path: Optional path to save the plot.

        Returns:
            pd.DataFrame: Distribution summary with counts and percentages.
        """
        if "Product" not in self.df.columns:
            logger.warning("'Product' column not found in dataframe")
            return pd.DataFrame()

        # Calculate distribution
        product_counts = self.df["Product"].value_counts()
        product_pct = self.df["Product"].value_counts(normalize=True) * 100

        distribution = pd.DataFrame(
            {"Count": product_counts, "Percentage": product_pct}
        )

        logger.info(
            f"Analyzed distribution across {len(distribution)} product categories"
        )

        # Create visualization if requested
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Bar plot
            product_counts.head(10).plot(kind="barh", ax=ax1, color="skyblue")
            ax1.set_xlabel("Number of Complaints")
            ax1.set_ylabel("Product Category")
            ax1.set_title("Top 10 Products by Complaint Count")
            ax1.grid(axis="x", alpha=0.3)

            # Pie chart for top products
            top_products = product_counts.head(5)
            others_count = product_counts[5:].sum()
            if others_count > 0:
                top_products = pd.concat(
                    [top_products, pd.Series({"Others": others_count})]
                )

            ax2.pie(
                top_products,
                labels=top_products.index,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax2.set_title("Product Distribution (Top 5 + Others)")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved product distribution plot to {save_path}")

            plt.show()

        return distribution

    def analyze_narrative_length(
        self, plot: bool = True, save_path: Optional[str] = None
    ) -> Dict:
        """
        Analyze the length (word count) of consumer complaint narratives.

        Args:
            plot: Whether to create visualization of length distribution.
            save_path: Optional path to save the plot.

        Returns:
            dict: Statistics about narrative lengths including:
                - mean, median, std
                - min, max
                - quartiles
                - very_short (< 10 words)
                - very_long (> 500 words)
        """
        if "Consumer complaint narrative" not in self.df.columns:
            logger.warning("'Consumer complaint narrative' column not found")
            return {}

        # Calculate word counts (only for non-null narratives)
        narratives = self.df["Consumer complaint narrative"].dropna()
        word_counts = narratives.str.split().str.len()

        stats = {
            "count": len(word_counts),
            "mean": word_counts.mean(),
            "median": word_counts.median(),
            "std": word_counts.std(),
            "min": word_counts.min(),
            "max": word_counts.max(),
            "q25": word_counts.quantile(0.25),
            "q75": word_counts.quantile(0.75),
            "very_short_count": (word_counts < 10).sum(),
            "very_short_pct": (word_counts < 10).sum() / len(word_counts) * 100,
            "very_long_count": (word_counts > 500).sum(),
            "very_long_pct": (word_counts > 500).sum() / len(word_counts) * 100,
        }

        logger.info(
            f"Narrative length analysis: Mean={stats['mean']:.1f}, Median={stats['median']:.1f}"
        )
        logger.info(
            f"Very short (<10 words): {stats['very_short_count']} ({stats['very_short_pct']:.2f}%)"
        )
        logger.info(
            f"Very long (>500 words): {stats['very_long_count']} ({stats['very_long_pct']:.2f}%)"
        )

        # Create visualization if requested
        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Histogram
            axes[0, 0].hist(
                word_counts, bins=50, color="steelblue", edgecolor="black", alpha=0.7
            )
            axes[0, 0].set_xlabel("Word Count")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].set_title("Distribution of Narrative Lengths")
            axes[0, 0].axvline(
                stats["mean"],
                color="red",
                linestyle="--",
                label=f"Mean: {stats['mean']:.1f}",
            )
            axes[0, 0].axvline(
                stats["median"],
                color="green",
                linestyle="--",
                label=f"Median: {stats['median']:.1f}",
            )
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)

            # Box plot
            axes[0, 1].boxplot(word_counts, vert=True)
            axes[0, 1].set_ylabel("Word Count")
            axes[0, 1].set_title("Narrative Length Box Plot")
            axes[0, 1].grid(alpha=0.3)

            # Histogram zoomed (removing outliers for better view)
            filtered_counts = word_counts[word_counts <= word_counts.quantile(0.95)]  # type: ignore
            axes[1, 0].hist(
                filtered_counts,
                bins=50,
                color="lightcoral",
                edgecolor="black",
                alpha=0.7,
            )
            axes[1, 0].set_xlabel("Word Count")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].set_title("Distribution (Zoomed: 0-95th Percentile)")
            axes[1, 0].grid(alpha=0.3)

            # Summary statistics text
            axes[1, 1].axis("off")
            stats_text = f"""
            Narrative Length Statistics
            ───────────────────────────
            Count:        {stats['count']:,}
            Mean:         {stats['mean']:.1f} words
            Median:       {stats['median']:.1f} words
            Std Dev:      {stats['std']:.1f}
            
            Min:          {stats['min']:.0f} words
            25th %:       {stats['q25']:.1f} words
            75th %:       {stats['q75']:.1f} words
            Max:          {stats['max']:.0f} words
            
            Very Short (<10):  {stats['very_short_count']:,} ({stats['very_short_pct']:.2f}%)
            Very Long (>500):  {stats['very_long_count']:,} ({stats['very_long_pct']:.2f}%)
            """
            axes[1, 1].text(
                0.1,
                0.5,
                stats_text,
                fontsize=11,
                family="monospace",
                verticalalignment="center",
            )

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved narrative length plot to {save_path}")

            plt.show()

        return stats

    def identify_missing_narratives(self) -> Dict:
        """
        Identify complaints with and without narratives.

        Returns:
            dict: Summary of missing narrative information including:
                - total_complaints
                - with_narrative (count and percentage)
                - without_narrative (count and percentage)
                - narrative_indices (list of indices with narratives)
        """
        if "Consumer complaint narrative" not in self.df.columns:
            logger.warning("'Consumer complaint narrative' column not found")
            return {}

        total = len(self.df)
        missing = self.df["Consumer complaint narrative"].isnull().sum()
        present = total - missing

        summary = {
            "total_complaints": total,
            "with_narrative": present,
            "with_narrative_pct": (present / total * 100) if total > 0 else 0,
            "without_narrative": missing,
            "without_narrative_pct": (missing / total * 100) if total > 0 else 0,
            "narrative_indices": self.df[
                self.df["Consumer complaint narrative"].notnull()
            ].index.tolist(),
        }

        logger.info(f"Missing narrative analysis:")
        logger.info(f"  Total complaints: {total:,}")
        logger.info(
            f"  With narrative: {present:,} ({summary['with_narrative_pct']:.2f}%)"
        )
        logger.info(
            f"  Without narrative: {missing:,} ({summary['without_narrative_pct']:.2f}%)"
        )

        return summary

    def generate_eda_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive EDA report.

        Args:
            output_path: Optional path to save the report as text file.

        Returns:
            str: Formatted EDA report as string.
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("CFPB COMPLAINTS - EXPLORATORY DATA ANALYSIS REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")

        # Basic info
        report_lines.append("1. DATASET OVERVIEW")
        report_lines.append("-" * 70)
        report_lines.append(f"Total Records: {len(self.df):,}")
        report_lines.append(f"Total Columns: {len(self.df.columns)}")
        report_lines.append(f"Columns: {', '.join(self.df.columns.tolist())}")
        report_lines.append("")

        # Product distribution
        report_lines.append("2. PRODUCT DISTRIBUTION")
        report_lines.append("-" * 70)
        product_dist = self.analyze_product_distribution(plot=False)
        if not product_dist.empty:
            for product, row in product_dist.head(10).iterrows():
                report_lines.append(
                    f"{product:50s}: {row['Count']:8,} ({row['Percentage']:5.2f}%)"
                )
        report_lines.append("")

        # Narrative analysis
        report_lines.append("3. NARRATIVE LENGTH ANALYSIS")
        report_lines.append("-" * 70)
        narrative_stats = self.analyze_narrative_length(plot=False)
        if narrative_stats:
            report_lines.append(f"Count: {narrative_stats['count']:,}")
            report_lines.append(f"Mean Length: {narrative_stats['mean']:.1f} words")
            report_lines.append(f"Median Length: {narrative_stats['median']:.1f} words")
            report_lines.append(f"Std Dev: {narrative_stats['std']:.1f}")
            report_lines.append(
                f"Range: {narrative_stats['min']:.0f} - {narrative_stats['max']:.0f} words"
            )
            report_lines.append(
                f"Very Short (<10 words): {narrative_stats['very_short_count']:,} ({narrative_stats['very_short_pct']:.2f}%)"
            )
            report_lines.append(
                f"Very Long (>500 words): {narrative_stats['very_long_count']:,} ({narrative_stats['very_long_pct']:.2f}%)"
            )
        report_lines.append("")

        # Missing data
        report_lines.append("4. MISSING NARRATIVE DATA")
        report_lines.append("-" * 70)
        missing_info = self.identify_missing_narratives()
        if missing_info:
            report_lines.append(
                f"Total Complaints: {missing_info['total_complaints']:,}"
            )
            report_lines.append(
                f"With Narrative: {missing_info['with_narrative']:,} ({missing_info['with_narrative_pct']:.2f}%)"
            )
            report_lines.append(
                f"Without Narrative: {missing_info['without_narrative']:,} ({missing_info['without_narrative_pct']:.2f}%)"
            )
        report_lines.append("")

        report_lines.append("=" * 70)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 70)

        report = "\n".join(report_lines)

        # Save to file if path provided
        if output_path:
            Path(output_path).write_text(report)
            logger.info(f"Saved EDA report to {output_path}")

        return report
