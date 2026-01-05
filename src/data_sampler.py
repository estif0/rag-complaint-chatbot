"""
Data Sampling Module for CFPB Complaints

This module provides functionality to create stratified samples from the
complaint dataset, ensuring proportional representation across product categories.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StratifiedSampler:
    """
    Stratified Sampler for CFPB Complaints Dataset.

    This class creates stratified samples that maintain proportional
    representation across product categories, essential for balanced
    training data in RAG systems.

    Attributes:
        df (pd.DataFrame): The complaints dataframe to sample from.
        product_column (str): Name of the column containing product categories.
    """

    def __init__(self, df: pd.DataFrame, product_column: str = "Product"):
        """
        Initialize the Stratified Sampler.

        Args:
            df: Pandas DataFrame containing complaints data.
            product_column: Name of the column to stratify by.

        Raises:
            ValueError: If df is None, empty, or product_column doesn't exist.
        """
        if df is None or len(df) == 0:
            raise ValueError("DataFrame cannot be None or empty")

        if product_column not in df.columns:
            raise ValueError(f"Column '{product_column}' not found in DataFrame")

        self.df = df.copy()
        self.product_column = product_column
        logger.info(f"Initialized StratifiedSampler with {len(self.df)} records")

    def create_stratified_sample(
        self, sample_size: int, random_state: Optional[int] = 42
    ) -> pd.DataFrame:
        """
        Create a stratified sample maintaining product distribution.

        Args:
            sample_size: Target number of samples to create.
            random_state: Random seed for reproducibility.

        Returns:
            pd.DataFrame: Stratified sample of the data.

        Raises:
            ValueError: If sample_size is invalid or exceeds dataset size.
        """
        if sample_size <= 0:
            raise ValueError("Sample size must be positive")

        if sample_size > len(self.df):
            raise ValueError(
                f"Sample size {sample_size} exceeds dataset size {len(self.df)}"
            )

        logger.info(f"Creating stratified sample of size {sample_size}")

        # Calculate sampling fraction
        frac = sample_size / len(self.df)

        # Perform stratified sampling
        sampled_df = (
            self.df.groupby(self.product_column, group_keys=False)
            .apply(
                lambda x: x.sample(frac=frac, random_state=random_state),
                include_groups=True,
            )
            .reset_index(drop=True)
        )

        # If we didn't get exactly the requested size due to rounding,
        # adjust by sampling additional rows or removing excess
        current_size = len(sampled_df)

        if current_size < sample_size:
            # Need more samples
            remaining = sample_size - current_size
            additional = self.df[~self.df.index.isin(sampled_df.index)].sample(
                n=remaining, random_state=random_state
            )
            sampled_df = pd.concat([sampled_df, additional]).reset_index(drop=True)
        elif current_size > sample_size:
            # Have too many samples
            sampled_df = sampled_df.sample(
                n=sample_size, random_state=random_state
            ).reset_index(drop=True)

        logger.info(f"Created sample with {len(sampled_df)} records")

        return sampled_df

    def validate_sample_distribution(
        self, sample_df: pd.DataFrame, tolerance: float = 0.05
    ) -> Dict:
        """
        Validate that sample maintains proportional distribution.

        Args:
            sample_df: The sampled DataFrame to validate.
            tolerance: Acceptable deviation from original distribution (default 5%).

        Returns:
            dict: Validation report containing:
                - original_distribution: Original product percentages
                - sample_distribution: Sample product percentages
                - deviations: Absolute differences
                - within_tolerance: Whether all deviations are acceptable
                - max_deviation: Maximum deviation observed
        """
        if sample_df is None or len(sample_df) == 0:
            raise ValueError("Sample DataFrame cannot be None or empty")

        # Calculate distributions
        original_dist = (
            self.df[self.product_column].value_counts(normalize=True).sort_index()
        )
        sample_dist = (
            sample_df[self.product_column].value_counts(normalize=True).sort_index()
        )

        # Ensure both have same products
        all_products = sorted(set(original_dist.index) | set(sample_dist.index))
        original_dist = original_dist.reindex(all_products, fill_value=0)
        sample_dist = sample_dist.reindex(all_products, fill_value=0)

        # Calculate deviations
        deviations = abs(original_dist - sample_dist)
        max_deviation = deviations.max()
        within_tolerance = (deviations <= tolerance).all()

        validation_report = {
            "original_distribution": original_dist.to_dict(),
            "sample_distribution": sample_dist.to_dict(),
            "deviations": deviations.to_dict(),
            "within_tolerance": within_tolerance,
            "max_deviation": max_deviation,
            "tolerance": tolerance,
        }

        # Log results
        logger.info(f"Validation complete:")
        logger.info(f"  Max deviation: {max_deviation:.4f}")
        logger.info(f"  Within tolerance ({tolerance}): {within_tolerance}")

        if not within_tolerance:
            logger.warning(f"Sample distribution deviates beyond tolerance!")
            for product, deviation in deviations.items():
                if deviation > tolerance:
                    logger.warning(f"  {product}: {deviation:.4f} (>{tolerance})")

        return validation_report

    def get_sample_summary(self, sample_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get a summary comparison of original vs sample distributions.

        Args:
            sample_df: The sampled DataFrame to summarize.

        Returns:
            pd.DataFrame: Summary table with original and sample counts/percentages.
        """
        original_counts = self.df[self.product_column].value_counts().sort_index()
        sample_counts = sample_df[self.product_column].value_counts().sort_index()

        summary = pd.DataFrame(
            {
                "Original_Count": original_counts,
                "Original_Pct": (original_counts / len(self.df) * 100),
                "Sample_Count": sample_counts,
                "Sample_Pct": (sample_counts / len(sample_df) * 100),
            }
        )

        summary["Difference_Pct"] = summary["Sample_Pct"] - summary["Original_Pct"]

        return summary
