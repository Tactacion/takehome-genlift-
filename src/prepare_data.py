"""
Data preparation script: transforms raw event streams into customer-level features.

Usage:
    python src/prepare_data.py --events-path events.csv
"""

import argparse
from pathlib import Path
import pandas as pd

from config import Config
from preprocessing import ChurnFeatureEngineer


def main(events_path: Path, output_path: Path | None = None) -> None:
    """
    Load raw events and generate customer-level feature matrix.

    Args:
        events_path: Path to events.csv with raw event stream
        output_path: Optional output path (defaults to data/customers.parquet)
    """
    config = Config()

    if output_path is None:
        output_path = config.paths.processed_data

    print("="*60)
    print("Data Preparation Pipeline")
    print("="*60)
    print(f"\nInput:  {events_path}")
    print(f"Output: {output_path}")

    df = pd.read_csv(events_path)
    print(f"\n[+] Loaded {len(df):,} events from {df['customer_id'].nunique():,} customers")

    engineer = ChurnFeatureEngineer(config)
    features_df = engineer.fit_transform(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)

    print(f"\n[+] Features saved")
    print(f"    Shape: {features_df.shape}")
    print(f"    Customers: {len(features_df):,}")
    print(f"    Churn rate: {features_df['label_churn_30d'].mean():.1%}")
    print(f"    Features: {len([c for c in features_df.columns if c.startswith('f_')])}")

    print(f"\n[+] Complete. Ready for training.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare customer-level features from raw event data'
    )
    parser.add_argument(
        '--events-path',
        type=Path,
        required=True,
        help='Path to events.csv',
    )
    parser.add_argument(
        '--output-path',
        type=Path,
        default=None,
        help='Output path for features (default: data/customers.parquet)',
    )

    args = parser.parse_args()
    main(events_path=args.events_path, output_path=args.output_path)
