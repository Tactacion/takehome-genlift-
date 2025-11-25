"""
Feature engineering for churn prediction with strict temporal windowing.

Implements velocity-based features to capture behavioral decay and
friction metrics to identify product adoption issues.
"""

from typing import Tuple
import pandas as pd
import numpy as np
from datetime import timedelta

from config import Config


class ChurnFeatureEngineer:
    """
    Transforms variable-length event sequences into fixed-dimensional features.

    Uses strict temporal cutoff to prevent data leakage: observation window
    for features, prediction window for labeling.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.cutoff_date: pd.Timestamp | None = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute full feature engineering pipeline with temporal split.

        Args:
            df: Raw event stream with columns [customer_id, event_timestamp,
                event_type, event_value, label_churn_30d]

        Returns:
            Feature matrix with customer-level aggregations and binary labels.
        """
        df = self._prepare_timestamps(df)
        self.cutoff_date = self._calculate_cutoff_date(df)

        observation_df = self._filter_observation_window(df)
        prediction_df = self._filter_prediction_window(df)

        features = self._build_feature_matrix(observation_df)
        labels = self._extract_labels(observation_df, prediction_df)

        return features.merge(labels, on='customer_id', how='left').fillna(0)

    def _prepare_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert timestamp strings to datetime and sort chronologically."""
        df = df.copy()
        df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
        return df.sort_values('event_timestamp')

    def _calculate_cutoff_date(self, df: pd.DataFrame) -> pd.Timestamp:
        """
        Determine observation/prediction boundary.

        Cutoff = max_date - prediction_window ensures we have ground truth
        for labeling without using future information for features.
        """
        max_date = df['event_timestamp'].max()
        return max_date - timedelta(days=self.config.temporal.prediction_window_days)

    def _filter_observation_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract events before cutoff for feature calculation.

        Further restrict to last N days before cutoff to focus on recent behavior.
        """
        observation_start = self.cutoff_date - timedelta(
            days=self.config.temporal.observation_window_days
        )
        mask = (df['event_timestamp'] > observation_start) & (
            df['event_timestamp'] <= self.cutoff_date
        )
        return df[mask].copy()

    def _filter_prediction_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract events after cutoff for label assignment."""
        mask = df['event_timestamp'] > self.cutoff_date
        return df[mask].copy()

    def _build_feature_matrix(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate event streams into fixed-dimensional feature vectors.

        Combines intensity, velocity, recency, and friction features at
        customer level using vectorized pandas operations.
        """
        intensity_features = self._compute_intensity_features(obs_df)
        velocity_features = self._compute_velocity_features(obs_df)
        recency_features = self._compute_recency_features(obs_df)
        friction_features = self._compute_friction_features(obs_df)

        all_features = [
            intensity_features,
            velocity_features,
            recency_features,
            friction_features,
        ]

        result = all_features[0]
        for features in all_features[1:]:
            result = result.merge(features, on='customer_id', how='outer')

        return result.fillna(0)

    def _compute_intensity_features(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Raw volume metrics: total activity and engagement depth.

        High-value events (e.g., API calls, exports) signal power users.
        """
        agg_dict = {
            'event_timestamp': 'count',
            'event_value': 'sum',
            'event_type': 'nunique',
        }

        intensity = obs_df.groupby('customer_id').agg(agg_dict).reset_index()
        intensity.columns = [
            'customer_id',
            'f_total_events',
            'f_total_event_value',
            'f_unique_event_types',
        ]

        event_type_counts = obs_df.pivot_table(
            index='customer_id',
            columns='event_type',
            values='event_timestamp',
            aggfunc='count',
            fill_value=0,
        ).reset_index()

        event_type_counts.columns = [
            'customer_id' if col == 'customer_id' else f'f_count_{col}'
            for col in event_type_counts.columns
        ]

        return intensity.merge(event_type_counts, on='customer_id', how='left').fillna(0)

    def _compute_velocity_features(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Behavioral decay indicators via recent vs historical activity ratio.

        A power user dropping to 20% of their historical rate is high-risk.
        This captures churn as a process, not a state.
        """
        l7_cutoff = self.cutoff_date - timedelta(
            days=self.config.temporal.velocity_short_window_days
        )

        l7_events = obs_df[obs_df['event_timestamp'] > l7_cutoff]
        l7_counts = (
            l7_events.groupby('customer_id')
            .size()
            .reset_index(name='activity_l7')
        )

        l28_counts = (
            obs_df.groupby('customer_id')
            .size()
            .reset_index(name='activity_l28')
        )

        velocity_df = l7_counts.merge(l28_counts, on='customer_id', how='outer').fillna(0)

        velocity_df['f_velocity_ratio'] = np.where(
            velocity_df['activity_l28'] > 0,
            (velocity_df['activity_l7'] * self.config.temporal.velocity_multiplier)
            / velocity_df['activity_l28'],
            0,
        )

        l7_logins = (
            l7_events[l7_events['event_type'] == 'login']
            .groupby('customer_id')
            .size()
            .reset_index(name='logins_l7')
        )

        l28_logins = (
            obs_df[obs_df['event_type'] == 'login']
            .groupby('customer_id')
            .size()
            .reset_index(name='logins_l28')
        )

        login_velocity = l7_logins.merge(l28_logins, on='customer_id', how='outer').fillna(0)

        login_velocity['f_login_velocity_ratio'] = np.where(
            login_velocity['logins_l28'] > 0,
            (login_velocity['logins_l7'] * self.config.temporal.velocity_multiplier)
            / login_velocity['logins_l28'],
            0,
        )

        velocity_df = velocity_df.merge(
            login_velocity[['customer_id', 'f_login_velocity_ratio']],
            on='customer_id',
            how='outer',
        ).fillna(0)

        return velocity_df[['customer_id', 'f_velocity_ratio', 'f_login_velocity_ratio']]

    def _compute_recency_features(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Days since last activity: single best predictor of imminent churn.

        14+ days of silence is a red flag regardless of historical usage.
        """
        last_event = (
            obs_df.groupby('customer_id')['event_timestamp']
            .max()
            .reset_index()
        )

        last_event['f_days_since_last_event'] = (
            self.cutoff_date - last_event['event_timestamp']
        ).dt.total_seconds() / 86400

        return last_event[['customer_id', 'f_days_since_last_event']]

    def _compute_friction_features(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Product adoption quality metrics.

        High support-ticket-to-login ratio indicates users are stuck.
        High error rate suggests technical issues driving churn.
        """
        support_tickets = (
            obs_df[obs_df['event_type'] == 'support_ticket']
            .groupby('customer_id')
            .size()
            .reset_index(name='support_tickets')
        )

        logins = (
            obs_df[obs_df['event_type'] == 'login']
            .groupby('customer_id')
            .size()
            .reset_index(name='logins')
        )

        errors = (
            obs_df[obs_df['event_type'] == 'error']
            .groupby('customer_id')
            .size()
            .reset_index(name='errors')
        )

        total_events = (
            obs_df.groupby('customer_id')
            .size()
            .reset_index(name='total_events')
        )

        friction = (
            support_tickets.merge(logins, on='customer_id', how='outer')
            .merge(errors, on='customer_id', how='outer')
            .merge(total_events, on='customer_id', how='outer')
            .fillna(0)
        )

        alpha = self.config.training.smoothing_alpha

        friction['f_friction_index'] = (friction['support_tickets'] + alpha) / (
            friction['logins'] + alpha
        )

        friction['f_error_rate'] = friction['errors'] / (friction['total_events'] + 1e-6)

        return friction[['customer_id', 'f_friction_index', 'f_error_rate']]

    def _extract_labels(
        self, obs_df: pd.DataFrame, pred_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Binary churn labels from prediction window.

        Customer churned = no events in 30 days after cutoff.
        Inverts presence logic: if we see them, they didn't churn.
        """
        all_customers = obs_df[['customer_id']].drop_duplicates()
        active_in_prediction = pred_df[['customer_id']].drop_duplicates()
        active_in_prediction['had_activity'] = 1

        labels = all_customers.merge(
            active_in_prediction, on='customer_id', how='left'
        )

        labels['label_churn_30d'] = (labels['had_activity'].isna()).astype(int)

        return labels[['customer_id', 'label_churn_30d']]
