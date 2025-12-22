"""
VizForge Synthetic Visualization Engine v2.0

Privacy-safe synthetic data generation for visualization testing,
demos, education, and benchmarking.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal


@dataclass
class SyntheticVizConfig:
    """Configuration for synthetic visualization generation."""
    seed: int = 42
    n_points: int = 200
    trend: float = 0.0  # Linear trend coefficient
    seasonality: float = 0.0  # Seasonal amplitude (0-1)
    noise: float = 0.1  # Noise level (0-1)
    anomaly_rate: float = 0.05  # Percentage of anomalies (0-1)
    cycles_per_year: int = 4  # For seasonality (4 = quarterly)


class SyntheticVisualizationEngine:
    """
    Generate privacy-safe synthetic data for visualization.

    Capabilities:
    - Time series with trend, seasonality, noise
    - Distribution matching (normal, lognormal, exponential)
    - Correlation preservation
    - Anomaly injection
    - Deterministic (seed-based)

    Use Cases:
    - Public demos without exposing real data
    - Educational materials
    - Testing and validation
    - Benchmarking visualization algorithms
    - Privacy-compliant data sharing
    """

    def __init__(self, config: SyntheticVizConfig = None):
        self.config = config or SyntheticVizConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def generate_time_series(
        self,
        trend: Optional[float] = None,
        seasonality: Optional[float] = None,
        noise: Optional[float] = None,
        anomaly_rate: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic time series with trend, seasonality, and anomalies.

        Args:
            trend: Override config trend
            seasonality: Override config seasonality
            noise: Override config noise
            anomaly_rate: Override config anomaly rate

        Returns:
            DataFrame with columns: t, value, is_anomaly
        """
        # Use provided values or fall back to config
        trend = trend if trend is not None else self.config.trend
        seasonality = seasonality if seasonality is not None else self.config.seasonality
        noise = noise if noise is not None else self.config.noise
        anomaly_rate = anomaly_rate if anomaly_rate is not None else self.config.anomaly_rate

        t = np.arange(self.config.n_points)

        # Components
        trend_component = trend * t
        seasonal_component = seasonality * np.sin(
            2 * np.pi * t / (self.config.n_points / self.config.cycles_per_year)
        )
        noise_component = self.rng.normal(0, noise, size=len(t))

        # Base series
        series = trend_component + seasonal_component + noise_component

        # Inject anomalies
        anomaly_mask = self.rng.random(len(t)) < anomaly_rate
        anomaly_multipliers = self.rng.uniform(1.5, 3.0, size=len(t))
        series = np.where(anomaly_mask, series * anomaly_multipliers, series)

        return pd.DataFrame({
            "t": t,
            "value": series,
            "is_anomaly": anomaly_mask
        })

    def generate_distribution(
        self,
        kind: Literal["normal", "lognormal", "exponential", "bimodal"] = "normal",
        params: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Generate synthetic data from specified distribution.

        Args:
            kind: Distribution type
            params: Distribution parameters

        Returns:
            Array of synthetic values
        """
        params = params or {}

        if kind == "normal":
            loc = params.get("loc", 0)
            scale = params.get("scale", 1)
            return self.rng.normal(loc, scale, self.config.n_points)

        elif kind == "lognormal":
            mean = params.get("mean", 0)
            sigma = params.get("sigma", 1)
            return self.rng.lognormal(mean, sigma, self.config.n_points)

        elif kind == "exponential":
            scale = params.get("scale", 1.0)
            return self.rng.exponential(scale, self.config.n_points)

        elif kind == "bimodal":
            # Mix of two normals
            loc1 = params.get("loc1", -2)
            loc2 = params.get("loc2", 2)
            scale1 = params.get("scale1", 0.5)
            scale2 = params.get("scale2", 0.5)
            mix_ratio = params.get("mix_ratio", 0.5)

            n1 = int(self.config.n_points * mix_ratio)
            n2 = self.config.n_points - n1

            component1 = self.rng.normal(loc1, scale1, n1)
            component2 = self.rng.normal(loc2, scale2, n2)

            combined = np.concatenate([component1, component2])
            self.rng.shuffle(combined)
            return combined

        else:
            raise ValueError(f"Unknown distribution kind: {kind}")

    def match_distribution(
        self,
        real_data: np.ndarray,
        preserve: List[str] = None
    ) -> np.ndarray:
        """
        Generate synthetic data matching real data distribution.

        Args:
            real_data: Real data to match
            preserve: Properties to preserve ["mean", "std", "skewness", "kurtosis"]

        Returns:
            Synthetic data matching specified properties
        """
        preserve = preserve or ["mean", "std"]

        # Start with random normal
        synthetic = self.rng.normal(0, 1, self.config.n_points)

        # Match requested moments
        if "mean" in preserve:
            target_mean = np.mean(real_data)
            synthetic = synthetic - np.mean(synthetic) + target_mean

        if "std" in preserve:
            target_std = np.std(real_data)
            current_std = np.std(synthetic)
            if current_std > 0:
                synthetic = (synthetic - np.mean(synthetic)) * (target_std / current_std) + np.mean(synthetic)

        if "skewness" in preserve or "kurtosis" in preserve:
            # Use rank-based matching for higher moments
            synthetic_sorted = np.sort(synthetic)
            real_sorted = np.sort(real_data)

            # Interpolate to match n_points
            if len(real_sorted) != self.config.n_points:
                real_sorted = np.interp(
                    np.linspace(0, len(real_sorted) - 1, self.config.n_points),
                    np.arange(len(real_sorted)),
                    real_sorted
                )

            synthetic = real_sorted[np.argsort(np.argsort(synthetic))]

        return synthetic

    def generate_correlation_matrix(
        self,
        n_features: int,
        correlation_strength: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate synthetic correlated features.

        Args:
            n_features: Number of features to generate
            correlation_strength: Average correlation strength (0-1)

        Returns:
            DataFrame with correlated synthetic features
        """
        # Generate random correlation matrix
        A = self.rng.normal(0, 1, (n_features, n_features))
        cov_matrix = A @ A.T

        # Scale to desired correlation strength
        D = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
        corr_matrix = D @ cov_matrix @ D
        corr_matrix = corr_matrix * correlation_strength + np.eye(n_features) * (1 - correlation_strength)

        # Generate data
        data = self.rng.multivariate_normal(
            mean=np.zeros(n_features),
            cov=corr_matrix,
            size=self.config.n_points
        )

        return pd.DataFrame(
            data,
            columns=[f"feature_{i}" for i in range(n_features)]
        )

    def inject_outliers(
        self,
        data: np.ndarray,
        outlier_rate: float = 0.05,
        outlier_magnitude: float = 3.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Inject outliers into data.

        Args:
            data: Original data
            outlier_rate: Fraction of outliers to inject
            outlier_magnitude: How many std devs from mean

        Returns:
            (modified_data, outlier_mask)
        """
        data = data.copy()
        outlier_mask = self.rng.random(len(data)) < outlier_rate

        mean = np.mean(data)
        std = np.std(data)

        # Random direction (positive or negative)
        directions = self.rng.choice([-1, 1], size=len(data))
        outlier_values = mean + directions * outlier_magnitude * std

        data[outlier_mask] = outlier_values[outlier_mask]

        return data, outlier_mask

    def generate_scenario(
        self,
        baseline: pd.DataFrame,
        scenario_type: Literal["growth", "decline", "shock", "recovery"],
        magnitude: float = 0.2,
        start_period: int = 0
    ) -> pd.DataFrame:
        """
        Generate scenario variations from baseline data.

        Args:
            baseline: Baseline time series DataFrame (must have 'value' column)
            scenario_type: Type of scenario to generate
            magnitude: Magnitude of change (0-1)
            start_period: When to start applying scenario

        Returns:
            DataFrame with scenario data
        """
        scenario_df = baseline.copy()
        n_periods = len(baseline) - start_period

        if scenario_type == "growth":
            # Exponential growth
            growth_factors = np.exp(magnitude * np.arange(n_periods) / n_periods)
            scenario_df.loc[start_period:, "value"] *= growth_factors

        elif scenario_type == "decline":
            # Exponential decline
            decline_factors = np.exp(-magnitude * np.arange(n_periods) / n_periods)
            scenario_df.loc[start_period:, "value"] *= decline_factors

        elif scenario_type == "shock":
            # Sudden drop then recovery
            shock_periods = min(int(n_periods * 0.3), 20)  # 30% of remaining or 20 periods
            shock_curve = np.concatenate([
                np.full(shock_periods, 1 - magnitude),  # Shock period
                np.linspace(1 - magnitude, 1, n_periods - shock_periods)  # Recovery
            ])
            scenario_df.loc[start_period:, "value"] *= shock_curve

        elif scenario_type == "recovery":
            # Gradual recovery to normal + overshoot
            recovery_curve = 1 - magnitude + magnitude * (1 + 0.2 * np.sin(np.linspace(0, np.pi, n_periods)))
            scenario_df.loc[start_period:, "value"] *= recovery_curve

        return scenario_df

    def export(
        self,
        data: pd.DataFrame,
        output_path: str,
        format: Literal["csv", "json", "parquet"] = "csv"
    ):
        """Export synthetic data to file."""
        if format == "csv":
            data.to_csv(output_path, index=False)
        elif format == "json":
            data.to_json(output_path, orient="records", indent=2)
        elif format == "parquet":
            data.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
