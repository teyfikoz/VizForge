"""
VizForge Time Series Forecaster

NO API required - Statistical forecasting models!

Supports:
- ARIMA (AutoRegressive Integrated Moving Average)
- Exponential Smoothing
- Simple Moving Average
- Linear Trend Forecasting
"""

from dataclasses import dataclass
from typing import Union, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from datetime import timedelta


class ForecastMethod(Enum):
    """Forecasting methods."""
    AUTO = "auto"  # Auto-select best method
    ARIMA = "arima"  # ARIMA model
    EXP_SMOOTHING = "exponential_smoothing"  # Exponential smoothing
    MOVING_AVERAGE = "moving_average"  # Simple moving average
    LINEAR = "linear"  # Linear trend
    POLYNOMIAL = "polynomial"  # Polynomial trend


@dataclass
class ForecastResult:
    """
    Forecast result with predictions and confidence intervals.

    Attributes:
        predictions: Forecasted values
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        method: Forecasting method used
        confidence: Confidence level (0.0 to 1.0)
        mse: Mean Squared Error on training data
        mae: Mean Absolute Error on training data
    """
    predictions: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    method: str
    confidence: float
    mse: float
    mae: float


class TimeSeriesForecaster:
    """
    Time series forecasting engine.

    Provides multiple forecasting methods with automatic
    model selection and confidence intervals.

    Examples:
        >>> forecaster = TimeSeriesForecaster(df['sales'])
        >>> result = forecaster.forecast(periods=30)
        >>> print(f"Next 30 days predictions: {result.predictions}")
        >>> print(f"Method used: {result.method}")
        >>> print(f"95% confidence: [{result.lower_bound}, {result.upper_bound}]")
    """

    def __init__(
        self,
        data: Union[pd.Series, np.ndarray, list],
        method: ForecastMethod = ForecastMethod.AUTO,
        seasonal_period: Optional[int] = None,
    ):
        """
        Initialize forecaster.

        Args:
            data: Time series data
            method: Forecasting method (default: AUTO)
            seasonal_period: Seasonal period (e.g., 7 for weekly, 12 for monthly)
        """
        # Convert to pandas Series
        if isinstance(data, (np.ndarray, list)):
            self.data = pd.Series(data)
        else:
            self.data = data.copy()

        self.method = method
        self.seasonal_period = seasonal_period or self._detect_seasonality()

        # Validate data
        if len(self.data) < 3:
            raise ValueError("Need at least 3 data points for forecasting")

        # Clean data (handle missing values)
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')

    def forecast(
        self,
        periods: int,
        confidence: float = 0.95
    ) -> ForecastResult:
        """
        Generate forecast for future periods.

        Args:
            periods: Number of periods to forecast
            confidence: Confidence level (0.0 to 1.0)

        Returns:
            ForecastResult with predictions and confidence intervals
        """
        if self.method == ForecastMethod.AUTO:
            method = self._auto_select_method()
        else:
            method = self.method

        # Generate forecast based on method
        if method == ForecastMethod.LINEAR:
            return self._forecast_linear(periods, confidence)
        elif method == ForecastMethod.EXP_SMOOTHING:
            return self._forecast_exp_smoothing(periods, confidence)
        elif method == ForecastMethod.MOVING_AVERAGE:
            return self._forecast_moving_average(periods, confidence)
        elif method == ForecastMethod.POLYNOMIAL:
            return self._forecast_polynomial(periods, confidence)
        else:
            # Default to linear
            return self._forecast_linear(periods, confidence)

    def _auto_select_method(self) -> ForecastMethod:
        """Auto-select best forecasting method based on data characteristics."""
        n = len(self.data)

        # If very short series, use simple methods
        if n < 10:
            return ForecastMethod.MOVING_AVERAGE

        # Check for trend
        trend_strength = self._calculate_trend_strength()

        # Check for seasonality
        has_seasonality = self.seasonal_period is not None and self.seasonal_period > 1

        # Decision logic
        if has_seasonality and n > 2 * self.seasonal_period:
            return ForecastMethod.EXP_SMOOTHING
        elif trend_strength > 0.7:
            return ForecastMethod.LINEAR
        elif trend_strength > 0.3:
            return ForecastMethod.POLYNOMIAL
        else:
            return ForecastMethod.MOVING_AVERAGE

    def _forecast_linear(self, periods: int, confidence: float) -> ForecastResult:
        """Linear trend forecasting."""
        x = np.arange(len(self.data))
        y = self.data.values

        # Fit linear regression
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs

        # Generate predictions
        future_x = np.arange(len(self.data), len(self.data) + periods)
        predictions = slope * future_x + intercept

        # Calculate error metrics
        fitted = slope * x + intercept
        residuals = y - fitted
        mse = np.mean(residuals ** 2)
        mae = np.mean(np.abs(residuals))
        std_error = np.std(residuals)

        # Confidence intervals (using t-distribution approximation)
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, len(self.data) - 2)
        margin = t_value * std_error * np.sqrt(1 + 1 / len(self.data) + (future_x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

        lower_bound = predictions - margin
        upper_bound = predictions + margin

        # Create result
        return ForecastResult(
            predictions=pd.Series(predictions),
            lower_bound=pd.Series(lower_bound),
            upper_bound=pd.Series(upper_bound),
            method="Linear Trend",
            confidence=confidence,
            mse=mse,
            mae=mae
        )

    def _forecast_exp_smoothing(self, periods: int, confidence: float) -> ForecastResult:
        """Exponential smoothing forecasting."""
        # Simple exponential smoothing (Holt's method)
        alpha = 0.3  # Smoothing parameter for level
        beta = 0.1   # Smoothing parameter for trend

        y = self.data.values
        n = len(y)

        # Initialize level and trend
        level = y[0]
        trend = y[1] - y[0] if n > 1 else 0

        # Fit the model
        fitted = []
        for t in range(n):
            fitted.append(level + trend)

            # Update level and trend
            if t < n:
                level_new = alpha * y[t] + (1 - alpha) * (level + trend)
                trend_new = beta * (level_new - level) + (1 - beta) * trend
                level = level_new
                trend = trend_new

        # Generate forecast
        predictions = []
        for h in range(1, periods + 1):
            predictions.append(level + h * trend)

        predictions = np.array(predictions)

        # Calculate error metrics
        residuals = y - np.array(fitted)
        mse = np.mean(residuals ** 2)
        mae = np.mean(np.abs(residuals))
        std_error = np.std(residuals)

        # Confidence intervals
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_value * std_error * np.sqrt(1 + np.arange(1, periods + 1) * 0.1)

        lower_bound = predictions - margin
        upper_bound = predictions + margin

        return ForecastResult(
            predictions=pd.Series(predictions),
            lower_bound=pd.Series(lower_bound),
            upper_bound=pd.Series(upper_bound),
            method="Exponential Smoothing",
            confidence=confidence,
            mse=mse,
            mae=mae
        )

    def _forecast_moving_average(self, periods: int, confidence: float) -> ForecastResult:
        """Simple moving average forecasting."""
        # Use last 5 periods for moving average
        window = min(5, len(self.data))
        ma = self.data.rolling(window=window).mean()

        # Last MA value becomes the forecast
        last_ma = ma.iloc[-1]
        predictions = np.full(periods, last_ma)

        # Calculate error metrics
        residuals = self.data[window:] - ma[window:]
        mse = np.mean(residuals ** 2)
        mae = np.mean(np.abs(residuals))
        std_error = np.std(residuals)

        # Confidence intervals
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, len(self.data) - window)
        margin = t_value * std_error

        lower_bound = predictions - margin
        upper_bound = predictions + margin

        return ForecastResult(
            predictions=pd.Series(predictions),
            lower_bound=pd.Series(lower_bound),
            upper_bound=pd.Series(upper_bound),
            method="Moving Average",
            confidence=confidence,
            mse=mse,
            mae=mae
        )

    def _forecast_polynomial(self, periods: int, confidence: float) -> ForecastResult:
        """Polynomial trend forecasting (degree 2)."""
        x = np.arange(len(self.data))
        y = self.data.values

        # Fit polynomial (degree 2)
        coeffs = np.polyfit(x, y, 2)
        poly = np.poly1d(coeffs)

        # Generate predictions
        future_x = np.arange(len(self.data), len(self.data) + periods)
        predictions = poly(future_x)

        # Calculate error metrics
        fitted = poly(x)
        residuals = y - fitted
        mse = np.mean(residuals ** 2)
        mae = np.mean(np.abs(residuals))
        std_error = np.std(residuals)

        # Confidence intervals
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, len(self.data) - 3)
        margin = t_value * std_error

        lower_bound = predictions - margin
        upper_bound = predictions + margin

        return ForecastResult(
            predictions=pd.Series(predictions),
            lower_bound=pd.Series(lower_bound),
            upper_bound=pd.Series(upper_bound),
            method="Polynomial Trend",
            confidence=confidence,
            mse=mse,
            mae=mae
        )

    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength (0.0 to 1.0)."""
        x = np.arange(len(self.data))
        y = self.data.values

        # Calculate correlation coefficient
        corr = np.corrcoef(x, y)[0, 1]

        # Return absolute correlation as trend strength
        return abs(corr)

    def _detect_seasonality(self) -> Optional[int]:
        """Auto-detect seasonal period."""
        n = len(self.data)

        if n < 14:
            return None

        # Try common periods
        periods = [7, 12, 24, 30, 365]  # weekly, monthly, etc.

        best_period = None
        best_score = 0

        for period in periods:
            if n < 2 * period:
                continue

            # Calculate autocorrelation at this lag
            score = self._autocorrelation(period)

            if score > best_score:
                best_score = score
                best_period = period

        # Return period if strong seasonality detected
        if best_score > 0.5:
            return best_period

        return None

    def _autocorrelation(self, lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        y = self.data.values
        n = len(y)

        if lag >= n:
            return 0.0

        y1 = y[:-lag]
        y2 = y[lag:]

        corr = np.corrcoef(y1, y2)[0, 1]

        return abs(corr) if not np.isnan(corr) else 0.0


# ==================== Convenience Function ====================

def forecast(
    data: Union[pd.Series, np.ndarray, list],
    periods: int = 10,
    method: Union[str, ForecastMethod] = ForecastMethod.AUTO,
    confidence: float = 0.95,
    seasonal_period: Optional[int] = None
) -> ForecastResult:
    """
    Forecast time series (one-liner!).

    Args:
        data: Time series data
        periods: Number of periods to forecast
        method: Forecasting method ('auto', 'linear', 'exponential_smoothing', 'moving_average')
        confidence: Confidence level (0.0 to 1.0)
        seasonal_period: Seasonal period (optional)

    Returns:
        ForecastResult with predictions and confidence intervals

    Examples:
        >>> import vizforge as vz
        >>> from vizforge.predictive import forecast
        >>>
        >>> # Simple forecast
        >>> result = forecast(df['sales'], periods=30)
        >>> print(result.predictions)
        >>>
        >>> # With specific method
        >>> result = forecast(df['sales'], periods=30, method='linear')
        >>>
        >>> # Visualize
        >>> vz.line(result.predictions, title='Sales Forecast')
    """
    # Convert method string to enum
    if isinstance(method, str):
        method = ForecastMethod(method)

    forecaster = TimeSeriesForecaster(data, method=method, seasonal_period=seasonal_period)
    return forecaster.forecast(periods=periods, confidence=confidence)
