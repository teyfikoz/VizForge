"""
VizForge Intelligence Module Tests

Tests for chart selection, data profiling, insights, and color optimization.
Target: 95%+ coverage for intelligence module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

# Import intelligence module components
from ..intelligence.chart_selector import ChartSelector, ChartRecommendation
from ..intelligence.data_profiler import DataProfiler, DataProfile, DataQualityScorer
from ..intelligence.insights_engine import InsightsEngine, Insight, InsightType, Severity
from ..intelligence.color_optimizer import ColorOptimizer
from ..intelligence.recommendation import RecommendationEngine, Recommendation, RecommendationType


# ==================== Fixtures ====================

@pytest.fixture
def sample_timeseries_data():
    """Create sample time series data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'date': dates,
        'sales': np.random.randint(100, 1000, 100),
        'revenue': np.random.uniform(1000, 5000, 100),
    })


@pytest.fixture
def sample_categorical_data():
    """Create sample categorical data."""
    return pd.DataFrame({
        'category': ['A', 'B', 'C', 'D', 'E'] * 20,
        'value': np.random.randint(10, 100, 100),
        'count': np.random.randint(1, 50, 100),
    })


@pytest.fixture
def sample_geographic_data():
    """Create sample geographic data."""
    return pd.DataFrame({
        'country': ['USA', 'Canada', 'UK', 'Germany', 'France'] * 20,
        'city': ['New York', 'Toronto', 'London', 'Berlin', 'Paris'] * 20,
        'latitude': [40.7, 43.7, 51.5, 52.5, 48.9] * 20,
        'longitude': [-74.0, -79.4, -0.1, 13.4, 2.4] * 20,
        'sales': np.random.randint(1000, 10000, 100),
    })


@pytest.fixture
def sample_correlation_data():
    """Create sample data with correlations."""
    x = np.random.randn(100)
    return pd.DataFrame({
        'x': x,
        'y_correlated': x * 2 + np.random.randn(100) * 0.5,  # Strong correlation
        'y_uncorrelated': np.random.randn(100),  # No correlation
        'z': np.random.randint(1, 100, 100),
    })


# ==================== ChartSelector Tests ====================

class TestChartSelector:
    """Tests for ChartSelector class."""

    def test_initialization(self):
        """Test ChartSelector initialization."""
        selector = ChartSelector()
        assert selector is not None
        assert hasattr(selector, 'GEO_KEYWORDS')
        assert hasattr(selector, 'TEMPORAL_KEYWORDS')

    def test_recommend_timeseries(self, sample_timeseries_data):
        """Test recommendation for time series data."""
        selector = ChartSelector()
        result = selector.recommend(sample_timeseries_data, x='date', y='sales')

        assert result is not None
        assert 'primary' in result
        assert result['primary'] in ['LineChart', 'AreaChart', 'TimeSeriesChart']
        assert result['confidence'] >= 0.85  # High confidence for obvious time series

    def test_recommend_geographic(self, sample_geographic_data):
        """Test recommendation for geographic data."""
        selector = ChartSelector()
        result = selector.recommend(sample_geographic_data, x='latitude', y='longitude')

        assert result is not None
        assert 'primary' in result
        assert result['primary'] in ['ChoroplethMap', 'ScatterGeoMap']
        assert result['confidence'] >= 0.90  # Very high confidence for geo data

    def test_recommend_categorical(self, sample_categorical_data):
        """Test recommendation for categorical data."""
        selector = ChartSelector()
        result = selector.recommend(sample_categorical_data, x='category', y='value')

        assert result is not None
        assert 'primary' in result
        assert result['primary'] in ['BarChart', 'ColumnChart', 'PieChart']

    def test_recommend_correlation(self, sample_correlation_data):
        """Test recommendation for correlation analysis."""
        selector = ChartSelector()
        result = selector.recommend(
            sample_correlation_data,
            x='x',
            y='y_correlated',
            intent='correlation'
        )

        assert result is not None
        assert 'primary' in result
        assert result['primary'] in ['ScatterChart', 'HeatmapChart']

    def test_alternatives_provided(self, sample_timeseries_data):
        """Test that alternatives are provided."""
        selector = ChartSelector()
        result = selector.recommend(sample_timeseries_data, x='date', y='sales')

        assert 'alternatives' in result
        assert isinstance(result['alternatives'], list)
        assert len(result['alternatives']) >= 1

    def test_reasoning_provided(self, sample_timeseries_data):
        """Test that reasoning is provided."""
        selector = ChartSelector()
        result = selector.recommend(sample_timeseries_data, x='date', y='sales')

        assert 'reasoning' in result
        assert isinstance(result['reasoning'], str)
        assert len(result['reasoning']) > 0

    def test_confidence_range(self, sample_timeseries_data):
        """Test confidence scores are in valid range."""
        selector = ChartSelector()
        result = selector.recommend(sample_timeseries_data, x='date', y='sales')

        assert 'confidence' in result
        assert 0.0 <= result['confidence'] <= 1.0


# ==================== DataProfiler Tests ====================

class TestDataProfiler:
    """Tests for DataProfiler class."""

    def test_profile_timeseries(self, sample_timeseries_data):
        """Test profiling time series data."""
        profiler = DataProfiler()
        profile = profiler.profile(sample_timeseries_data)

        assert profile is not None
        assert profile.n_rows == 100
        assert profile.n_cols == 3
        assert len(profile.temporal_cols) >= 1  # 'date' column
        assert len(profile.numeric_cols) >= 2   # 'sales', 'revenue'

    def test_profile_geographic(self, sample_geographic_data):
        """Test profiling geographic data."""
        profiler = DataProfiler()
        profile = profiler.profile(sample_geographic_data)

        assert profile is not None
        assert profile.has_geo is True
        assert len(profile.categorical_cols) >= 2  # 'country', 'city'

    def test_profile_empty_dataframe(self):
        """Test profiling empty DataFrame."""
        profiler = DataProfiler()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError):
            profiler.profile(empty_df)

    def test_cardinality_detection(self, sample_categorical_data):
        """Test cardinality detection."""
        profiler = DataProfiler()
        profile = profiler.profile(sample_categorical_data)

        # 'category' has 5 unique values
        assert 'category' in profile.categorical_cols

    def test_missing_values_detection(self):
        """Test missing values detection."""
        df = pd.DataFrame({
            'a': [1, 2, None, 4, 5],
            'b': [1, 2, 3, 4, 5],
        })

        profiler = DataProfiler()
        profile = profiler.profile(df)

        assert profile.missing_values['a'] > 0
        assert profile.missing_values['b'] == 0


class TestDataQualityScorer:
    """Tests for DataQualityScorer class."""

    def test_score_clean_data(self):
        """Test scoring clean data."""
        df = pd.DataFrame({
            'a': range(100),
            'b': range(100, 200),
            'c': ['x', 'y', 'z'] * 33 + ['x'],
        })

        scorer = DataQualityScorer()
        report = scorer.score(df)

        assert report.score >= 80  # Clean data should score high
        assert 0 <= report.score <= 100

    def test_score_dirty_data(self):
        """Test scoring dirty data."""
        df = pd.DataFrame({
            'a': [1, 2, None, None, 5] * 20,  # 40% missing
            'b': [1, 1, 1, 1, 1] * 20,        # No variance
            'c': list(range(100)),            # All unique (duplicates issue)
        })

        scorer = DataQualityScorer()
        report = scorer.score(df)

        assert report.score < 80  # Dirty data should score lower

    def test_completeness_score(self):
        """Test completeness scoring."""
        df = pd.DataFrame({
            'a': [1, 2, None, 4, 5],
            'b': [1, 2, 3, 4, 5],
        })

        scorer = DataQualityScorer()
        report = scorer.score(df)

        # Completeness should be affected by 20% missing values
        assert report.completeness < 100


# ==================== InsightsEngine Tests ====================

class TestInsightsEngine:
    """Tests for InsightsEngine class."""

    def test_detect_trends(self, sample_timeseries_data):
        """Test trend detection."""
        # Create data with clear upward trend
        df = pd.DataFrame({
            'value': range(100),  # Perfect linear trend
        })

        engine = InsightsEngine()
        insights = engine.generate_insights(df, target_column='value')

        # Should detect upward trend
        trend_insights = [i for i in insights if i.insight_type == InsightType.TREND]
        assert len(trend_insights) > 0
        assert trend_insights[0].confidence > 0.8

    def test_detect_correlations(self, sample_correlation_data):
        """Test correlation detection."""
        engine = InsightsEngine()
        insights = engine.generate_insights(sample_correlation_data, target_column='x')

        # Should detect correlation between x and y_correlated
        corr_insights = [i for i in insights if i.insight_type == InsightType.CORRELATION]
        assert len(corr_insights) > 0

    def test_detect_anomalies(self):
        """Test anomaly detection."""
        # Create data with outliers
        normal_data = np.random.randn(95)
        outliers = np.array([10, -10, 15, -15, 20])
        df = pd.DataFrame({
            'value': np.concatenate([normal_data, outliers])
        })

        engine = InsightsEngine()
        insights = engine.generate_insights(df, target_column='value')

        # Should detect anomalies
        anomaly_insights = [i for i in insights if i.insight_type == InsightType.ANOMALY]
        assert len(anomaly_insights) > 0

    def test_insight_severity_ordering(self, sample_timeseries_data):
        """Test insights are sorted by severity."""
        engine = InsightsEngine()
        insights = engine.generate_insights(sample_timeseries_data)

        if len(insights) > 1:
            # Check severity ordering (HIGH > MEDIUM > LOW)
            severities = [i.severity for i in insights]
            severity_order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2, Severity.INFO: 3}
            severity_values = [severity_order[s] for s in severities]

            # Should be sorted (non-increasing)
            for i in range(len(severity_values) - 1):
                assert severity_values[i] <= severity_values[i + 1]

    def test_confidence_scores(self, sample_timeseries_data):
        """Test confidence scores are valid."""
        engine = InsightsEngine()
        insights = engine.generate_insights(sample_timeseries_data)

        for insight in insights:
            assert 0.0 <= insight.confidence <= 1.0


# ==================== ColorOptimizer Tests ====================

class TestColorOptimizer:
    """Tests for ColorOptimizer class."""

    def test_get_palette(self):
        """Test getting predefined palettes."""
        optimizer = ColorOptimizer()

        palette = optimizer.get_palette('colorblind_safe')
        assert palette is not None
        assert len(palette) > 0
        assert all(color.startswith('#') for color in palette)

    def test_all_palettes_available(self):
        """Test all palettes are accessible."""
        optimizer = ColorOptimizer()

        for palette_name in ['colorblind_safe', 'high_contrast', 'tableau', 'material', 'pastel']:
            palette = optimizer.get_palette(palette_name)
            assert palette is not None
            assert len(palette) > 0

    def test_generate_palette(self):
        """Test generating custom palette."""
        optimizer = ColorOptimizer()

        palette = optimizer.generate_palette(n_colors=5, colorblind_safe=True)
        assert len(palette) == 5
        assert all(color.startswith('#') for color in palette)

    def test_calculate_contrast(self):
        """Test contrast calculation."""
        optimizer = ColorOptimizer()

        # Black and white should have maximum contrast
        contrast = optimizer.calculate_contrast('#000000', '#FFFFFF')
        assert contrast >= 20  # WCAG AAA requires 7:1, black/white is 21:1

        # Same color should have minimum contrast
        contrast = optimizer.calculate_contrast('#FF0000', '#FF0000')
        assert contrast == 1.0

    def test_optimize_for_background(self):
        """Test color optimization for background."""
        optimizer = ColorOptimizer()

        colors = ['#FF0000', '#00FF00', '#0000FF']
        optimized = optimizer.optimize_for_background(colors, background='#FFFFFF')

        assert len(optimized) == len(colors)
        # All colors should have sufficient contrast with white background
        for color in optimized:
            contrast = optimizer.calculate_contrast(color, '#FFFFFF')
            assert contrast >= 4.5  # WCAG AA minimum


# ==================== RecommendationEngine Tests ====================

class TestRecommendationEngine:
    """Tests for RecommendationEngine class."""

    def test_generate_recommendations(self, sample_timeseries_data):
        """Test recommendation generation."""
        engine = RecommendationEngine()
        recommendations = engine.generate_recommendations(sample_timeseries_data)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_recommendation_types(self, sample_timeseries_data):
        """Test different recommendation types are generated."""
        engine = RecommendationEngine()
        recommendations = engine.generate_recommendations(sample_timeseries_data)

        # Should have multiple types
        types = set(r.recommendation_type for r in recommendations)
        assert len(types) > 0

    def test_recommendation_confidence(self, sample_timeseries_data):
        """Test recommendations have valid confidence scores."""
        engine = RecommendationEngine()
        recommendations = engine.generate_recommendations(sample_timeseries_data)

        for rec in recommendations:
            assert 0.0 <= rec.confidence <= 1.0


# ==================== Integration Tests ====================

class TestIntelligenceIntegration:
    """Integration tests for intelligence module."""

    def test_full_workflow(self, sample_timeseries_data):
        """Test complete intelligence workflow."""
        # Step 1: Profile data
        profiler = DataProfiler()
        profile = profiler.profile(sample_timeseries_data)
        assert profile is not None

        # Step 2: Score quality
        scorer = DataQualityScorer()
        quality_report = scorer.score(sample_timeseries_data)
        assert quality_report.score > 0

        # Step 3: Get chart recommendation
        selector = ChartSelector()
        chart_rec = selector.recommend(sample_timeseries_data, x='date', y='sales')
        assert chart_rec is not None

        # Step 4: Generate insights
        engine = InsightsEngine()
        insights = engine.generate_insights(sample_timeseries_data)
        assert isinstance(insights, list)

        # Step 5: Optimize colors
        optimizer = ColorOptimizer()
        palette = optimizer.get_palette('colorblind_safe')
        assert len(palette) > 0

    def test_auto_chart_workflow(self, sample_timeseries_data):
        """Test automatic chart creation workflow."""
        # This simulates vz.auto_chart(df) workflow
        selector = ChartSelector()
        result = selector.recommend(sample_timeseries_data)

        assert result['primary'] is not None
        assert result['confidence'] > 0.5
        assert len(result['alternatives']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
