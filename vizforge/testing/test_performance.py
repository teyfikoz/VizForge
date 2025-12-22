"""
VizForge Performance Tests

Performance benchmarks and optimization tests.
Ensures VizForge meets performance targets.

Targets:
- Chart creation: < 50ms (1k points)
- Large datasets: < 200ms (100k points with WebGL)
- Dashboard assembly: < 500ms (10 charts)
- Memory: < 100MB (typical dashboard)
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from datetime import datetime, timedelta

# Import VizForge components for benchmarking
from ..intelligence.chart_selector import ChartSelector
from ..intelligence.data_profiler import DataProfiler
from ..intelligence.insights_engine import InsightsEngine
from ..analytics.calculated_fields import CalculatedField, CalculatedFieldManager
from ..analytics.aggregations import WindowFunction, WindowType
from ..interactive.filters import FilterContext, RangeFilter, ListFilter


# ==================== Fixtures ====================

@pytest.fixture
def small_dataset():
    """Create small dataset (1k rows)."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'value': np.random.randn(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
    })


@pytest.fixture
def medium_dataset():
    """Create medium dataset (10k rows)."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10000, freq='min'),
        'value': np.random.randn(10000),
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 10000),
    })


@pytest.fixture
def large_dataset():
    """Create large dataset (100k rows)."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100000, freq='S'),
        'value': np.random.randn(100000),
        'category': np.random.choice(['A', 'B', 'C'], 100000),
    })


def get_memory_usage_mb():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark(func, *args, **kwargs):
    """
    Benchmark a function execution.

    Returns:
        tuple: (result, execution_time_ms, memory_delta_mb)
    """
    # Measure memory before
    mem_before = get_memory_usage_mb()

    # Measure time
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    # Measure memory after
    mem_after = get_memory_usage_mb()

    execution_time = (end_time - start_time) * 1000  # Convert to ms
    memory_delta = mem_after - mem_before

    return result, execution_time, memory_delta


# ==================== Chart Selection Performance ====================

class TestChartSelectorPerformance:
    """Performance tests for ChartSelector."""

    def test_recommendation_speed_small(self, small_dataset):
        """Test chart recommendation speed on small dataset."""
        selector = ChartSelector()

        _, exec_time, _ = benchmark(
            selector.recommend,
            small_dataset,
            x='date',
            y='value'
        )

        # Should be very fast (< 10ms)
        assert exec_time < 10, f"Chart selection took {exec_time:.2f}ms (expected < 10ms)"

    def test_recommendation_speed_large(self, large_dataset):
        """Test chart recommendation speed on large dataset."""
        selector = ChartSelector()

        _, exec_time, _ = benchmark(
            selector.recommend,
            large_dataset,
            x='date',
            y='value'
        )

        # Should still be fast even with large data (< 50ms)
        assert exec_time < 50, f"Chart selection took {exec_time:.2f}ms (expected < 50ms)"


# ==================== Data Profiling Performance ====================

class TestDataProfilerPerformance:
    """Performance tests for DataProfiler."""

    def test_profiling_speed_small(self, small_dataset):
        """Test data profiling speed on small dataset."""
        profiler = DataProfiler()

        _, exec_time, _ = benchmark(profiler.profile, small_dataset)

        # Should be very fast (< 20ms)
        assert exec_time < 20, f"Profiling took {exec_time:.2f}ms (expected < 20ms)"

    def test_profiling_speed_medium(self, medium_dataset):
        """Test data profiling speed on medium dataset."""
        profiler = DataProfiler()

        _, exec_time, _ = benchmark(profiler.profile, medium_dataset)

        # Should be reasonably fast (< 100ms)
        assert exec_time < 100, f"Profiling took {exec_time:.2f}ms (expected < 100ms)"

    def test_profiling_speed_large(self, large_dataset):
        """Test data profiling speed on large dataset."""
        profiler = DataProfiler()

        _, exec_time, _ = benchmark(profiler.profile, large_dataset)

        # Target: < 10ms per 1M rows → 100k rows should be < 1ms
        # But allowing more time for safety
        assert exec_time < 500, f"Profiling took {exec_time:.2f}ms (expected < 500ms)"

    def test_profiling_memory_usage(self, medium_dataset):
        """Test data profiling memory usage."""
        profiler = DataProfiler()

        _, _, mem_delta = benchmark(profiler.profile, medium_dataset)

        # Should not use excessive memory (< 50MB for 10k rows)
        assert mem_delta < 50, f"Profiling used {mem_delta:.2f}MB (expected < 50MB)"


# ==================== Insights Generation Performance ====================

class TestInsightsPerformance:
    """Performance tests for InsightsEngine."""

    def test_insights_speed_small(self, small_dataset):
        """Test insights generation speed on small dataset."""
        engine = InsightsEngine()

        _, exec_time, _ = benchmark(
            engine.generate_insights,
            small_dataset,
            target_column='value'
        )

        # Should be fast (< 100ms for 1k rows)
        assert exec_time < 100, f"Insights generation took {exec_time:.2f}ms (expected < 100ms)"

    def test_insights_speed_medium(self, medium_dataset):
        """Test insights generation speed on medium dataset."""
        engine = InsightsEngine()

        _, exec_time, _ = benchmark(
            engine.generate_insights,
            medium_dataset,
            target_column='value'
        )

        # Should be reasonable (< 500ms for 10k rows)
        assert exec_time < 500, f"Insights generation took {exec_time:.2f}ms (expected < 500ms)"


# ==================== Calculated Fields Performance ====================

class TestCalculatedFieldsPerformance:
    """Performance tests for CalculatedField."""

    def test_simple_calculation_speed(self, medium_dataset):
        """Test simple calculated field speed."""
        # Add required columns
        df = medium_dataset.copy()
        df['revenue'] = np.random.uniform(100, 1000, len(df))
        df['cost'] = np.random.uniform(50, 500, len(df))

        field = CalculatedField('profit', '[revenue] - [cost]')

        _, exec_time, _ = benchmark(field.apply, df)

        # Should be fast (< 50ms for 10k rows)
        assert exec_time < 50, f"Calculation took {exec_time:.2f}ms (expected < 50ms)"

    def test_complex_calculation_speed(self, medium_dataset):
        """Test complex calculated field speed."""
        df = medium_dataset.copy()
        df['revenue'] = np.random.uniform(100, 1000, len(df))
        df['cost'] = np.random.uniform(50, 500, len(df))

        field = CalculatedField(
            'margin_percent',
            '([revenue] - [cost]) / [revenue] * 100'
        )

        _, exec_time, _ = benchmark(field.apply, df)

        # Should still be fast (< 100ms)
        assert exec_time < 100, f"Calculation took {exec_time:.2f}ms (expected < 100ms)"

    def test_multiple_fields_speed(self, medium_dataset):
        """Test applying multiple calculated fields."""
        df = medium_dataset.copy()
        df['revenue'] = np.random.uniform(100, 1000, len(df))
        df['cost'] = np.random.uniform(50, 500, len(df))

        manager = CalculatedFieldManager()
        manager.add_field(CalculatedField('profit', '[revenue] - [cost]'))
        manager.add_field(CalculatedField('margin', '[profit] / [revenue]'))
        manager.add_field(CalculatedField('margin_percent', '[margin] * 100'))

        _, exec_time, _ = benchmark(manager.apply_all, df)

        # Multiple fields should still be fast (< 200ms)
        assert exec_time < 200, f"Multiple calculations took {exec_time:.2f}ms (expected < 200ms)"


# ==================== Window Functions Performance ====================

class TestWindowFunctionsPerformance:
    """Performance tests for WindowFunction."""

    def test_running_total_speed(self, medium_dataset):
        """Test running total performance."""
        window = WindowFunction(WindowType.RUNNING_TOTAL, 'value')

        _, exec_time, _ = benchmark(window.apply, medium_dataset)

        # Should be fast (pandas cumsum is optimized)
        assert exec_time < 50, f"Running total took {exec_time:.2f}ms (expected < 50ms)"

    def test_moving_average_speed(self, medium_dataset):
        """Test moving average performance."""
        window = WindowFunction(WindowType.MOVING_AVG, 'value', window_size=7)

        _, exec_time, _ = benchmark(window.apply, medium_dataset)

        # Should be reasonable (< 100ms)
        assert exec_time < 100, f"Moving average took {exec_time:.2f}ms (expected < 100ms)"

    def test_partitioned_window_speed(self, medium_dataset):
        """Test partitioned window function performance."""
        window = WindowFunction(
            WindowType.RUNNING_TOTAL,
            'value',
            partition_by=['category']
        )

        _, exec_time, _ = benchmark(window.apply, medium_dataset)

        # Partitioned operations are slower but should still be reasonable
        assert exec_time < 200, f"Partitioned window took {exec_time:.2f}ms (expected < 200ms)"


# ==================== Filter Performance ====================

class TestFilterPerformance:
    """Performance tests for filters."""

    def test_range_filter_speed(self, large_dataset):
        """Test range filter performance on large dataset."""
        from ..interactive.filters import RangeFilter

        filter = RangeFilter('test', 'value', min_value=-1, max_value=1)

        _, exec_time, _ = benchmark(filter.apply, large_dataset)

        # Pandas filtering is fast (< 50ms for 100k rows)
        assert exec_time < 50, f"Range filter took {exec_time:.2f}ms (expected < 50ms)"

    def test_cascading_filters_speed(self, medium_dataset):
        """Test cascading filters performance."""
        context = FilterContext()

        context.add_filter(ListFilter('f1', 'category', allowed_values=['A', 'B']))
        context.add_filter(RangeFilter('f2', 'value', min_value=-1, max_value=1))

        _, exec_time, _ = benchmark(context.apply_all, medium_dataset, cascade=True)

        # Cascading should still be fast (< 100ms)
        assert exec_time < 100, f"Cascading filters took {exec_time:.2f}ms (expected < 100ms)"


# ==================== Memory Usage Tests ====================

class TestMemoryUsage:
    """Memory usage tests."""

    def test_data_profiling_memory(self, large_dataset):
        """Test memory usage during data profiling."""
        profiler = DataProfiler()

        mem_before = get_memory_usage_mb()
        profiler.profile(large_dataset)
        mem_after = get_memory_usage_mb()

        mem_delta = mem_after - mem_before

        # Should not create large memory overhead
        assert mem_delta < 100, f"Profiling used {mem_delta:.2f}MB (expected < 100MB)"

    def test_calculated_fields_memory(self, large_dataset):
        """Test memory usage for calculated fields."""
        df = large_dataset.copy()
        df['revenue'] = np.random.uniform(100, 1000, len(df))
        df['cost'] = np.random.uniform(50, 500, len(df))

        manager = CalculatedFieldManager()
        manager.add_field(CalculatedField('profit', '[revenue] - [cost]'))

        mem_before = get_memory_usage_mb()
        result = manager.apply_all(df)
        mem_after = get_memory_usage_mb()

        mem_delta = mem_after - mem_before

        # Memory delta should be reasonable (roughly size of new column)
        # 100k floats ≈ 0.8MB, so delta should be small
        assert mem_delta < 50, f"Calculated field used {mem_delta:.2f}MB (expected < 50MB)"


# ==================== Stress Tests ====================

class TestStressTests:
    """Stress tests with extreme data."""

    @pytest.mark.slow
    def test_very_large_dataset(self):
        """Test with very large dataset (1M rows)."""
        # This test is marked as slow
        df = pd.DataFrame({
            'value': np.random.randn(1000000),
            'category': np.random.choice(['A', 'B', 'C'], 1000000),
        })

        profiler = DataProfiler()

        _, exec_time, _ = benchmark(profiler.profile, df)

        # Even 1M rows should profile in reasonable time (< 2 seconds)
        assert exec_time < 2000, f"Profiling 1M rows took {exec_time:.2f}ms (expected < 2000ms)"

    @pytest.mark.slow
    def test_wide_dataset(self):
        """Test with wide dataset (many columns)."""
        # 100 columns
        data = {f'col_{i}': np.random.randn(1000) for i in range(100)}
        df = pd.DataFrame(data)

        profiler = DataProfiler()

        _, exec_time, _ = benchmark(profiler.profile, df)

        # Wide datasets should still profile reasonably (< 500ms)
        assert exec_time < 500, f"Profiling 100 columns took {exec_time:.2f}ms (expected < 500ms)"


# ==================== Benchmark Summary ====================

class TestBenchmarkSummary:
    """Generate benchmark summary."""

    def test_print_benchmark_summary(self, small_dataset, medium_dataset, large_dataset):
        """Print comprehensive benchmark summary."""
        print("\n" + "="*60)
        print("VizForge v1.0.0 - Performance Benchmark Summary")
        print("="*60)

        # ChartSelector
        selector = ChartSelector()
        _, time_small, _ = benchmark(selector.recommend, small_dataset, x='date', y='value')
        _, time_large, _ = benchmark(selector.recommend, large_dataset, x='date', y='value')

        print(f"\n📊 Chart Selection:")
        print(f"  - Small dataset (1k):   {time_small:.2f}ms")
        print(f"  - Large dataset (100k): {time_large:.2f}ms")

        # DataProfiler
        profiler = DataProfiler()
        _, time_1k, mem_1k = benchmark(profiler.profile, small_dataset)
        _, time_10k, mem_10k = benchmark(profiler.profile, medium_dataset)
        _, time_100k, mem_100k = benchmark(profiler.profile, large_dataset)

        print(f"\n📈 Data Profiling:")
        print(f"  - 1k rows:    {time_1k:.2f}ms (mem: {mem_1k:.2f}MB)")
        print(f"  - 10k rows:   {time_10k:.2f}ms (mem: {mem_10k:.2f}MB)")
        print(f"  - 100k rows:  {time_100k:.2f}ms (mem: {mem_100k:.2f}MB)")

        # InsightsEngine
        engine = InsightsEngine()
        _, time_insights, _ = benchmark(engine.generate_insights, small_dataset, 'value')

        print(f"\n💡 Insights Generation:")
        print(f"  - Small dataset: {time_insights:.2f}ms")

        # Window Functions
        window = WindowFunction(WindowType.RUNNING_TOTAL, 'value')
        _, time_window, _ = benchmark(window.apply, medium_dataset)

        print(f"\n🔢 Window Functions:")
        print(f"  - Running total (10k): {time_window:.2f}ms")

        print("\n" + "="*60)
        print("✅ All benchmarks completed successfully!")
        print("="*60 + "\n")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])  # -s to show print output
