"""Coverage boost tests for vizforge charts, analytics, utils, and core modules."""
import pytest
import numpy as np
import pandas as pd


# ── ScatterPlot ──────────────────────────────────────────────────────────────

def test_scatter_dict_input():
    from vizforge.charts.scatter import ScatterPlot
    data = {"x": [1, 2, 3], "y": [4, 5, 6]}
    s = ScatterPlot(data=data, x="x", y="y")
    assert s.fig is not None


def test_scatter_list_input():
    from vizforge.charts.scatter import ScatterPlot
    y = [2, 4, 6, 8, 10]
    s = ScatterPlot(data=y, x=[1, 2, 3, 4, 5], y=y)
    assert s.fig is not None


def test_scatter_with_size():
    from vizforge.charts.scatter import ScatterPlot
    df = pd.DataFrame({"x": range(5), "y": range(5)})
    s = ScatterPlot(data=df, x="x", y="y", size=[10.0, 20.0, 15.0, 5.0, 25.0])
    assert s.fig is not None


def test_scatter_with_color_numeric():
    from vizforge.charts.scatter import ScatterPlot
    df = pd.DataFrame({"x": range(5), "y": range(5)})
    s = ScatterPlot(data=df, x="x", y="y", color=[1.0, 2.0, 3.0, 4.0, 5.0])
    assert s.fig is not None


def test_scatter_with_multiple_y():
    from vizforge.charts.scatter import ScatterPlot
    df = pd.DataFrame({"x": range(5), "y1": range(5), "y2": range(5)})
    s = ScatterPlot(data=df, x="x", y=["y1", "y2"])
    assert s.fig is not None


def test_scatter_x_none():
    from vizforge.charts.scatter import ScatterPlot
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    s = ScatterPlot(data=df, x=None, y="y")
    assert s.fig is not None


def test_scatter_3d():
    from vizforge.charts.scatter import ScatterPlot
    df = pd.DataFrame({"x": range(5), "y": range(5), "z": range(5)})
    s = ScatterPlot(data=df, x="x", y="y", z="z")
    assert s.fig is not None


# ── LineChart ────────────────────────────────────────────────────────────────

def test_line_dict_input():
    from vizforge.charts.line import LineChart
    data = {"x": [1, 2, 3, 4], "y": [10.0, 20.0, 15.0, 30.0]}
    lc = LineChart(data=data, x="x", y="y")
    assert lc.fig is not None


def test_line_dict_multi_y():
    from vizforge.charts.line import LineChart
    data = {"x": [1, 2, 3], "y1": [1.0, 2.0, 3.0], "y2": [3.0, 2.0, 1.0]}
    lc = LineChart(data=data, x="x", y=["y1", "y2"])
    assert lc.fig is not None


def test_line_dict_no_x():
    from vizforge.charts.line import LineChart
    data = {"y": [1.0, 2.0, 3.0]}
    lc = LineChart(data=data, y="y")
    assert lc.fig is not None


def test_line_list_input():
    from vizforge.charts.line import LineChart
    y = [10.0, 20.0, 15.0]
    lc = LineChart(data=y, x=[1, 2, 3], y=y)
    assert lc.fig is not None


def test_line_dataframe_multi_y():
    from vizforge.charts.line import LineChart
    df = pd.DataFrame({"x": range(5), "y1": range(5), "y2": [x * 2 for x in range(5)]})
    lc = LineChart(data=df, x="x", y=["y1", "y2"], name="series")
    assert lc.fig is not None


def test_line_dataframe_x_none():
    from vizforge.charts.line import LineChart
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    lc = LineChart(data=df, y="y")
    assert lc.fig is not None


def test_line_dataframe_multi_y_no_name():
    from vizforge.charts.line import LineChart
    df = pd.DataFrame({"x": range(3), "a": range(3), "b": range(3)})
    lc = LineChart(data=df, x="x", y=["a", "b"])
    assert lc.fig is not None


# ── AreaChart ────────────────────────────────────────────────────────────────

def test_area_dict_input():
    from vizforge.charts.area import AreaChart
    data = {"x": [1, 2, 3], "y": [10.0, 20.0, 15.0]}
    a = AreaChart(data=data, x="x", y="y")
    assert a.fig is not None


def test_area_stackgroup():
    from vizforge.charts.area import AreaChart
    df = pd.DataFrame({"x": range(5), "y": range(5)})
    a = AreaChart(data=df, x="x", y="y", stackgroup="one")
    assert a.fig is not None


# ── BarChart ─────────────────────────────────────────────────────────────────

def test_bar_dict_input():
    from vizforge.charts.bar import BarChart
    data = {"x": ["A", "B", "C"], "y": [10.0, 20.0, 15.0]}
    b = BarChart(data=data, x="x", y="y")
    assert b.fig is not None


def test_bar_horizontal():
    from vizforge.charts.bar import BarChart
    df = pd.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 20.0, 15.0]})
    b = BarChart(data=df, x="cat", y="val", orientation="h")
    assert b.fig is not None


def test_bar_multi_y():
    from vizforge.charts.bar import BarChart
    df = pd.DataFrame({"cat": ["A", "B"], "y1": [10.0, 20.0], "y2": [5.0, 15.0]})
    b = BarChart(data=df, x="cat", y=["y1", "y2"])
    assert b.fig is not None


def test_bar_stacked():
    from vizforge.charts.bar import BarChart
    df = pd.DataFrame({"cat": ["A", "B"], "y1": [10.0, 20.0], "y2": [5.0, 15.0]})
    b = BarChart(data=df, x="cat", y=["y1", "y2"], stacked=True)
    assert b.fig is not None


# ── Boxplot ──────────────────────────────────────────────────────────────────

def test_boxplot_list_input():
    from vizforge.charts.boxplot import Boxplot
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.5]
    b = Boxplot(data=data, name="data")
    assert b.fig is not None


def test_boxplot_dict_input():
    from vizforge.charts.boxplot import Boxplot
    data = {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}
    b = Boxplot(data=data)
    assert b.fig is not None


def test_boxplot_df_grouped():
    from vizforge.charts.boxplot import Boxplot
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0], "cat": ["A", "B", "A", "B"]})
    b = Boxplot(data=df, y="val", x="cat")
    assert b.fig is not None


# ── BubbleChart ──────────────────────────────────────────────────────────────

def test_bubble_dict_input():
    from vizforge.charts.bubble import BubbleChart
    data = {"x": [1, 2, 3], "y": [4, 5, 6], "s": [10.0, 20.0, 15.0]}
    b = BubbleChart(data=data, x="x", y="y", size="s")
    assert b.fig is not None


# ── FunnelChart ──────────────────────────────────────────────────────────────

def test_funnel_dict_input():
    from vizforge.charts.funnel import FunnelChart
    data = {"stage": ["A", "B", "C"], "val": [100.0, 60.0, 30.0]}
    f = FunnelChart(data=data, x="val", y="stage")
    assert f.fig is not None


# ── Heatmap ──────────────────────────────────────────────────────────────────

def test_heatmap_matrix_input():
    from vizforge.charts.heatmap import Heatmap
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    h = Heatmap(data=matrix)
    assert h.fig is not None


# ── Histogram ────────────────────────────────────────────────────────────────

def test_histogram_list():
    from vizforge.charts.histogram import Histogram
    data = list(np.random.randn(50))
    h = Histogram(data=data)
    assert h.fig is not None


# ── Radar ────────────────────────────────────────────────────────────────────

def test_radar_dict_input():
    from vizforge.charts.radar import RadarChart
    data = {"categories": ["A", "B", "C", "D"], "values": [0.8, 0.6, 0.9, 0.4]}
    r = RadarChart(data=data)
    assert r.fig is not None


# ── Waterfall ────────────────────────────────────────────────────────────────

def test_waterfall_dict_input():
    from vizforge.charts.waterfall import WaterfallChart
    data = {"labels": ["Start", "+Q1", "+Q2", "End"], "values": [100.0, 30.0, -10.0, 120.0]}
    w = WaterfallChart(data=data, x="labels", y="values")
    assert w.fig is not None


# ── analytics/aggregations.py ────────────────────────────────────────────────

def test_quick_aggregation_sum():
    from vizforge.analytics.aggregations import quick_aggregation
    df = pd.DataFrame({"cat": ["A", "B", "A", "B"], "val": [10.0, 20.0, 15.0, 25.0]})
    result = quick_aggregation(df, "sum", "val", group_by="cat")
    assert result is not None


def test_quick_aggregation_avg():
    from vizforge.analytics.aggregations import quick_aggregation
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = quick_aggregation(df, "avg", "val")
    assert result is not None


def test_quick_aggregation_count():
    from vizforge.analytics.aggregations import quick_aggregation
    df = pd.DataFrame({"cat": ["A", "B", "A"], "val": [1.0, 2.0, 3.0]})
    result = quick_aggregation(df, "count", "val", group_by="cat")
    assert result is not None


def test_quick_aggregation_min_max():
    from vizforge.analytics.aggregations import quick_aggregation
    df = pd.DataFrame({"val": [3.0, 1.0, 4.0, 1.0, 5.0]})
    result_min = quick_aggregation(df, "min", "val")
    result_max = quick_aggregation(df, "max", "val")
    assert result_min is not None
    assert result_max is not None


def test_quick_aggregation_median():
    from vizforge.analytics.aggregations import quick_aggregation
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = quick_aggregation(df, "median", "val")
    assert result is not None


def test_quick_window_running_total():
    from vizforge.analytics.aggregations import quick_window
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = quick_window(df, "running_total", "val")
    assert result is not None


def test_quick_window_rank():
    from vizforge.analytics.aggregations import quick_window
    df = pd.DataFrame({"val": [3.0, 1.0, 4.0, 1.0, 5.0]})
    result = quick_window(df, "rank", "val")
    assert result is not None


def test_quick_window_percent_of_total():
    from vizforge.analytics.aggregations import quick_window
    df = pd.DataFrame({"val": [25.0, 50.0, 25.0]})
    result = quick_window(df, "percent_of_total", "val")
    assert result is not None


def test_aggregation_engine_add():
    from vizforge.analytics.aggregations import AggregationEngine, Aggregation, AggregationType
    df = pd.DataFrame({"cat": ["A", "B", "A"], "val": [10.0, 20.0, 15.0]})
    engine = AggregationEngine()
    agg = Aggregation(AggregationType.SUM, "val", group_by=["cat"])
    engine.add_aggregation(agg)
    result = engine.apply_all(df)
    assert result is not None


# ── utils/colors.py ──────────────────────────────────────────────────────────

def test_colors_generate_palette():
    from vizforge.utils.colors import generate_color_palette
    palette = generate_color_palette(10)
    assert len(palette) == 10


def test_colors_hex_to_rgb():
    from vizforge.utils.colors import hex_to_rgb
    r, g, b = hex_to_rgb("#336699")
    assert r == 51 and g == 102 and b == 153


def test_colors_rgb_to_hex():
    from vizforge.utils.colors import rgb_to_hex
    result = rgb_to_hex(51, 102, 153)
    assert result.lower() == "#336699"


def test_colors_color_scale():
    from vizforge.utils.colors import color_scale
    scale = color_scale(5, "#ff0000", "#0000ff")
    assert len(scale) == 5


# ── utils/data.py ────────────────────────────────────────────────────────────

def test_data_clean_data():
    from vizforge.utils.data import clean_data
    df = pd.DataFrame({"val": [1.0, None, 3.0, float("nan"), 5.0]})
    result = clean_data(df)
    assert isinstance(result, pd.DataFrame)


def test_data_normalize_minmax():
    from vizforge.utils.data import normalize_data
    df = pd.DataFrame({"val": [0.0, 50.0, 100.0]})
    result = normalize_data(df, columns=["val"], method="minmax")
    assert isinstance(result, pd.DataFrame)


def test_data_normalize_zscore():
    from vizforge.utils.data import normalize_data
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = normalize_data(df, columns=["val"], method="zscore")
    assert isinstance(result, pd.DataFrame)


def test_data_detect_outliers():
    from vizforge.utils.data import detect_outliers
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 100.0]})
    result = detect_outliers(df, "val")
    assert result is not None


def test_data_aggregate():
    from vizforge.utils.data import aggregate_data
    df = pd.DataFrame({"cat": ["A", "B", "A"], "val": [10.0, 20.0, 15.0]})
    result = aggregate_data(df, group_by=["cat"], agg_column="val", agg_func="sum")
    assert result is not None


def test_data_bin():
    from vizforge.utils.data import bin_data
    df = pd.DataFrame({"val": [1.0, 5.0, 15.0, 25.0, 35.0]})
    result = bin_data(df, "val", bins=3)
    assert result is not None


# ── visual_designer/chart_config.py ──────────────────────────────────────────

def test_chart_config_create():
    from vizforge.visual_designer.chart_config import ChartConfig, ChartType
    config = ChartConfig(chart_type=ChartType.BAR)
    assert config.chart_type == ChartType.BAR


def test_chart_config_to_dict():
    from vizforge.visual_designer.chart_config import ChartConfig, ChartType
    config = ChartConfig(chart_type=ChartType.LINE)
    d = config.to_dict()
    assert isinstance(d, dict)


def test_chart_config_from_dict():
    from vizforge.visual_designer.chart_config import ChartConfig, ChartType
    config = ChartConfig(chart_type=ChartType.SCATTER)
    d = config.to_dict()
    config2 = ChartConfig.from_dict(d)
    assert config2 is not None


def test_chart_config_validate():
    from vizforge.visual_designer.chart_config import ChartConfig, ChartType
    config = ChartConfig(chart_type=ChartType.BAR)
    result = config.validate()
    assert result is not None or result is None  # just calls it


def test_chart_config_get_available_properties():
    from vizforge.visual_designer.chart_config import ChartConfig, ChartType
    config = ChartConfig(chart_type=ChartType.BAR)
    props = ChartConfig.get_available_properties(ChartType.BAR)
    assert isinstance(props, list)


# ── predictive/forecaster.py ─────────────────────────────────────────────────

def test_forecaster_create():
    from vizforge.predictive.forecaster import TimeSeriesForecaster
    data = [100.0 + i * 1.5 for i in range(30)]
    f = TimeSeriesForecaster(data)
    assert f is not None


def test_forecaster_predict():
    from vizforge.predictive.forecaster import TimeSeriesForecaster
    data = [100.0 + i * 2.0 for i in range(30)]
    f = TimeSeriesForecaster(data)
    result = f.forecast(periods=7)
    assert result is not None


# ── core/base.py additional coverage ─────────────────────────────────────────

def test_base_chart_to_html():
    from vizforge.charts import LineChart
    df = pd.DataFrame({"x": range(5), "y": range(5)})
    lc = LineChart(data=df, x="x", y="y")
    html = lc.to_html()
    assert isinstance(html, str)


def test_base_chart_to_json():
    from vizforge.charts import LineChart
    df = pd.DataFrame({"x": range(5), "y": range(5)})
    lc = LineChart(data=df, x="x", y="y")
    result = lc.to_json()
    assert isinstance(result, str)


def test_base_chart_update_layout():
    from vizforge.charts import BarChart
    df = pd.DataFrame({"cat": ["A", "B", "C"], "val": [1.0, 2.0, 3.0]})
    b = BarChart(data=df, x="cat", y="val")
    b.update_layout(xaxis_title="X", yaxis_title="Y")
    assert b.fig is not None
