"""
VizForge Visual Bias Detector v2.0

Detects misleading visual encodings and chart manipulations.
"""
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class BiasReport:
    """Result of bias detection analysis."""
    detected_biases: List[Dict]
    severity: str  # "none", "low", "medium", "high"
    overall_score: float  # 0 (no bias) to 1 (severe bias)
    recommendations: List[str]


class VisualBiasDetector:
    """
    Detect misleading visual encodings in charts.

    Detects:
    - Truncated/manipulated axes
    - Cherry-picked data ranges
    - Misleading aspect ratios
    - Inappropriate dual axes
    - Distorted 3D effects
    - Overplotting
    - Color scheme issues
    """

    def __init__(self):
        self.bias_thresholds = {
            "axis_truncation": 0.1,  # 10% of range
            "aspect_ratio": 2.0,  # Width/height ratio
            "dual_axis_ratio": 10.0,  # Scale difference
            "overplotting_density": 0.05  # 5% overlap
        }

    def detect_axis_truncation(
        self,
        data: np.ndarray,
        axis_min: float,
        axis_max: float
    ) -> Dict:
        """
        Detect if axis is truncated (doesn't start at zero for bar charts).

        This is a common manipulation that exaggerates differences.
        """
        data_min = np.min(data)
        data_max = np.max(data)
        data_range = data_max - data_min

        bias_detected = False
        severity = "none"
        explanation = ""

        # Check if axis should start at zero
        if data_min >= 0:  # All positive data
            if axis_min > 0 and axis_min > data_range * self.bias_thresholds["axis_truncation"]:
                bias_detected = True
                truncation_pct = (axis_min / data_range) * 100
                severity = "high" if truncation_pct > 50 else "medium"
                explanation = f"Y-axis doesn't start at zero (truncated by {truncation_pct:.1f}%), exaggerating differences"

        return {
            "type": "axis_truncation",
            "detected": bias_detected,
            "severity": severity,
            "explanation": explanation,
            "recommendation": "Start y-axis at zero for bar charts" if bias_detected else None
        }

    def detect_cherry_picking(
        self,
        full_data: np.ndarray,
        displayed_data: np.ndarray
    ) -> Dict:
        """
        Detect if data range was cherry-picked to show favorable trend.
        """
        bias_detected = False
        severity = "none"
        explanation = ""

        if len(displayed_data) < len(full_data):
            display_ratio = len(displayed_data) / len(full_data)

            if display_ratio < 0.5:  # Showing less than 50% of data
                # Check if trend changes significantly with full data
                display_trend = np.polyfit(range(len(displayed_data)), displayed_data, 1)[0]
                full_trend = np.polyfit(range(len(full_data)), full_data, 1)[0]

                if abs(display_trend - full_trend) / abs(full_trend + 1e-9) > 0.3:  # 30% trend difference
                    bias_detected = True
                    severity = "high"
                    explanation = f"Only {display_ratio:.0%} of data shown, trend differs from full dataset"

        return {
            "type": "cherry_picking",
            "detected": bias_detected,
            "severity": severity,
            "explanation": explanation,
            "recommendation": "Show full data range or clearly indicate partial view" if bias_detected else None
        }

    def detect_aspect_ratio_distortion(
        self,
        width: float,
        height: float,
        data_x_range: float,
        data_y_range: float
    ) -> Dict:
        """
        Detect if aspect ratio distorts perception of trend.
        """
        bias_detected = False
        severity = "none"
        explanation = ""

        physical_ratio = width / height
        data_ratio = data_x_range / (data_y_range + 1e-9)

        # Banking to 45 degrees principle (Cleveland & McGill)
        # Ideal slope perception occurs around 1:1 aspect ratio
        distortion = abs(physical_ratio - data_ratio) / data_ratio

        if distortion > 2.0:  # 200% distortion
            bias_detected = True
            severity = "medium" if distortion < 5.0 else "high"
            explanation = f"Aspect ratio ({physical_ratio:.1f}:1) distorts trend perception"

        return {
            "type": "aspect_ratio_distortion",
            "detected": bias_detected,
            "severity": severity,
            "explanation": explanation,
            "recommendation": "Adjust aspect ratio to match data range for accurate slope perception" if bias_detected else None
        }

    def detect_dual_axis_abuse(
        self,
        left_axis_range: float,
        right_axis_range: float
    ) -> Dict:
        """
        Detect misleading dual-axis charts where scales are manipulated.
        """
        bias_detected = False
        severity = "none"
        explanation = ""

        scale_ratio = max(left_axis_range, right_axis_range) / (min(left_axis_range, right_axis_range) + 1e-9)

        if scale_ratio > self.bias_thresholds["dual_axis_ratio"]:
            bias_detected = True
            severity = "high"
            explanation = f"Dual axes have vastly different scales ({scale_ratio:.1f}x), can mislead correlation perception"

        return {
            "type": "dual_axis_abuse",
            "detected": bias_detected,
            "severity": severity,
            "explanation": explanation,
            "recommendation": "Normalize scales or use separate charts" if bias_detected else None
        }

    def detect_overplotting(
        self,
        n_points: int,
        plot_area: float  # in square pixels
    ) -> Dict:
        """
        Detect overplotting that obscures data distribution.
        """
        bias_detected = False
        severity = "none"
        explanation = ""

        # Estimate point density (assuming 5px point radius)
        point_area = np.pi * 5**2
        total_point_area = n_points * point_area
        density = total_point_area / plot_area

        if density > self.bias_thresholds["overplotting_density"]:
            bias_detected = True
            severity = "medium" if density < 0.2 else "high"
            explanation = f"High point density ({density:.1%} coverage) obscures distribution"

        return {
            "type": "overplotting",
            "detected": bias_detected,
            "severity": severity,
            "explanation": explanation,
            "recommendation": "Use transparency, hexbin, or density plot" if bias_detected else None
        }

    def detect_color_bias(
        self,
        color_scheme: str,
        data_type: str  # "sequential", "diverging", "categorical"
    ) -> Dict:
        """
        Detect inappropriate color schemes for data type.
        """
        bias_detected = False
        severity = "none"
        explanation = ""

        # Sequential data should use sequential colors
        if data_type == "sequential" and color_scheme in ["diverging", "categorical"]:
            bias_detected = True
            severity = "medium"
            explanation = f"Sequential data using {color_scheme} color scheme may confuse ordering"

        # Diverging data should use diverging colors
        if data_type == "diverging" and color_scheme != "diverging":
            bias_detected = True
            severity = "medium"
            explanation = f"Diverging data should use diverging color scheme to show positive/negative split"

        return {
            "type": "color_bias",
            "detected": bias_detected,
            "severity": severity,
            "explanation": explanation,
            "recommendation": f"Use {data_type} color scheme" if bias_detected else None
        }

    def detect_3d_distortion(
        self,
        chart_type: str,
        has_3d: bool
    ) -> Dict:
        """
        Detect inappropriate use of 3D effects that distort perception.
        """
        bias_detected = False
        severity = "none"
        explanation = ""

        if has_3d and chart_type in ["pie", "bar"]:
            bias_detected = True
            severity = "high"
            explanation = "3D effects distort area/length perception and serve no analytical purpose"

        return {
            "type": "3d_distortion",
            "detected": bias_detected,
            "severity": severity,
            "explanation": explanation,
            "recommendation": "Remove 3D effects and use 2D chart" if bias_detected else None
        }

    def analyze_chart(
        self,
        chart_metadata: Dict
    ) -> BiasReport:
        """
        Comprehensive bias analysis of a chart.

        Args:
            chart_metadata: Dict with chart properties:
                - data: np.array of values
                - axis_min, axis_max: axis ranges
                - chart_type: str
                - width, height: dimensions
                - etc.

        Returns:
            BiasReport with detected biases and recommendations
        """
        detected_biases = []

        # Run all bias detectors
        detectors = [
            self.detect_axis_truncation(
                chart_metadata.get("data", np.array([])),
                chart_metadata.get("axis_min", 0),
                chart_metadata.get("axis_max", 100)
            ),
            self.detect_aspect_ratio_distortion(
                chart_metadata.get("width", 800),
                chart_metadata.get("height", 600),
                chart_metadata.get("x_range", 10),
                chart_metadata.get("y_range", 10)
            ),
            self.detect_3d_distortion(
                chart_metadata.get("chart_type", "bar"),
                chart_metadata.get("has_3d", False)
            )
        ]

        # Collect detected biases
        for result in detectors:
            if result["detected"]:
                detected_biases.append(result)

        # Calculate overall severity
        severity_scores = {"none": 0, "low": 0.25, "medium": 0.5, "high": 1.0}
        if detected_biases:
            max_severity = max(b["severity"] for b in detected_biases)
            overall_score = severity_scores[max_severity]
        else:
            max_severity = "none"
            overall_score = 0.0

        # Collect recommendations
        recommendations = [
            b["recommendation"]
            for b in detected_biases
            if b.get("recommendation")
        ]

        return BiasReport(
            detected_biases=detected_biases,
            severity=max_severity,
            overall_score=overall_score,
            recommendations=recommendations
        )

    def generate_report(self, bias_report: BiasReport) -> str:
        """Generate human-readable bias report."""
        if bias_report.severity == "none":
            return "✅ No visual biases detected. Chart appears honest and accurate."

        report = f"""
⚠️  VISUAL BIAS DETECTED (Severity: {bias_report.severity.upper()})
Overall Bias Score: {bias_report.overall_score:.0%}

DETECTED ISSUES:
{chr(10).join(f"  • {b['type'].replace('_', ' ').title()}: {b['explanation']}" for b in bias_report.detected_biases)}

RECOMMENDATIONS:
{chr(10).join(f"  → {rec}" for rec in bias_report.recommendations)}
        """
        return report.strip()
