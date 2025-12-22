"""
VizForge Enhanced Auto-Insights v2

Automatic intelligent insights generation using Pattern Detector.
NO API required! Template-based natural language generation.

Features:
- Automatic report generation
- Natural language explanations
- Actionable recommendations
- Executive summaries
- Trend predictions
- Anomaly explanations

Part of VizForge v1.2.0 - ULTRA Intelligence Features
"""

from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .pattern_detector import PatternDetector, Pattern, PatternType


@dataclass
class InsightReport:
    """Comprehensive insight report."""
    summary: str  # Executive summary
    key_findings: List[str]  # Top insights
    recommendations: List[str]  # Action items
    detailed_insights: Dict[str, List[str]]  # Grouped by category
    statistics: Dict[str, Any]  # Key metrics
    generated_at: datetime = None

    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now()

    def to_markdown(self) -> str:
        """Export report as Markdown."""
        md = f"# Data Insights Report\n\n"
        md += f"*Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

        md += f"## Executive Summary\n\n{self.summary}\n\n"

        md += f"## Key Findings\n\n"
        for i, finding in enumerate(self.key_findings, 1):
            md += f"{i}. {finding}\n"

        md += f"\n## Recommendations\n\n"
        for i, rec in enumerate(self.recommendations, 1):
            md += f"{i}. {rec}\n"

        md += f"\n## Detailed Insights\n\n"
        for category, insights in self.detailed_insights.items():
            md += f"### {category}\n\n"
            for insight in insights:
                md += f"- {insight}\n"
            md += "\n"

        md += f"## Key Statistics\n\n"
        for stat, value in self.statistics.items():
            if isinstance(value, float):
                md += f"- **{stat}**: {value:.2f}\n"
            else:
                md += f"- **{stat}**: {value}\n"

        return md

    def to_html(self) -> str:
        """Export report as HTML."""
        # Build statistics HTML properly
        stats_html = []
        for stat, value in self.statistics.items():
            if isinstance(value, (float, np.floating)):
                stats_html.append(f'<span class="stat"><strong>{stat}:</strong> {value:.2f}</span>')
            else:
                stats_html.append(f'<span class="stat"><strong>{stat}:</strong> {value}</span>')

        html = f"""
        <html>
        <head>
            <title>Data Insights Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                h3 {{ color: #7f8c8d; }}
                .summary {{ background: #ecf0f1; padding: 20px; border-left: 4px solid #3498db; margin: 20px 0; }}
                .finding {{ background: #e8f8f5; padding: 10px; margin: 10px 0; border-left: 3px solid #27ae60; }}
                .recommendation {{ background: #fef5e7; padding: 10px; margin: 10px 0; border-left: 3px solid #f39c12; }}
                .stat {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>Data Insights Report</h1>
            <p class="timestamp">Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Executive Summary</h2>
            <div class="summary">{self.summary}</div>

            <h2>Key Findings</h2>
            {"".join([f'<div class="finding">{i}. {f}</div>' for i, f in enumerate(self.key_findings, 1)])}

            <h2>Recommendations</h2>
            {"".join([f'<div class="recommendation">{i}. {r}</div>' for i, r in enumerate(self.recommendations, 1)])}

            <h2>Detailed Insights</h2>
            {"".join([f'<h3>{cat}</h3><ul>{"".join([f"<li>{ins}</li>" for ins in insights])}</ul>' for cat, insights in self.detailed_insights.items()])}

            <h2>Key Statistics</h2>
            <div>
                {"".join(stats_html)}
            </div>
        </body>
        </html>
        """
        return html


class EnhancedInsightsEngine:
    """
    Enhanced Auto-Insights Engine v2.

    Generates comprehensive insights using Pattern Detector.
    NO API required - template-based natural language generation!

    Features:
    - Automatic insight generation
    - Natural language explanations
    - Actionable recommendations
    - Executive summaries
    - Trend predictions

    Example:
        >>> engine = EnhancedInsightsEngine(df)
        >>> report = engine.generate_report()
        >>> print(report.summary)
        >>> print(report.to_markdown())
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        confidence_threshold: float = 0.7
    ):
        """
        Initialize insights engine.

        Args:
            data: Input data
            confidence_threshold: Minimum confidence for insights
        """
        self.data = self._prepare_data(data)
        self.confidence_threshold = confidence_threshold
        self.detector = PatternDetector(self.data, confidence_threshold)
        self.patterns: List[Pattern] = []

    def _prepare_data(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
        """Convert input to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, pd.Series):
            return data.to_frame()
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                return pd.DataFrame({'value': data})
            else:
                return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    # ==================== INSIGHT GENERATION ====================

    def generate_report(self, verbose: bool = False) -> InsightReport:
        """
        Generate comprehensive insight report.

        Args:
            verbose: Print progress

        Returns:
            InsightReport with all insights
        """
        if verbose:
            print("🔍 Analyzing data...")

        # Detect all patterns
        self.patterns = self.detector.detect_all_patterns(verbose=verbose)

        if verbose:
            print(f"✅ Found {len(self.patterns)} patterns")
            print("\n📝 Generating insights...")

        # Generate insights
        summary = self._generate_summary()
        key_findings = self._generate_key_findings()
        recommendations = self._generate_recommendations()
        detailed_insights = self._generate_detailed_insights()
        statistics = self._calculate_statistics()

        if verbose:
            print("✅ Report generated!")

        return InsightReport(
            summary=summary,
            key_findings=key_findings,
            recommendations=recommendations,
            detailed_insights=detailed_insights,
            statistics=statistics
        )

    def _generate_summary(self) -> str:
        """Generate executive summary."""
        n_patterns = len(self.patterns)
        n_rows = len(self.data)
        n_cols = len(self.data.columns)

        if n_patterns == 0:
            return f"Analysis of {n_rows:,} rows and {n_cols} variables shows no significant patterns at {self.confidence_threshold:.0%} confidence level."

        # Group patterns by type
        pattern_counts = {}
        for p in self.patterns:
            ptype = p.pattern_type.value
            pattern_counts[ptype] = pattern_counts.get(ptype, 0) + 1

        # Find most significant pattern
        top_pattern = self.patterns[0] if self.patterns else None

        summary = f"Analysis of {n_rows:,} observations across {n_cols} variables revealed {n_patterns} significant patterns. "

        if top_pattern:
            summary += f"Most notable: {top_pattern.description} "

        # Add pattern diversity
        if len(pattern_counts) > 1:
            summary += f"The data exhibits {len(pattern_counts)} distinct pattern types, "
            summary += f"indicating complex behavior worthy of deeper investigation."
        else:
            summary += f"The data shows consistent behavior with a single dominant pattern type."

        return summary

    def _generate_key_findings(self, max_findings: int = 5) -> List[str]:
        """Generate top key findings."""
        findings = []

        # Group patterns by type
        by_type = {}
        for p in self.patterns:
            ptype = p.pattern_type.value
            if ptype not in by_type:
                by_type[ptype] = []
            by_type[ptype].append(p)

        # Generate findings for each type
        for ptype, patterns in sorted(by_type.items(), key=lambda x: len(x[1]), reverse=True):
            if len(findings) >= max_findings:
                break

            top_pattern = patterns[0]  # Highest confidence

            if ptype == 'trend_increasing':
                findings.append(
                    f"Strong upward trend detected with {top_pattern.confidence:.1%} confidence. "
                    f"Data shows consistent growth over time (slope: {top_pattern.metadata.get('slope', 0):.4f})."
                )

            elif ptype == 'trend_decreasing':
                findings.append(
                    f"Declining trend identified with {top_pattern.confidence:.1%} confidence. "
                    f"Data exhibits downward movement (slope: {top_pattern.metadata.get('slope', 0):.4f})."
                )

            elif ptype == 'seasonal':
                period = top_pattern.metadata.get('period', 0)
                findings.append(
                    f"Seasonal pattern detected with period of {period} units ({top_pattern.confidence:.1%} confidence). "
                    f"This suggests cyclical behavior that repeats regularly."
                )

            elif ptype == 'correlation_strong':
                var1 = top_pattern.metadata.get('var1', 'Variable 1')
                var2 = top_pattern.metadata.get('var2', 'Variable 2')
                corr = top_pattern.metadata.get('correlation', 0)
                direction = "positive" if corr > 0 else "negative"

                findings.append(
                    f"Strong {direction} correlation ({abs(corr):.2f}) between {var1} and {var2}. "
                    f"Changes in one variable are likely to affect the other."
                )

            elif ptype == 'cluster':
                n_clusters = top_pattern.metadata.get('n_clusters', 0)
                findings.append(
                    f"Data naturally segments into {n_clusters} distinct clusters ({top_pattern.confidence:.1%} confidence). "
                    f"This suggests {n_clusters} different behavioral groups."
                )

            elif ptype == 'anomaly':
                findings.append(
                    f"Anomalies detected at {len(patterns)} locations. "
                    f"These outliers warrant individual investigation for data quality or exceptional events."
                )

            elif ptype == 'spike':
                if len(patterns) > 0:
                    findings.append(
                        f"{len(patterns)} significant spike(s) detected. "
                        f"These sudden increases may indicate special events or data collection issues."
                    )

            elif ptype == 'dip':
                if len(patterns) > 0:
                    findings.append(
                        f"{len(patterns)} significant dip(s) identified. "
                        f"These sudden decreases require attention to understand underlying causes."
                    )

            elif ptype == 'volatility_high':
                findings.append(
                    f"High volatility detected in recent data ({top_pattern.confidence:.1%} confidence). "
                    f"Values show increased variation compared to historical norms."
                )

            elif ptype == 'normal_distribution':
                findings.append(
                    f"Data follows normal distribution ({top_pattern.confidence:.1%} confidence). "
                    f"This enables use of parametric statistical methods and simplifies analysis."
                )

            elif ptype == 'skewed_distribution':
                skew = top_pattern.metadata.get('skewness', 0)
                direction = "right" if skew > 0 else "left"
                findings.append(
                    f"Distribution is {direction}-skewed (skewness: {skew:.2f}). "
                    f"Consider log transformation or non-parametric methods for analysis."
                )

        # Fill with generic insights if needed
        while len(findings) < min(max_findings, len(self.patterns)):
            remaining = [p for p in self.patterns if p.description not in [f.split('.')[0] for f in findings]]
            if remaining:
                findings.append(remaining[0].description)
            else:
                break

        return findings[:max_findings]

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Group patterns by type
        by_type = {}
        for p in self.patterns:
            ptype = p.pattern_type.value
            if ptype not in by_type:
                by_type[ptype] = []
            by_type[ptype].append(p)

        # Generate recommendations
        if 'trend_increasing' in by_type:
            recommendations.append(
                "Capitalize on upward trend: Consider increasing investment, expanding capacity, "
                "or accelerating growth initiatives to leverage positive momentum."
            )

        if 'trend_decreasing' in by_type:
            recommendations.append(
                "Address declining trend: Investigate root causes, implement corrective measures, "
                "and monitor closely to prevent further decline."
            )

        if 'seasonal' in by_type:
            patterns = by_type['seasonal']
            period = patterns[0].metadata.get('period', 0)
            recommendations.append(
                f"Plan for seasonality: Adjust resource allocation, inventory, or staffing "
                f"based on {period}-unit cycles. Consider seasonal forecasting models."
            )

        if 'correlation_strong' in by_type:
            recommendations.append(
                "Leverage correlations: Use strongly correlated variables for prediction, "
                "optimization, or causal analysis. Monitor for changes in relationships."
            )

        if 'cluster' in by_type:
            n_clusters = by_type['cluster'][0].metadata.get('n_clusters', 0)
            recommendations.append(
                f"Implement segmentation strategy: Tailor approaches for each of the {n_clusters} "
                f"distinct groups. Consider separate analysis or interventions per segment."
            )

        if 'anomaly' in by_type or 'spike' in by_type or 'dip' in by_type:
            recommendations.append(
                "Investigate anomalies: Review each outlier for data quality issues, "
                "exceptional events, or opportunities. Implement anomaly detection monitoring."
            )

        if 'volatility_high' in by_type:
            recommendations.append(
                "Manage volatility: Implement risk mitigation strategies, increase monitoring frequency, "
                "and consider smoothing or stabilization techniques."
            )

        if 'skewed_distribution' in by_type:
            recommendations.append(
                "Address skewness: Apply appropriate transformations (log, sqrt) for statistical analysis. "
                "Use non-parametric methods or consider data preprocessing."
            )

        # Generic recommendations
        if len(recommendations) == 0:
            recommendations.append(
                "Continue monitoring: While no critical patterns detected, maintain regular analysis "
                "to identify emerging trends or changes in behavior."
            )

        if len(self.patterns) > 5:
            recommendations.append(
                f"Prioritize investigations: With {len(self.patterns)} patterns detected, "
                f"focus on highest confidence insights first. Assign dedicated resources for analysis."
            )

        return recommendations

    def _generate_detailed_insights(self) -> Dict[str, List[str]]:
        """Generate detailed insights grouped by category."""
        detailed = {
            'Time Series Patterns': [],
            'Correlations & Relationships': [],
            'Data Quality & Anomalies': [],
            'Statistical Properties': [],
        }

        for pattern in self.patterns:
            ptype = pattern.pattern_type.value

            if ptype in ['trend_increasing', 'trend_decreasing', 'trend_stable', 'seasonal', 'cyclical']:
                detailed['Time Series Patterns'].append(pattern.description)

            elif ptype in ['correlation_strong', 'correlation_weak']:
                detailed['Correlations & Relationships'].append(pattern.description)

            elif ptype in ['anomaly', 'spike', 'dip', 'cluster']:
                detailed['Data Quality & Anomalies'].append(pattern.description)

            elif ptype in ['normal_distribution', 'skewed_distribution', 'volatility_high', 'volatility_low']:
                detailed['Statistical Properties'].append(pattern.description)

        # Remove empty categories
        detailed = {k: v for k, v in detailed.items() if v}

        return detailed

    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate key statistics."""
        stats = {
            'Total Observations': len(self.data),
            'Total Variables': len(self.data.columns),
            'Patterns Detected': len(self.patterns),
        }

        # Numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            # Calculate overall stats
            all_values = self.data[numeric_cols].values.flatten()
            all_values = all_values[~np.isnan(all_values)]

            if len(all_values) > 0:
                stats['Mean'] = float(np.mean(all_values))
                stats['Median'] = float(np.median(all_values))
                stats['Std Dev'] = float(np.std(all_values))
                stats['Min'] = float(np.min(all_values))
                stats['Max'] = float(np.max(all_values))

        # Pattern type distribution
        pattern_types = {}
        for p in self.patterns:
            ptype = p.pattern_type.value
            pattern_types[ptype] = pattern_types.get(ptype, 0) + 1

        stats['Pattern Types'] = len(pattern_types)

        # Highest confidence
        if self.patterns:
            stats['Highest Confidence'] = float(self.patterns[0].confidence)

        return stats

    # ==================== SPECIALIZED INSIGHTS ====================

    def explain_trend(self, column: str = None) -> str:
        """
        Explain trend in detail with predictions.

        Args:
            column: Column name

        Returns:
            Detailed trend explanation
        """
        if column is None:
            column = self.data.select_dtypes(include=[np.number]).columns[0]

        # Detect trend
        trend_patterns = [p for p in self.patterns if 'trend' in p.pattern_type.value]

        if not trend_patterns:
            return f"No significant trend detected in {column}."

        pattern = trend_patterns[0]
        slope = pattern.metadata.get('slope', 0)
        r_squared = pattern.metadata.get('r_squared', 0)

        explanation = f"Trend Analysis for {column}:\n\n"
        explanation += f"• Direction: {'Increasing' if slope > 0 else 'Decreasing' if slope < 0 else 'Stable'}\n"
        explanation += f"• Slope: {slope:.4f} units per observation\n"
        explanation += f"• Fit Quality: {r_squared:.1%} (R-squared)\n"
        explanation += f"• Confidence: {pattern.confidence:.1%}\n\n"

        # Simple prediction
        values = self.data[column].dropna().values
        current = values[-1]
        predicted_next = current + slope
        predicted_5 = current + (slope * 5)

        explanation += f"Predictions (assuming trend continues):\n"
        explanation += f"• Next value: {predicted_next:.2f}\n"
        explanation += f"• 5 steps ahead: {predicted_5:.2f}\n"

        return explanation

    def explain_correlation(self, var1: str = None, var2: str = None) -> str:
        """
        Explain correlation in detail.

        Args:
            var1: First variable
            var2: Second variable

        Returns:
            Detailed correlation explanation
        """
        corr_patterns = [p for p in self.patterns if 'correlation' in p.pattern_type.value]

        if not corr_patterns:
            return "No significant correlations detected."

        pattern = corr_patterns[0]
        var1 = pattern.metadata.get('var1', var1)
        var2 = pattern.metadata.get('var2', var2)
        corr = pattern.metadata.get('correlation', 0)

        explanation = f"Correlation Analysis: {var1} vs {var2}\n\n"
        explanation += f"• Correlation Coefficient: {corr:.3f}\n"
        explanation += f"• Direction: {'Positive' if corr > 0 else 'Negative'}\n"
        explanation += f"• Strength: "

        if abs(corr) > 0.9:
            explanation += "Very Strong\n"
        elif abs(corr) > 0.7:
            explanation += "Strong\n"
        elif abs(corr) > 0.5:
            explanation += "Moderate\n"
        else:
            explanation += "Weak\n"

        explanation += f"\nInterpretation:\n"
        if corr > 0:
            explanation += f"When {var1} increases, {var2} tends to increase as well.\n"
        else:
            explanation += f"When {var1} increases, {var2} tends to decrease.\n"

        explanation += f"\nCaution: Correlation does not imply causation. Further investigation needed."

        return explanation


# ==================== CONVENIENCE FUNCTIONS ====================

def generate_insights(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    confidence_threshold: float = 0.7,
    verbose: bool = False
) -> InsightReport:
    """
    Quick insights generation.

    Args:
        data: Input data
        confidence_threshold: Minimum confidence
        verbose: Print progress

    Returns:
        InsightReport

    Example:
        >>> report = generate_insights(df, verbose=True)
        >>> print(report.summary)
        >>> print(report.to_markdown())
    """
    engine = EnhancedInsightsEngine(data, confidence_threshold)
    return engine.generate_report(verbose=verbose)


def quick_summary(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    confidence_threshold: float = 0.7
) -> str:
    """
    Get quick summary string.

    Args:
        data: Input data
        confidence_threshold: Minimum confidence

    Returns:
        Summary string

    Example:
        >>> summary = quick_summary(df)
        >>> print(summary)
    """
    report = generate_insights(data, confidence_threshold)
    return report.summary
