"""
VizForge Chart Reasoning Engine v2.0

Intelligent chart type selection based on data characteristics,
user intent, and visualization best practices.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Literal, Optional
from dataclasses import dataclass


@dataclass
class ChartDecision:
    """Result of chart reasoning process."""
    recommended: str
    confidence: float
    rejected: Dict[str, str]  # chart_type -> reason
    reasoning: List[str]
    alternatives: List[Dict[str, float]]  # [{type: score}, ...]
    data_profile: Dict
    risks: List[str]


class ChartReasoningEngine:
    """
    Intelligent chart type selection engine.

    Makes data-driven decisions about optimal chart types based on:
    - Data characteristics (temporal, categorical, continuous)
    - Data shape (cardinality, distribution)
    - User intent (trend, comparison, composition, distribution)
    - Visualization best practices
    - Cognitive load considerations
    """

    CHART_TYPES = [
        "line", "bar", "scatter", "histogram", "box", "violin",
        "heatmap", "area", "pie", "donut", "radar", "treemap"
    ]

    def __init__(self):
        self.decision_rules = self._build_decision_rules()

    def _build_decision_rules(self) -> Dict:
        """Build rule-based decision tree."""
        return {
            "temporal_continuous": {
                "preferred": ["line", "area"],
                "avoid": ["pie", "bar", "radar"],
                "reasoning": "Temporal continuity requires continuous visual encoding"
            },
            "categorical_comparison": {
                "preferred": ["bar", "dot"],
                "avoid": ["line", "area"],
                "reasoning": "Categorical data needs discrete visual separation"
            },
            "distribution_single": {
                "preferred": ["histogram", "density", "box", "violin"],
                "avoid": ["line", "pie"],
                "reasoning": "Distribution analysis requires shape visualization"
            },
            "composition": {
                "preferred": ["stacked_bar", "treemap", "pie"],
                "avoid": ["line", "scatter"],
                "reasoning": "Part-to-whole relationships need area encoding"
            },
            "correlation": {
                "preferred": ["scatter", "heatmap"],
                "avoid": ["pie", "bar"],
                "reasoning": "Correlation requires 2D position encoding"
            },
            "high_cardinality": {
                "preferred": ["heatmap", "treemap"],
                "avoid": ["pie", "radar", "bar"],
                "reasoning": "High cardinality (>12 categories) causes visual clutter"
            }
        }

    def analyze_data(self, df: pd.DataFrame, x_col: str = None, y_col: str = None) -> Dict:
        """
        Profile data characteristics.

        Returns:
            Dict with data features for reasoning
        """
        profile = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "has_temporal": False,
            "has_categorical": False,
            "has_continuous": False,
            "cardinality": {},
            "null_ratio": {},
            "data_types": {}
        }

        for col in df.columns:
            dtype = df[col].dtype

            # Data type classification
            if pd.api.types.is_datetime64_any_dtype(dtype):
                profile["has_temporal"] = True
                profile["data_types"][col] = "temporal"
            elif pd.api.types.is_numeric_dtype(dtype):
                profile["has_continuous"] = True
                profile["data_types"][col] = "continuous"
                profile["cardinality"][col] = df[col].nunique()
            elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                profile["has_categorical"] = True
                profile["data_types"][col] = "categorical"
                profile["cardinality"][col] = df[col].nunique()

            # Null ratio
            profile["null_ratio"][col] = df[col].isna().sum() / len(df)

        # Special analyses
        if x_col and y_col:
            profile["x_type"] = profile["data_types"].get(x_col, "unknown")
            profile["y_type"] = profile["data_types"].get(y_col, "unknown")
            profile["x_cardinality"] = profile["cardinality"].get(x_col, 0)
            profile["y_cardinality"] = profile["cardinality"].get(y_col, 0)

            # Check for correlation (if both numeric)
            if profile["x_type"] == "continuous" and profile["y_type"] == "continuous":
                try:
                    profile["correlation"] = df[[x_col, y_col]].corr().iloc[0, 1]
                except:
                    profile["correlation"] = None

        return profile

    def infer_intent(
        self,
        data_profile: Dict,
        explicit_intent: Optional[str] = None
    ) -> str:
        """
        Infer user intent from data characteristics.

        Args:
            data_profile: Result from analyze_data()
            explicit_intent: User-specified intent (overrides inference)

        Returns:
            Intent string: trend, comparison, composition, distribution, correlation
        """
        if explicit_intent:
            return explicit_intent

        # Rule-based intent inference
        if data_profile.get("has_temporal"):
            return "trend"

        if data_profile.get("x_type") == "categorical" and data_profile.get("y_type") == "continuous":
            return "comparison"

        if data_profile.get("x_type") == "continuous" and data_profile.get("y_type") == "continuous":
            return "correlation"

        if data_profile.get("n_cols") == 1 and data_profile.get("has_continuous"):
            return "distribution"

        # Default
        return "comparison"

    def score_chart_types(
        self,
        data_profile: Dict,
        intent: str
    ) -> Dict[str, float]:
        """
        Score all chart types based on data and intent.

        Returns:
            Dict mapping chart_type -> suitability_score (0-1)
        """
        scores = {}

        for chart_type in self.CHART_TYPES:
            score = 0.5  # Baseline

            # Temporal data
            if data_profile.get("has_temporal"):
                if chart_type in ["line", "area"]:
                    score += 0.3
                elif chart_type in ["pie", "radar", "bar"]:
                    score -= 0.4

            # High cardinality penalty
            max_cardinality = max(data_profile.get("cardinality", {}).values() or [0])
            if max_cardinality > 12:
                if chart_type in ["pie", "radar", "bar"]:
                    score -= 0.3
                if chart_type in ["heatmap", "treemap"]:
                    score += 0.2

            # Intent-based scoring
            intent_boosts = {
                "trend": {"line": 0.4, "area": 0.3, "scatter": 0.1},
                "comparison": {"bar": 0.4, "dot": 0.3, "box": 0.2},
                "composition": {"pie": 0.3, "treemap": 0.4, "stacked_bar": 0.3},
                "distribution": {"histogram": 0.4, "box": 0.3, "violin": 0.3},
                "correlation": {"scatter": 0.4, "heatmap": 0.3}
            }

            if intent in intent_boosts and chart_type in intent_boosts[intent]:
                score += intent_boosts[intent][chart_type]

            # Continuous vs categorical
            if data_profile.get("x_type") == "categorical":
                if chart_type == "line":
                    score -= 0.3  # Lines imply continuity

            scores[chart_type] = max(0.0, min(1.0, score))  # Clamp to [0, 1]

        return scores

    def generate_rejection_reasons(
        self,
        data_profile: Dict,
        intent: str,
        scores: Dict[str, float]
    ) -> Dict[str, str]:
        """Generate human-readable rejection reasons for low-scoring charts."""
        rejected = {}
        threshold = 0.4

        for chart_type, score in scores.items():
            if score < threshold:
                reasons = []

                # Temporal mismatch
                if data_profile.get("has_temporal") and chart_type in ["pie", "radar"]:
                    reasons.append("loses temporal continuity")

                # Categorical mismatch
                if data_profile.get("x_type") == "categorical" and chart_type == "line":
                    reasons.append("categorical data requires discrete encoding")

                # High cardinality
                max_card = max(data_profile.get("cardinality", {}).values() or [0])
                if max_card > 12 and chart_type in ["pie", "radar"]:
                    reasons.append(f"too many categories ({max_card}) causes clutter")

                # Intent mismatch
                if intent == "distribution" and chart_type in ["line", "pie"]:
                    reasons.append("doesn't show distribution shape")

                if intent == "correlation" and chart_type in ["pie", "bar"]:
                    reasons.append("can't show 2D correlation")

                if reasons:
                    rejected[chart_type] = "; ".join(reasons)

        return rejected

    def assess_risks(
        self,
        chart_type: str,
        data_profile: Dict
    ) -> List[str]:
        """Identify potential risks/biases for chosen chart."""
        risks = []

        # Truncated axis risk
        if chart_type in ["bar", "line"]:
            risks.append("Ensure y-axis starts at zero to avoid scale distortion")

        # Overplotting risk
        if chart_type == "scatter" and data_profile.get("n_rows", 0) > 1000:
            risks.append("High point density may cause overplotting - consider hexbin or density plot")

        # Pie chart risks
        if chart_type in ["pie", "donut"]:
            cardinality = max(data_profile.get("cardinality", {}).values() or [0])
            if cardinality > 5:
                risks.append("Too many slices reduce readability - consider bar chart")
            risks.append("Humans struggle with angle/area comparison - consider alternatives")

        # Dual axis risk
        if chart_type in ["line", "bar"] and data_profile.get("n_cols", 0) > 2:
            risks.append("Dual y-axes can be misleading - ensure scale appropriateness")

        # 3D risk
        if chart_type in ["3d_bar", "3d_pie"]:
            risks.append("3D effects distort perception - avoid unless specifically needed")

        return risks

    def decide(
        self,
        df: pd.DataFrame,
        x_col: str = None,
        y_col: str = None,
        intent: Optional[str] = None
    ) -> ChartDecision:
        """
        Make intelligent chart type decision.

        Args:
            df: Data to visualize
            x_col: X-axis column
            y_col: Y-axis column
            intent: User intent (or None to infer)

        Returns:
            ChartDecision object with recommendation and reasoning
        """
        # Step 1: Analyze data
        data_profile = self.analyze_data(df, x_col, y_col)

        # Step 2: Infer or use explicit intent
        inferred_intent = self.infer_intent(data_profile, intent)

        # Step 3: Score all chart types
        scores = self.score_chart_types(data_profile, inferred_intent)

        # Step 4: Select best chart
        recommended = max(scores, key=scores.get)
        confidence = scores[recommended]

        # Step 5: Generate rejection reasons
        rejected = self.generate_rejection_reasons(data_profile, inferred_intent, scores)

        # Step 6: Get alternatives (top 3 excluding recommended)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        alternatives = [
            {"type": t, "score": round(s, 3)}
            for t, s in sorted_scores[1:4]  # Next 3 best
        ]

        # Step 7: Assess risks
        risks = self.assess_risks(recommended, data_profile)

        # Step 8: Build reasoning trace
        reasoning = [
            f"Data profile: {data_profile.get('n_rows')} rows, {data_profile.get('n_cols')} columns",
            f"Detected types: {', '.join(set(data_profile.get('data_types', {}).values()))}",
            f"Inferred intent: {inferred_intent}",
            f"Recommended: {recommended} (confidence: {confidence:.2f})"
        ]

        if data_profile.get("has_temporal"):
            reasoning.append("Temporal data detected - prioritizing continuity")

        max_card = max(data_profile.get("cardinality", {}).values() or [0])
        if max_card > 12:
            reasoning.append(f"High cardinality ({max_card}) - avoiding cluttered charts")

        return ChartDecision(
            recommended=recommended,
            confidence=round(confidence, 3),
            rejected=rejected,
            reasoning=reasoning,
            alternatives=alternatives,
            data_profile=data_profile,
            risks=risks
        )

    def explain_decision(self, decision: ChartDecision) -> str:
        """Generate human-readable explanation of decision."""
        explanation = f"""
Chart Recommendation: {decision.recommended.upper()}
Confidence: {decision.confidence:.1%}

WHY THIS CHART?
{chr(10).join(f"  • {r}" for r in decision.reasoning)}

REJECTED ALTERNATIVES:
{chr(10).join(f"  ✗ {chart}: {reason}" for chart, reason in decision.rejected.items())}

OTHER OPTIONS:
{chr(10).join(f"  • {alt['type']}: {alt['score']:.1%} suitable" for alt in decision.alternatives)}

RISKS TO WATCH:
{chr(10).join(f"  ⚠️  {risk}" for risk in decision.risks)}
        """
        return explanation.strip()
