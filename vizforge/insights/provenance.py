"""
VizForge Insight Provenance Engine v2.0

Track evidence and reasoning behind every insight.
"""
import numpy as np
from scipy import stats
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class Insight:
    """Insight with full provenance tracking."""
    type: str  # "trend", "anomaly", "correlation", "distribution"
    description: str
    confidence: float
    evidence: List[Dict]
    alternative_explanations: List[Dict]
    recommended_actions: List[str]


class InsightProvenanceEngine:
    """
    Generate insights with full evidence tracking.

    Every insight includes:
    - Statistical test results
    - Confidence scores
    - Alternative explanations
    - Recommended actions
    """

    def detect_trend(self, data: np.ndarray) -> Insight:
        """Detect and prove trend with evidence."""
        # Linear regression
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)

        # Evidence collection
        evidence = [
            {
                "test": "linear_regression",
                "slope": round(slope, 4),
                "r_squared": round(r_value**2, 4),
                "p_value": round(p_value, 4),
                "interpretation": f"Slope={slope:.4f}, explains {r_value**2:.1%} of variance"
            }
        ]

        # Mann-Kendall trend test (non-parametric)
        try:
            tau, p_mk = stats.kendalltau(x, data)
            evidence.append({
                "test": "mann_kendall",
                "tau": round(tau, 4),
                "p_value": round(p_mk, 4),
                "interpretation": f"Kendall's tau={tau:.4f} (non-parametric confirmation)"
            })
        except:
            pass

        # Direction and significance
        direction = "upward" if slope > 0 else "downward"
        significant = p_value < 0.05
        confidence = (1 - p_value) if significant else 0.5

        description = f"{direction.capitalize()} trend detected"
        if significant:
            description += f" (statistically significant, p={p_value:.4f})"

        return Insight(
            type="trend",
            description=description,
            confidence=round(confidence, 3),
            evidence=evidence,
            alternative_explanations=[
                {"hypothesis": "Random variation", "probability": round(p_value, 3)}
            ],
            recommended_actions=[
                "Verify trend continues in future periods",
                "Check for external factors driving trend"
            ]
        )

    def detect_anomalies(self, data: np.ndarray, threshold: float = 3.0) -> List[Insight]:
        """Detect anomalies with statistical evidence."""
        mean = np.mean(data)
        std = np.std(data)

        anomalies = []
        for i, value in enumerate(data):
            z_score = abs((value - mean) / (std + 1e-9))

            if z_score > threshold:
                # IQR method confirmation
                q1, q3 = np.percentile(data, [25, 75])
                iqr = q3 - q1
                iqr_breach = (value < q1 - 1.5*iqr) or (value > q3 + 1.5*iqr)

                evidence = [
                    {
                        "test": "z_score",
                        "value": round(z_score, 2),
                        "threshold": threshold,
                        "interpretation": f"{z_score:.2f} standard deviations from mean"
                    }
                ]

                if iqr_breach:
                    evidence.append({
                        "test": "iqr_breach",
                        "interpretation": "Confirmed by IQR method"
                    })

                anomalies.append(Insight(
                    type="anomaly",
                    description=f"Anomaly at index {i}: value={value:.2f}",
                    confidence=min(z_score / 5.0, 0.99),  # Cap at 0.99
                    evidence=evidence,
                    alternative_explanations=[
                        {"hypothesis": "Data entry error", "probability": 0.3},
                        {"hypothesis": "Genuine outlier event", "probability": 0.7}
                    ],
                    recommended_actions=[
                        "Verify data source",
                        "Investigate context around this data point"
                    ]
                ))

        return anomalies
