"""DriftReport: the result bundle returned by Monitor.drift_report()."""
from __future__ import annotations

import html
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DriftReport:
    generated_at: float = field(default_factory=time.time)
    data_drift: Dict[str, dict] = field(default_factory=dict)
    prediction_drift: Optional[dict] = None
    concept_drift: Optional[dict] = None
    correlation_drift: Optional[dict] = None
    n_samples: int = 0

    def drifted_features(self) -> List[str]:
        return [f for f, r in self.data_drift.items() if r.get("is_drifted")]

    def top_drifted(self, n: int = 5) -> List[Dict[str, Any]]:
        rows = [{"feature": f, **r} for f, r in self.data_drift.items()]
        rows.sort(key=lambda r: r.get("drift_score", 0), reverse=True)
        return rows[:n]

    def to_dict(self) -> dict:
        d = {
            "generated_at": self.generated_at,
            "n_samples": self.n_samples,
            "data_drift": self.data_drift,
            "prediction_drift": self.prediction_drift,
            "concept_drift": self.concept_drift,
            "n_drifted_features": len(self.drifted_features()),
        }
        if self.correlation_drift is not None:
            d["correlation_drift"] = {
                "flagged_pairs": self.correlation_drift.get("flagged_pairs", []),
                "flagged_features": self.correlation_drift.get("flagged_features", []),
            }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, indent=2)

    def to_html(self, path: str) -> str:
        rows = []
        for feature, r in sorted(self.data_drift.items(), key=lambda kv: kv[1].get("drift_score", 0), reverse=True):
            rows.append(
                f"<tr><td>{html.escape(feature)}</td><td>{r.get('method')}</td>"
                f"<td>{r.get('drift_score', 0):.4f}</td><td>{r.get('is_drifted')}</td>"
                f"<td>{r.get('severity') or '-'}</td></tr>"
            )
        pred_html = ""
        if self.prediction_drift:
            pd_ = self.prediction_drift
            pred_html = (
                f"<p>Prediction drift score: {pd_.get('drift_score', 0):.4f} "
                f"(drifted: {pd_.get('is_drifted')}, mean shift: {pd_.get('mean_shift_pct', 0):.2%})</p>"
            )
        concept_html = ""
        if self.concept_drift:
            cd = self.concept_drift
            concept_html = (
                f"<p>Concept drift ({cd.get('method')}): detected={cd.get('drift_detected')}, "
                f"baseline error rate={cd.get('baseline_error_rate')}, "
                f"current error rate={cd.get('current_error_rate')}</p>"
            )
        doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ml-monitor drift report</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
th {{ background: #f0f0f0; }}
</style></head>
<body>
<h1>ml-monitor drift report</h1>
<p>Generated at {time.ctime(self.generated_at)} &mdash; {self.n_samples} samples in window.</p>
{pred_html}
{concept_html}
<h2>Feature drift</h2>
<table><thead><tr><th>Feature</th><th>Method</th><th>Drift score</th><th>Drifted</th><th>Severity</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
</body></html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
        return path
