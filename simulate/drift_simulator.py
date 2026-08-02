"""Synthetic drift injector for demo/testing. Injects gradual, sudden, or
seasonal drift into a numerical column of a dataset.

CLI:
    python simulate/drift_simulator.py --dataset data/reference.csv \
        --drift-type gradual --feature amount --shift 2.0
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def inject_drift(df: pd.DataFrame, feature: str, drift_type: str = "gradual", shift: float = 1.0,
                  start_frac: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """Return a copy of `df` with drift injected into `feature`.

    - sudden: from `start_frac` onward, add `shift` * std to the feature.
    - gradual: linearly ramp the shift from 0 to `shift` * std over the tail.
    - seasonal: add a sinusoidal oscillation of amplitude `shift` * std.
    """
    np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    std = out[feature].std() or 1.0
    start_idx = int(n * start_frac)

    if drift_type == "sudden":
        delta = np.zeros(n)
        delta[start_idx:] = shift * std
    elif drift_type == "gradual":
        delta = np.zeros(n)
        tail = n - start_idx
        delta[start_idx:] = np.linspace(0, shift * std, tail)
    elif drift_type == "seasonal":
        delta = shift * std * np.sin(np.linspace(0, 6 * np.pi, n))
    else:
        raise ValueError(f"Unknown drift_type: {drift_type}")

    out[feature] = out[feature].to_numpy() + delta
    return out


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Inject synthetic drift into a dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--drift-type", default="gradual", choices=["gradual", "sudden", "seasonal"])
    parser.add_argument("--feature", required=True)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    drifted = inject_drift(df, feature=args.feature, drift_type=args.drift_type, shift=args.shift)
    output = args.output or args.dataset.replace(".csv", f"_{args.drift_type}_drift.csv")
    drifted.to_csv(output, index=False)
    print(f"Drifted dataset ({args.drift_type}, shift={args.shift}) written to {output}")


if __name__ == "__main__":
    _main()
