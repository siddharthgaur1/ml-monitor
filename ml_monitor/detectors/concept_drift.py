"""Concept drift detectors operating on a stream of 0/1 error indicators
(1 = model prediction was wrong, 0 = correct). Require ground-truth labels.

Implements ADWIN, Page-Hinkley, and DDM as compact, working algorithms
(not stubs) following their standard published formulations:
  - ADWIN: Bifet & Gavalda, "Learning from Time-Changing Data with Adaptive
    Windowing" (2007) - exponential-histogram bucket variant.
  - Page-Hinkley: cumulative deviation test (Page, 1954).
  - DDM: Gama et al., "Learning with Drift Detection" (2004).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# ADWIN
# ---------------------------------------------------------------------------
class _Bucket:
    __slots__ = ("total", "variance", "count")

    def __init__(self, value: float):
        self.total = value
        self.variance = 0.0
        self.count = 1


class ADWIN:
    """Adaptive windowing drift detector. Feed a stream of scalars (error
    indicators); when the distribution of the recent window differs
    significantly from the older window, `drift_detected` becomes True and
    the older window is dropped.
    """

    def __init__(self, delta: float = 0.002, max_buckets: int = 5):
        self.delta = delta
        self.max_buckets = max_buckets  # buckets per "row" of the exponential histogram
        self._buckets: List[List[_Bucket]] = [[]]  # row i holds buckets of size 2**i
        self.width = 0
        self.total = 0.0
        self.variance = 0.0
        self.drift_detected = False
        self.n_seen = 0

    def _insert(self, value: float):
        self._buckets[0].append(_Bucket(value))
        self.width += 1
        self.total += value
        self._compress()

    def _compress(self):
        row = 0
        while row < len(self._buckets):
            if len(self._buckets[row]) <= self.max_buckets:
                break
            if row + 1 >= len(self._buckets):
                self._buckets.append([])
            a, b = self._buckets[row][0], self._buckets[row][1]
            merged = _Bucket(0.0)
            merged.count = a.count + b.count
            merged.total = a.total + b.total
            merged.variance = a.variance + b.variance
            self._buckets[row] = self._buckets[row][2:]
            self._buckets[row + 1].append(merged)
            row += 1

    def _flat_buckets(self):
        for row_idx, row in enumerate(self._buckets):
            for b in row:
                yield b

    def update(self, value: float) -> bool:
        self.n_seen += 1
        self._insert(value)
        self.drift_detected = False
        # Try cut points between accumulated prefix (n0) and suffix (n1)
        buckets = list(self._flat_buckets())
        n0, sum0 = 0, 0.0
        n_total = sum(b.count for b in buckets)
        total_sum = sum(b.total for b in buckets)
        for b in buckets[:-1]:
            n0 += b.count
            sum0 += b.total
            n1 = n_total - n0
            if n0 < 1 or n1 < 1:
                continue
            mean0 = sum0 / n0
            mean1 = (total_sum - sum0) / n1
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps_cut = math.sqrt((2.0 / m) * math.log(4.0 * n_total / self.delta))
            if abs(mean0 - mean1) > eps_cut:
                self.drift_detected = True
                # Drop the older portion (the n0 side)
                self._drop_oldest(n0)
                break
        self.width = sum(b.count for b in self._flat_buckets())
        self.total = sum(b.total for b in self._flat_buckets())
        return self.drift_detected

    def _drop_oldest(self, n0: int):
        remaining = n0
        for row in self._buckets:
            while remaining > 0 and row:
                b = row.pop(0)
                if b.count <= remaining:
                    remaining -= b.count
                else:
                    break

    @property
    def mean(self) -> float:
        return self.total / self.width if self.width else 0.0


# ---------------------------------------------------------------------------
# Page-Hinkley
# ---------------------------------------------------------------------------
class PageHinkley:
    """Page-Hinkley cumulative sum test for detecting a shift in the mean
    of a stream (here, the error rate).
    """

    def __init__(self, delta: float = 0.005, threshold: float = 50.0, alpha: float = 1 - 0.0001):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.mean = 0.0
        self.n = 0
        self.sum = 0.0
        self.min_sum = 0.0
        self.drift_detected = False

    def update(self, value: float) -> bool:
        self.n += 1
        self.mean = self.mean + (value - self.mean) / self.n
        self.sum = self.alpha * self.sum + (value - self.mean - self.delta)
        self.min_sum = min(self.min_sum, self.sum)
        self.drift_detected = (self.sum - self.min_sum) > self.threshold
        if self.drift_detected:
            # reset after signalling, ready to detect the next drift
            self.sum = 0.0
            self.min_sum = 0.0
        return self.drift_detected


# ---------------------------------------------------------------------------
# DDM
# ---------------------------------------------------------------------------
class DDM:
    """Drift Detection Method (Gama et al. 2004). Tracks the running error
    rate p and its std s = sqrt(p(1-p)/n). Flags WARNING when
    p + s >= p_min + 2*s_min, and DRIFT when p + s >= p_min + 3*s_min.
    """

    def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0, min_n: int = 30):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_n = min_n
        self.n = 0
        self.p = 1.0
        self.s = 0.0
        self.p_min = float("inf")
        self.s_min = float("inf")
        self.drift_detected = False
        self.warning_detected = False

    def update(self, error: int) -> bool:
        self.n += 1
        self.p = self.p + (error - self.p) / self.n
        self.s = math.sqrt(self.p * (1 - self.p) / self.n) if self.n else 0.0

        self.drift_detected = False
        self.warning_detected = False

        if self.n < self.min_n:
            return False

        # p_min/s_min only start tracking once we have a stable-enough estimate
        # (matches the reference DDM implementation) -- otherwise an early
        # lucky streak of zero errors locks p_min=0 and every later error
        # falsely reads as drift.
        if self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s

        if self.p + self.s >= self.p_min + self.drift_level * self.s_min:
            self.drift_detected = True
            # reset for next concept
            self.n = 0
            self.p = 1.0
            self.s = 0.0
            self.p_min = float("inf")
            self.s_min = float("inf")
        elif self.p + self.s >= self.p_min + self.warning_level * self.s_min:
            self.warning_detected = True
        return self.drift_detected


# ---------------------------------------------------------------------------
# Convenience: run a detector over a full error stream and summarize
# ---------------------------------------------------------------------------
_DETECTORS = {"adwin": ADWIN, "page_hinkley": PageHinkley, "ddm": DDM}


def detect_concept_drift(errors: list, method: str = "ddm", baseline_size: int = 30) -> dict:
    """Run the given detector over a 0/1 error stream (1 = misprediction).
    Returns {drift_detected, drift_point_index, current_error_rate, baseline_error_rate, method}.
    """
    if method not in _DETECTORS:
        raise ValueError(f"Unknown concept drift method: {method}")
    if len(errors) == 0:
        return {
            "drift_detected": False,
            "drift_point_index": None,
            "current_error_rate": None,
            "baseline_error_rate": None,
            "method": method,
        }

    detector = _DETECTORS[method]()
    drift_point = None
    for i, e in enumerate(errors):
        detected = detector.update(float(e))
        if detected and drift_point is None:
            drift_point = i

    baseline = errors[: min(baseline_size, len(errors))]
    recent = errors[-min(baseline_size, len(errors)):]
    baseline_rate = sum(baseline) / len(baseline)
    current_rate = sum(recent) / len(recent)

    return {
        "drift_detected": drift_point is not None,
        "drift_point_index": drift_point,
        "current_error_rate": float(current_rate),
        "baseline_error_rate": float(baseline_rate),
        "method": method,
    }
