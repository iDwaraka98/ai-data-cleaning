"""
validator.py
------------
Module 3: Multi-layer validation and confidence scoring.

Layers:
  1. Format Validation       — regex / datatype checks
  2. Logical Consistency     — cross-field business rules
  3. Statistical Plausibility — z-score vs column distribution

Confidence Score:
  score = 0.35 * model_conf + 0.25 * format + 0.25 * logic + 0.15 * stats

Routing:
  >= 0.80  → AUTO_APPLY
  0.50–0.79 → APPLY_FLAG  (applied but flagged for review)
  < 0.50   → HUMAN_REVIEW
"""

import pandas as pd
import numpy as np
import re
from modules.utils import VOCABULARY, log_info


# ── Weights (calibrated from training set) ─────────────────────────────
WEIGHTS = {
    'model_confidence': 0.35,
    'format':           0.25,
    'logical':          0.25,
    'statistical':      0.15,
}

AUTO_APPLY_THRESHOLD   = 0.80
HUMAN_REVIEW_THRESHOLD = 0.50


class Validator:
    """
    Validates a proposed correction and assigns a confidence score.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._col_stats = self._compute_stats()

    # ──────────────────────────────────────────────────────────────────
    # COLUMN STATISTICS (pre-computed)
    # ──────────────────────────────────────────────────────────────────

    def _compute_stats(self) -> dict:
        from modules.utils import clean_currency, clean_integer

        stats = {}
        numeric_map = {
            'Price per Unit':    clean_currency(self.df['Price per Unit']),
            'Units Sold':        clean_integer(self.df['Units Sold']).astype(float),
            'Total Sales':       clean_currency(self.df['Total Sales']),
            'Operating Profit':  clean_currency(self.df['Operating Profit']),
        }
        for col, series in numeric_map.items():
            s = series.dropna()
            if len(s) > 0:
                stats[col] = {
                    'mean':   s.mean(),
                    'std':    s.std() + 1e-8,
                    'median': s.median(),
                    'q1':     s.quantile(0.25),
                    'q3':     s.quantile(0.75),
                    'min':    s.min(),
                    'max':    s.max(),
                }
        return stats

    # ──────────────────────────────────────────────────────────────────
    # LAYER 1: FORMAT VALIDATION
    # ──────────────────────────────────────────────────────────────────

    def validate_format(self, value, column: str) -> tuple:
        """Returns (passes: bool, score: float)."""
        val = str(value).strip()

        if column == 'Invoice Date':
            patterns = [
                r'^\d{1,2}/\d{1,2}/\d{4}$',   # M/D/YYYY
                r'^\d{4}-\d{2}-\d{2}$',         # ISO
                r'^\d{2}-[A-Za-z]{3}-\d{4}$',   # DD-Mon-YYYY
            ]
            passes = any(re.match(p, val) for p in patterns)
            return passes, 1.0 if passes else 0.0

        if column == 'Price per Unit':
            passes = bool(re.match(r'^\$\d+(\.\d{1,2})?\s*$', val))
            return passes, 1.0 if passes else 0.3

        if column in VOCABULARY:
            passes = val in VOCABULARY[column]
            return passes, 1.0 if passes else 0.0

        if column in ['Units Sold']:
            try:
                v = int(str(value).replace(',', '').strip())
                passes = v >= 0
                return passes, 1.0 if passes else 0.0
            except (ValueError, TypeError):
                return False, 0.0

        if column in ['Total Sales', 'Operating Profit']:
            try:
                v = float(str(value).replace('$', '').replace(',', '').strip())
                passes = v >= 0
                return passes, 1.0 if passes else 0.0
            except (ValueError, TypeError):
                return False, 0.0

        return True, 1.0   # Default: unknown column → pass

    # ──────────────────────────────────────────────────────────────────
    # LAYER 2: LOGICAL CONSISTENCY
    # ──────────────────────────────────────────────────────────────────

    def validate_logic(self, value, column: str, record: dict) -> tuple:
        """Returns (passes: bool, score: float)."""

        def _to_float(v):
            try:
                return float(str(v).replace('$', '').replace(',', '').strip())
            except (ValueError, TypeError):
                return None

        if column == 'Units Sold':
            v = _to_float(value)
            if v is not None:
                return (v > 0, 1.0 if v > 0 else 0.0)
            return False, 0.0

        if column == 'Price per Unit':
            v = _to_float(value)
            if v is not None:
                return (v > 0, 1.0 if v > 0 else 0.0)
            return False, 0.0

        if column == 'Total Sales':
            price = _to_float(record.get('Price per Unit'))
            units = _to_float(record.get('Units Sold'))
            sales = _to_float(value)
            if all(x is not None for x in [price, units, sales]):
                expected = price * units
                # Allow ratio between 0.5 and 1.5 (discounts / rounding)
                ratio = sales / expected if expected > 0 else 0
                passes = 0.05 <= ratio <= 1.5
                score  = 1.0 if 0.5 <= ratio <= 1.05 else 0.5 if passes else 0.0
                return passes, score
            return True, 0.5  # Can't verify — neutral

        if column == 'Operating Profit':
            profit = _to_float(value)
            sales  = _to_float(record.get('Total Sales'))
            if profit is not None and sales is not None:
                # Profit should be ≥0 and < Total Sales
                passes = 0 <= profit < sales
                return passes, 1.0 if passes else 0.0
            return True, 0.5

        return True, 1.0

    # ──────────────────────────────────────────────────────────────────
    # LAYER 3: STATISTICAL PLAUSIBILITY
    # ──────────────────────────────────────────────────────────────────

    def validate_statistics(self, value, column: str) -> tuple:
        """Returns (passes: bool, score: float 0–1)."""
        if column not in self._col_stats:
            return True, 1.0

        try:
            v = float(str(value).replace('$', '').replace(',', '').strip())
        except (ValueError, TypeError):
            return True, 1.0

        s = self._col_stats[column]
        z = abs((v - s['mean']) / s['std'])

        if z <= 2.0:
            return True,  1.0
        elif z <= 3.0:
            return True,  0.7
        elif z <= 4.0:
            return False, 0.4
        else:
            return False, max(0.0, 0.4 - (z - 4.0) * 0.1)

    # ──────────────────────────────────────────────────────────────────
    # COMPOSITE CONFIDENCE SCORE
    # ──────────────────────────────────────────────────────────────────

    def score(self, corrected_value, column: str,
              record: dict, model_confidence: float) -> dict:
        """
        Returns a dict with:
          confidence_score, routing, format_pass, logic_pass, stat_pass,
          format_score, logic_score, stat_score
        """
        fmt_pass,  fmt_score  = self.validate_format(corrected_value, column)
        log_pass,  log_score  = self.validate_logic(corrected_value, column, record)
        stat_pass, stat_score = self.validate_statistics(corrected_value, column)

        composite = (
            WEIGHTS['model_confidence'] * float(model_confidence) +
            WEIGHTS['format']           * fmt_score +
            WEIGHTS['logical']          * log_score +
            WEIGHTS['statistical']      * stat_score
        )
        composite = round(min(max(composite, 0.0), 1.0), 4)

        if composite >= AUTO_APPLY_THRESHOLD:
            routing = 'AUTO_APPLY'
        elif composite >= HUMAN_REVIEW_THRESHOLD:
            routing = 'APPLY_FLAG'
        else:
            routing = 'HUMAN_REVIEW'

        return {
            'confidence_score': composite,
            'routing':          routing,
            'format_pass':      fmt_pass,
            'logic_pass':       log_pass,
            'stat_pass':        stat_pass,
            'format_score':     fmt_score,
            'logic_score':      log_score,
            'stat_score':       stat_score,
        }
