"""
error_detector.py
-----------------
Module 1: Multi-strategy error detection for the retail sales dataset.

Detects:
  1. Missing Values
  2. Typographical Errors  (fuzzy match against vocabulary)
  3. Format Inconsistencies (dates, currency symbols)
  4. Duplicate Records
  5. Logical Errors         (Total Sales != Price x Units)
  6. Outliers               (IQR + Isolation Forest)
"""

import pandas as pd
import numpy as np
import re
from fuzzywuzzy import fuzz, process
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

from modules.utils import (
    VOCABULARY, NULL_EQUIVALENTS,
    clean_currency, clean_integer, parse_dates,
    log_info, log_warn, log_success
)


class ErrorDetector:
    """
    Runs all six detection strategies and returns a unified error report.
    Each error entry is a dict with keys:
        row_id, column, error_type, observed_value, context, suggestion
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.errors = []
        self._prepare_numeric_columns()

    # ─────────────────────────────────────────
    # INTERNAL PREP
    # ─────────────────────────────────────────

    def _prepare_numeric_columns(self):
        """Parse and cache clean numeric versions of currency columns."""
        self.df['_price']  = clean_currency(self.df['Price per Unit'])
        self.df['_units']  = clean_integer(self.df['Units Sold'])
        self.df['_sales']  = clean_currency(self.df['Total Sales'])
        self.df['_profit'] = clean_currency(self.df['Operating Profit'])
        self.df['_date']   = parse_dates(self.df['Invoice Date'])

    def _add_error(self, row_id, column, error_type, observed_value,
                   suggestion=None, context=None):
        self.errors.append({
            'row_id':         int(row_id),
            'column':         column,
            'error_type':     error_type,
            'observed_value': str(observed_value),
            'suggestion':     str(suggestion) if suggestion is not None else None,
            'context':        context or {}
        })

    # ─────────────────────────────────────────
    # 1. MISSING VALUES
    # ─────────────────────────────────────────

    def detect_missing_values(self) -> int:
        log_info("Detecting missing values...")
        count = 0
        for col in self.df.columns:
            if col.startswith('_'):
                continue
            mask = (
                self.df[col].isnull() |
                self.df[col].astype(str).str.strip().isin(NULL_EQUIVALENTS)
            )
            for idx in self.df[mask].index:
                self._add_error(
                    row_id=idx,
                    column=col,
                    error_type='MISSING_VALUE',
                    observed_value=self.df.at[idx, col],
                    context=self.df.loc[idx, [c for c in self.df.columns
                                              if not c.startswith('_')]].to_dict()
                )
                count += 1

        # Also flag nulls introduced during numeric parsing
        for raw_col, parsed_col in [('Price per Unit', '_price'),
                                     ('Units Sold', '_units'),
                                     ('Total Sales', '_sales'),
                                     ('Operating Profit', '_profit')]:
            null_mask = self.df[parsed_col].isnull() & self.df[raw_col].notna()
            for idx in self.df[null_mask].index:
                self._add_error(
                    row_id=idx,
                    column=raw_col,
                    error_type='MISSING_VALUE',
                    observed_value=self.df.at[idx, raw_col],
                    context={}
                )
                count += 1

        log_success(f"  Missing values found: {count}")
        return count

    # ─────────────────────────────────────────
    # 2. TYPOGRAPHICAL ERRORS
    # ─────────────────────────────────────────

    def detect_typographical_errors(self, threshold: int = 60) -> int:
        log_info("Detecting typographical errors...")
        count = 0
        for col, valid_values in VOCABULARY.items():
            if col not in self.df.columns:
                continue
            unique_vals = self.df[col].dropna().unique()
            for val in unique_vals:
                val_str = str(val).strip()
                if val_str not in valid_values:
                    best_match, score = process.extractOne(
                        val_str, valid_values, scorer=fuzz.token_sort_ratio
                    )
                    if score >= threshold:
                        affected = self.df[self.df[col] == val].index
                        for idx in affected:
                            self._add_error(
                                row_id=idx,
                                column=col,
                                error_type='TYPOGRAPHICAL_ERROR',
                                observed_value=val_str,
                                suggestion=best_match,
                                context={
                                    'similarity_score': score,
                                    'best_match': best_match
                                }
                            )
                            count += 1

        log_success(f"  Typographical errors found: {count}")
        return count

    # ─────────────────────────────────────────
    # 3. FORMAT INCONSISTENCIES
    # ─────────────────────────────────────────

    def detect_format_inconsistencies(self) -> int:
        log_info("Detecting format inconsistencies...")
        count = 0

        # Dates that didn't parse
        bad_dates = self.df[self.df['_date'].isnull()].index
        for idx in bad_dates:
            self._add_error(
                row_id=idx,
                column='Invoice Date',
                error_type='FORMAT_INCONSISTENCY',
                observed_value=self.df.at[idx, 'Invoice Date'],
                suggestion='Convert to MM/DD/YYYY or YYYY-MM-DD'
            )
            count += 1

        # Price per Unit: missing $ or extra whitespace
        price_pattern = re.compile(r'^\$\d+(\.\d{1,2})?\s*$')
        for idx, row in self.df.iterrows():
            val = str(row['Price per Unit']).strip()
            if not price_pattern.match(val) and val not in NULL_EQUIVALENTS:
                self._add_error(
                    row_id=idx,
                    column='Price per Unit',
                    error_type='FORMAT_INCONSISTENCY',
                    observed_value=val,
                    suggestion='Should be $XX.XX format'
                )
                count += 1

        log_success(f"  Format inconsistencies found: {count}")
        return count

    # ─────────────────────────────────────────
    # 4. DUPLICATE RECORDS
    # ─────────────────────────────────────────

    def detect_duplicates(self) -> int:
        log_info("Detecting duplicate records...")
        cols = [c for c in self.df.columns if not c.startswith('_')]
        dup_mask = self.df.duplicated(subset=cols, keep='first')
        count = 0
        for idx in self.df[dup_mask].index:
            self._add_error(
                row_id=idx,
                column='ALL',
                error_type='DUPLICATE_RECORD',
                observed_value='Duplicate row',
                suggestion='Remove — keep first occurrence'
            )
            count += 1

        log_success(f"  Duplicate records found: {count}")
        return count

    # ─────────────────────────────────────────
    # 5. LOGICAL ERRORS
    # ─────────────────────────────────────────

    def detect_logical_errors(self, tolerance: float = 1.0) -> int:
        """
        Total Sales should be approximately Price per Unit × Units Sold.
        The dataset stores Total Sales in raw dollars (not thousands),
        but the actual values show a systematic discount/markup factor.
        We flag records where the ratio is wildly outside 0.1 – 1.5.
        """
        log_info("Detecting logical errors...")
        count = 0

        valid = (
            self.df['_price'].notna() &
            self.df['_units'].notna() &
            self.df['_sales'].notna() &
            (self.df['_price'] > 0) &
            (self.df['_units'] > 0) &
            (self.df['_sales'] > 0)
        )
        sub = self.df[valid].copy()

        # Compute implied ratio: Total Sales / (Price × Units)
        sub['_ratio'] = sub['_sales'] / (sub['_price'] * sub['_units'].astype(float))

        # Flag ratios outside expected range (discount 0–50% expected → ratio 0.5–1.0)
        # We use a wider band 0.05 – 2.0 to avoid too many false positives
        bad = sub[(sub['_ratio'] < 0.05) | (sub['_ratio'] > 2.0)]

        for idx, row in bad.iterrows():
            expected = round(row['_price'] * float(row['_units']), 2)
            self._add_error(
                row_id=idx,
                column='Total Sales',
                error_type='LOGICAL_ERROR',
                observed_value=row['_sales'],
                suggestion=f'Expected ~${expected:,.2f} (Price × Units)',
                context={
                    'price': row['_price'],
                    'units': int(row['_units']),
                    'ratio': round(row['_ratio'], 4)
                }
            )
            count += 1

        log_success(f"  Logical errors found: {count}")
        return count

    # ─────────────────────────────────────────
    # 6. OUTLIERS
    # ─────────────────────────────────────────

    def detect_outliers(self) -> int:
        log_info("Detecting outliers...")
        count = 0
        numeric_cols = {
            'Price per Unit': '_price',
            'Units Sold':     '_units',
            'Total Sales':    '_sales',
            'Operating Profit': '_profit',
        }

        for display_col, internal_col in numeric_cols.items():
            series = self.df[internal_col].dropna().astype(float)
            if len(series) < 10:
                continue

            # IQR method
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3.0 * IQR
            upper = Q3 + 3.0 * IQR
            iqr_out = set(series[(series < lower) | (series > upper)].index)

            # Isolation Forest
            iso = IsolationForest(contamination=0.03, random_state=42)
            preds = iso.fit_predict(series.values.reshape(-1, 1))
            iso_out = set(series.index[preds == -1])

            # Flag confirmed by both
            confirmed = iqr_out & iso_out
            for idx in confirmed:
                self._add_error(
                    row_id=idx,
                    column=display_col,
                    error_type='OUTLIER',
                    observed_value=self.df.at[idx, internal_col],
                    suggestion=f'Value outside expected range [{lower:.2f}, {upper:.2f}]'
                )
                count += 1

        log_success(f"  Outliers found: {count}")
        return count

    # ─────────────────────────────────────────
    # RUN ALL
    # ─────────────────────────────────────────

    def run_all(self) -> list:
        log_info("Running all detection strategies...\n")
        self.detect_missing_values()
        self.detect_typographical_errors()
        self.detect_format_inconsistencies()
        self.detect_duplicates()
        self.detect_logical_errors()
        self.detect_outliers()

        # Summary
        from collections import Counter
        summary = Counter(e['error_type'] for e in self.errors)
        log_success(f"\nTotal errors detected: {len(self.errors)}")
        for etype, cnt in summary.most_common():
            print(f"    {etype:<30} {cnt:>6}")

        return self.errors
