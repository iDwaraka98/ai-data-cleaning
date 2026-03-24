"""
baseline.py
-----------
Rule-based baseline cleaner for comparison against the AI pipeline.
Applies deterministic fixes only — no LLM calls.

Usage:
    python baseline.py
    python baseline.py --input data/data_sales.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import re
from fuzzywuzzy import process
from modules.utils import (
    load_dataset, clean_currency, clean_integer, parse_dates,
    VOCABULARY, timestamp, log_section, log_info, log_success, log_warn
)


class RuleBasedCleaner:

    def __init__(self, df: pd.DataFrame):
        self.df_original = df.copy()
        self.df_clean    = df.copy()
        self.fixes       = []

    # ──────────────────────────────────────────
    # FIX 1: Missing Values
    # ──────────────────────────────────────────
    def fix_missing_values(self):
        log_info("Fixing missing values (mode/median imputation)...")
        null_eq = ['', ' ', 'N/A', 'Unknown', '-', 'null', 'NULL', 'nan']
        count = 0
        for col in self.df_clean.columns:
            mask = (self.df_clean[col].isnull() |
                    self.df_clean[col].astype(str).str.strip().isin(null_eq))
            if mask.sum() == 0:
                continue
            if col in ['Price per Unit', 'Total Sales', 'Operating Profit']:
                series = clean_currency(self.df_clean[col])
                fill   = series.median()
                self.df_clean.loc[mask, col] = f"${fill:,.2f}"
            elif col == 'Units Sold':
                series = clean_integer(self.df_clean[col]).astype(float)
                fill   = int(series.median())
                self.df_clean.loc[mask, col] = str(fill)
            else:
                fill = self.df_clean[col].mode()[0] if len(self.df_clean[col].mode()) > 0 else 'Unknown'
                self.df_clean.loc[mask, col] = fill
            count += mask.sum()
            log_info(f"  {col}: filled {mask.sum()} missing values with '{fill}'")
        log_success(f"  Total missing values fixed: {count}")

    # ──────────────────────────────────────────
    # FIX 2: Typos — closest vocabulary match
    # ──────────────────────────────────────────
    def fix_typographical_errors(self):
        log_info("Fixing typographical errors (Levenshtein nearest match)...")
        count = 0
        for col, valid in VOCABULARY.items():
            if col not in self.df_clean.columns:
                continue
            for idx, val in self.df_clean[col].items():
                val_str = str(val).strip()
                if val_str not in valid:
                    best, score = process.extractOne(val_str, valid)
                    if score >= 60:
                        self.df_clean.at[idx, col] = best
                        self.fixes.append({'row': idx, 'col': col,
                                           'from': val_str, 'to': best, 'type': 'TYPO'})
                        count += 1
        log_success(f"  Typographical errors fixed: {count}")

    # ──────────────────────────────────────────
    # FIX 3: Format standardisation
    # ──────────────────────────────────────────
    def fix_format_inconsistencies(self):
        log_info("Fixing format inconsistencies...")
        count = 0

        # Normalise Price per Unit to $XX.XX
        for idx, val in self.df_clean['Price per Unit'].items():
            val_str = str(val).strip()
            numeric = re.sub(r'[$,\s]', '', val_str)
            try:
                new_val = f"${float(numeric):.2f}"
                if new_val != val_str:
                    self.df_clean.at[idx, 'Price per Unit'] = new_val
                    count += 1
            except ValueError:
                pass

        log_success(f"  Format inconsistencies fixed: {count}")

    # ──────────────────────────────────────────
    # FIX 4: Duplicates — keep first
    # ──────────────────────────────────────────
    def fix_duplicates(self):
        log_info("Removing duplicate rows...")
        before = len(self.df_clean)
        self.df_clean = self.df_clean.drop_duplicates(keep='first')
        removed = before - len(self.df_clean)
        log_success(f"  Duplicate rows removed: {removed}")

    # ──────────────────────────────────────────
    # FIX 5: Logical errors — recalculate Total Sales
    # ──────────────────────────────────────────
    def fix_logical_errors(self):
        log_info("Fixing logical errors (recalculating Total Sales)...")
        price  = clean_currency(self.df_clean['Price per Unit'])
        units  = clean_integer(self.df_clean['Units Sold']).astype(float)
        sales  = clean_currency(self.df_clean['Total Sales'])
        count  = 0

        for idx in self.df_clean.index:
            p, u, s = price[idx], units[idx], sales[idx]
            if pd.notna(p) and pd.notna(u) and pd.notna(s) and p > 0 and u > 0:
                ratio = s / (p * u)
                if ratio < 0.05 or ratio > 2.0:
                    recalc = p * u
                    self.df_clean.at[idx, 'Total Sales'] = f"${recalc:,.0f}"
                    count += 1
        log_success(f"  Logical errors fixed: {count}")

    # ──────────────────────────────────────────
    # FIX 6: Outliers — replace with median
    # ──────────────────────────────────────────
    def fix_outliers(self):
        log_info("Fixing outliers (replace with column median)...")
        numeric_map = {
            'Price per Unit':   clean_currency(self.df_clean['Price per Unit']),
            'Units Sold':       clean_integer(self.df_clean['Units Sold']).astype(float),
            'Total Sales':      clean_currency(self.df_clean['Total Sales']),
            'Operating Profit': clean_currency(self.df_clean['Operating Profit']),
        }
        count = 0
        for col, series in numeric_map.items():
            s = series.dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 3*iqr, q3 + 3*iqr
            median = s.median()
            outlier_idx = series[(series < lower) | (series > upper)].index
            for idx in outlier_idx:
                if col == 'Units Sold':
                    self.df_clean.at[idx, col] = str(int(median))
                else:
                    self.df_clean.at[idx, col] = f"${median:,.2f}"
                count += 1
        log_success(f"  Outliers fixed: {count}")

    # ──────────────────────────────────────────
    # RUN ALL & SAVE
    # ──────────────────────────────────────────
    def run_all(self, output_dir: str = 'outputs') -> pd.DataFrame:
        log_section("RULE-BASED BASELINE CLEANER")
        self.fix_missing_values()
        self.fix_typographical_errors()
        self.fix_format_inconsistencies()
        self.fix_duplicates()
        self.fix_logical_errors()
        self.fix_outliers()

        os.makedirs(output_dir, exist_ok=True)
        ts   = timestamp()
        path = os.path.join(output_dir, f'baseline_cleaned_{ts}.csv')
        self.df_clean.to_csv(path, index=False)
        log_success(f"\nBaseline cleaned dataset saved → {path}")
        log_success(f"Original rows: {len(self.df_original):,}  |  "
                    f"Cleaned rows: {len(self.df_clean):,}")
        return self.df_clean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rule-Based Baseline Cleaner')
    parser.add_argument('--input',  default='data/data_sales.csv')
    parser.add_argument('--output', default='outputs')
    args = parser.parse_args()

    df  = load_dataset(args.input)
    cleaner = RuleBasedCleaner(df)
    cleaner.run_all(output_dir=args.output)
