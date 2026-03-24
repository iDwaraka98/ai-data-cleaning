"""
utils.py
--------
Helper functions shared across the pipeline.
"""

import pandas as pd
import numpy as np
import re
import json
import os
from datetime import datetime
from colorama import Fore, Style, init

init(autoreset=True)


# ─────────────────────────────────────────────
# COLOUR LOGGING
# ─────────────────────────────────────────────

def log_info(msg: str):
    print(f"{Fore.CYAN}[INFO]  {Style.RESET_ALL}{msg}")

def log_success(msg: str):
    print(f"{Fore.GREEN}[OK]    {Style.RESET_ALL}{msg}")

def log_warn(msg: str):
    print(f"{Fore.YELLOW}[WARN]  {Style.RESET_ALL}{msg}")

def log_error(msg: str):
    print(f"{Fore.RED}[ERROR] {Style.RESET_ALL}{msg}")

def log_section(title: str):
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Style.RESET_ALL}\n")


# ─────────────────────────────────────────────
# DATA LOADING & CLEANING HELPERS
# ─────────────────────────────────────────────

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load CSV with UTF-8 BOM handling."""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    log_success(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def clean_currency(series: pd.Series) -> pd.Series:
    """Strip $, commas, spaces and convert to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(r'[$,\s]', '', regex=True),
        errors='coerce'
    )


def clean_integer(series: pd.Series) -> pd.Series:
    """Strip commas/spaces and convert to Int64."""
    return pd.to_numeric(
        series.astype(str).str.replace(r'[,\s]', '', regex=True),
        errors='coerce'
    ).astype('Int64')


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse dates with multiple format support."""
    return pd.to_datetime(series, errors='coerce', infer_datetime_format=True)


def standardise_text(series: pd.Series) -> pd.Series:
    """Strip whitespace and title-case a text column."""
    return series.astype(str).str.strip().str.title()


# ─────────────────────────────────────────────
# VOCABULARY / REFERENCE LISTS
# ─────────────────────────────────────────────

VOCABULARY = {
    'Product': [
        "Men's Street Footwear",
        "Men's Athletic Footwear",
        "Men's Apparel",
        "Women's Street Footwear",
        "Women's Athletic Footwear",
        "Women's Apparel",
    ],
    'Region': ['West', 'Northeast', 'Midwest', 'South', 'Southeast'],
    'Sales Method': ['Online', 'Outlet', 'In-store'],
    'Retailer': ['Foot Locker', 'West Gear', 'Sports Direct', "Kohl's", 'Amazon', 'Walmart'],
}

NULL_EQUIVALENTS = ['', ' ', 'N/A', 'Unknown', '-', 'null', 'NULL', 'nan', 'NaN', 'none', 'None']


# ─────────────────────────────────────────────
# JSON PARSING
# ─────────────────────────────────────────────

def safe_parse_json(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences."""
    if not text:
        return {}
    text = re.sub(r'```(?:json)?', '', text).strip('`').strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract first JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {}


# ─────────────────────────────────────────────
# REPORT SAVING
# ─────────────────────────────────────────────

def save_report(data: list, filepath: str):
    """Save a list of dicts as a JSON report."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    log_success(f"Report saved → {filepath}")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
