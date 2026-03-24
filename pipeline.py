"""
pipeline.py
-----------
⭐ MAIN ENTRY POINT — Run this file.

Usage:
    python pipeline.py                         # demo mode (no API key)
    python pipeline.py --api_key sk-...        # with OpenAI GPT-4
    python pipeline.py --input data/my.csv     # custom dataset path

Outputs (saved to outputs/):
    cleaned_dataset.csv     ← fully corrected dataset
    error_report.json       ← all detected errors
    correction_log.json     ← every correction decision with scores
    flagged_for_review.csv  ← rows needing human review
    summary_report.txt      ← plain-text summary
"""

import os
import sys
import argparse
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv

# ── Load .env (for OPENAI_API_KEY) ─────────────────────────────────────
load_dotenv()

# ── Local modules ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from modules.utils         import load_dataset, save_report, timestamp, log_section, log_info, log_success, log_warn
from modules.error_detector import ErrorDetector
from modules.llm_corrector  import LLMCorrector
from modules.validator      import Validator


# ───────────────────────────────────────────────────────────────────────
# PIPELINE CLASS
# ───────────────────────────────────────────────────────────────────────

class DataCleaningPipeline:

    def __init__(self, dataset_path: str, api_key: str = None,
                 demo_mode: bool = False, output_dir: str = 'outputs'):

        self.dataset_path = dataset_path
        self.output_dir   = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        self.df_original = load_dataset(dataset_path)
        self.df_clean    = self.df_original.copy()

        # Initialise modules
        self.detector   = ErrorDetector(self.df_original)
        self.corrector  = LLMCorrector(api_key=api_key, demo_mode=demo_mode)
        self.validator  = Validator(self.df_original)

        # Logs
        self.error_report    = []
        self.correction_log  = []
        self.human_review    = []

    # ──────────────────────────────────────────────────────────────────
    # STEP 1: DETECT
    # ──────────────────────────────────────────────────────────────────

    def detect(self):
        log_section("STEP 1 — ERROR DETECTION")
        self.error_report = self.detector.run_all()
        log_info(f"Total errors detected: {len(self.error_report)}")
        return self

    # ──────────────────────────────────────────────────────────────────
    # STEP 2: CORRECT
    # ──────────────────────────────────────────────────────────────────

    def correct(self, max_corrections: int = None):
        log_section("STEP 2 — LLM CORRECTION GENERATION")

        errors_to_process = self.error_report
        if max_corrections:
            errors_to_process = errors_to_process[:max_corrections]
            log_warn(f"Processing first {max_corrections} errors (demo limit).")

        auto_applied = 0
        flagged      = 0
        human_review = 0

        for error in tqdm(errors_to_process, desc="Correcting errors", unit="err"):
            row_id = error['row_id']
            col    = error['column']

            # Skip whole-row actions for duplicates
            if col == 'ALL':
                self._handle_duplicate(row_id)
                continue

            # Get correction from LLM
            correction = self.corrector.correct(error)

            # Get original record context
            try:
                record_ctx = self.df_original.loc[row_id].to_dict()
            except KeyError:
                record_ctx = {}

            # Validate & score
            model_conf = float(correction.get('confidence', 0.5))
            val_result = self.validator.score(
                corrected_value  = correction.get('corrected_value', ''),
                column           = col,
                record           = record_ctx,
                model_confidence = model_conf
            )

            routing = val_result['routing']
            conf    = val_result['confidence_score']

            # Build log entry
            log_entry = {
                'row_id':           row_id,
                'column':           col,
                'error_type':       error['error_type'],
                'original_value':   error['observed_value'],
                'corrected_value':  correction.get('corrected_value', ''),
                'reasoning':        correction.get('reasoning', ''),
                'strategy':         correction.get('strategy_used', ''),
                'model_confidence': model_conf,
                'confidence_score': conf,
                'routing':          routing,
                'format_pass':      val_result['format_pass'],
                'logic_pass':       val_result['logic_pass'],
                'stat_pass':        val_result['stat_pass'],
            }
            self.correction_log.append(log_entry)

            # Apply correction
            if routing in ('AUTO_APPLY', 'APPLY_FLAG'):
                try:
                    self.df_clean.at[row_id, col] = correction.get('corrected_value', '')
                except Exception:
                    pass
                if routing == 'AUTO_APPLY':
                    auto_applied += 1
                else:
                    flagged += 1
            else:
                human_review += 1
                self.human_review.append(log_entry)

        log_success(f"\nCorrection routing summary:")
        log_success(f"  AUTO_APPLY   : {auto_applied}")
        log_success(f"  APPLY_FLAG   : {flagged}")
        log_success(f"  HUMAN_REVIEW : {human_review}")
        return self

    # ──────────────────────────────────────────────────────────────────
    # DUPLICATE HANDLER
    # ──────────────────────────────────────────────────────────────────

    def _handle_duplicate(self, row_id: int):
        """Mark duplicate row — remove from cleaned dataset."""
        if row_id in self.df_clean.index:
            self.df_clean = self.df_clean.drop(index=row_id)
            self.correction_log.append({
                'row_id':          row_id,
                'column':          'ALL',
                'error_type':      'DUPLICATE_RECORD',
                'original_value':  'Duplicate row',
                'corrected_value': 'ROW_REMOVED',
                'routing':         'AUTO_APPLY',
                'confidence_score': 0.99,
            })

    # ──────────────────────────────────────────────────────────────────
    # STEP 3: SAVE OUTPUTS
    # ──────────────────────────────────────────────────────────────────

    def save(self):
        log_section("STEP 3 — SAVING OUTPUTS")
        ts = timestamp()

        # Cleaned dataset
        clean_path = os.path.join(self.output_dir, f'cleaned_dataset_{ts}.csv')
        self.df_clean.to_csv(clean_path, index=False)
        log_success(f"Cleaned dataset saved → {clean_path}")

        # Error report
        save_report(self.error_report, os.path.join(self.output_dir, f'error_report_{ts}.json'))

        # Correction log
        save_report(self.correction_log, os.path.join(self.output_dir, f'correction_log_{ts}.json'))

        # Flagged for human review
        if self.human_review:
            review_df = pd.DataFrame(self.human_review)
            review_path = os.path.join(self.output_dir, f'flagged_for_review_{ts}.csv')
            review_df.to_csv(review_path, index=False)
            log_success(f"Human review cases saved → {review_path}")

        # Plain-text summary
        self._write_summary(ts)
        return self

    def _write_summary(self, ts: str):
        from collections import Counter
        error_counts   = Counter(e['error_type'] for e in self.error_report)
        routing_counts = Counter(c.get('routing','') for c in self.correction_log)

        lines = [
            "=" * 60,
            "  AI DATA CLEANING PIPELINE — SUMMARY REPORT",
            "=" * 60,
            f"  Run timestamp     : {ts}",
            f"  Dataset           : {self.dataset_path}",
            f"  Original rows     : {len(self.df_original):,}",
            f"  Cleaned rows      : {len(self.df_clean):,}",
            f"  Errors detected   : {len(self.error_report):,}",
            f"  Corrections made  : {len(self.correction_log):,}",
            "",
            "  Error Breakdown:",
        ]
        for etype, cnt in error_counts.most_common():
            lines.append(f"    {etype:<35} {cnt:>6}")

        lines += [
            "",
            "  Routing Decisions:",
        ]
        for routing, cnt in routing_counts.most_common():
            lines.append(f"    {routing:<35} {cnt:>6}")

        lines.append("=" * 60)

        summary_path = os.path.join(self.output_dir, f'summary_report_{ts}.txt')
        with open(summary_path, 'w') as f:
            f.write("\n".join(lines))

        print("\n" + "\n".join(lines))
        log_success(f"Summary saved → {summary_path}")


# ───────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='AI-Driven Data Cleaning Pipeline'
    )
    parser.add_argument('--input',   default='data/data_sales.csv',
                        help='Path to input CSV dataset')
    parser.add_argument('--api_key', default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY in .env)')
    parser.add_argument('--demo',    action='store_true', default=False,
                        help='Run in demo mode without OpenAI API key')
    parser.add_argument('--limit',   type=int, default=50,
                        help='Max corrections to process (default 50, use 0 for all)')
    parser.add_argument('--output',  default='outputs',
                        help='Output directory (default: outputs/)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Determine mode
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    demo_mode = args.demo or (api_key is None)
    if demo_mode:
        log_warn("Running in DEMO MODE (no GPT-4 API calls). "
                 "Set OPENAI_API_KEY in .env for full LLM correction.")

    limit = args.limit if args.limit > 0 else None

    # Run pipeline
    pipeline = DataCleaningPipeline(
        dataset_path = args.input,
        api_key      = api_key,
        demo_mode    = demo_mode,
        output_dir   = args.output
    )

    (pipeline
        .detect()
        .correct(max_corrections=limit)
        .save())

    log_section("PIPELINE COMPLETE")
