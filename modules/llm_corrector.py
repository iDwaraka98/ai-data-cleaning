"""
llm_corrector.py
----------------
Module 2: GPT-4 powered correction generation.

Strategies:
  - zero_shot      → Simple format fixes (dates, currency)
  - few_shot       → Patterned corrections (typos, category names)
  - chain_of_thought → Ambiguous multi-step reasoning
  - rag            → Dataset-context-dependent corrections

Usage (no API key needed for demo mode):
    corrector = LLMCorrector(api_key=None, demo_mode=True)
    result = corrector.correct(error)
"""

import os
import json
import time
from typing import Optional

from modules.utils import (
    VOCABULARY, safe_parse_json,
    log_info, log_warn, log_error
)

# ── Few-shot examples for each error type ──────────────────────────────
FEW_SHOT_EXAMPLES = {
    'TYPOGRAPHICAL_ERROR': [
        {"dirty": "Mens Street Footwear",      "clean": "Men's Street Footwear"},
        {"dirty": "Womens Apperal",             "clean": "Women's Apparel"},
        {"dirty": "Men's Atheltic Footwear",    "clean": "Men's Athletic Footwear"},
        {"dirty": "Womens Stret Footwear",      "clean": "Women's Street Footwear"},
        {"dirty": "Men's aparel",               "clean": "Men's Apparel"},
    ],
    'FORMAT_INCONSISTENCY': [
        {"dirty": "103.00",   "clean": "$103.00", "column": "Price per Unit"},
        {"dirty": "$103.0 ",  "clean": "$103.00", "column": "Price per Unit"},
        {"dirty": "2021/6/17","clean": "6/17/2021","column": "Invoice Date"},
    ],
    'MISSING_VALUE': [
        {"context": "Product=Men's Apparel, Units=200, Sales=$6000",
         "column": "Price per Unit", "clean": "$30.00"},
    ],
}

# ── System prompt shared across all strategies ─────────────────────────
SYSTEM_PROMPT = """You are a senior data quality engineer specialising in US retail sales data.
Your task is to correct data errors in a retail transactions dataset.
Always respond with ONLY a valid JSON object — no markdown, no explanation outside the JSON.
Required JSON keys: "corrected_value", "reasoning", "confidence" (float 0.0-1.0)."""


class LLMCorrector:
    """
    Wraps the OpenAI GPT-4 API for data correction.
    Set demo_mode=True to run without an API key (returns rule-based fallback).
    """

    def __init__(self, api_key: Optional[str] = None, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.client = None

        if not demo_mode:
            try:
                from openai import OpenAI
                key = api_key or os.getenv("OPENAI_API_KEY")
                if not key:
                    log_warn("No OpenAI API key found. Switching to demo mode.")
                    self.demo_mode = True
                else:
                    self.client = OpenAI(api_key=key)
                    log_info("OpenAI client initialised (GPT-4).")
            except ImportError:
                log_warn("openai package not installed. Switching to demo mode.")
                self.demo_mode = True

    # ──────────────────────────────────────────────────────────────────
    # PUBLIC: CORRECT ONE ERROR
    # ──────────────────────────────────────────────────────────────────

    def correct(self, error: dict) -> dict:
        """
        Route an error to the appropriate prompting strategy and return:
        {corrected_value, reasoning, confidence, strategy_used}
        """
        error_type = error.get('error_type', '')
        strategy   = self._select_strategy(error_type)

        if self.demo_mode:
            return self._demo_correction(error, strategy)

        prompt = self._build_prompt(error, strategy)

        try:
            response = self._call_gpt4(prompt)
            parsed   = safe_parse_json(response)

            if not parsed or 'corrected_value' not in parsed:
                log_warn(f"Could not parse GPT-4 response for row {error['row_id']}.")
                return self._fallback(error)

            parsed['strategy_used'] = strategy
            return parsed

        except Exception as e:
            log_error(f"GPT-4 call failed: {e}")
            return self._fallback(error)

    # ──────────────────────────────────────────────────────────────────
    # STRATEGY SELECTION
    # ──────────────────────────────────────────────────────────────────

    def _select_strategy(self, error_type: str) -> str:
        strategy_map = {
            'FORMAT_INCONSISTENCY': 'zero_shot',
            'DUPLICATE_RECORD':     'zero_shot',
            'TYPOGRAPHICAL_ERROR':  'few_shot',
            'MISSING_VALUE':        'chain_of_thought',
            'LOGICAL_ERROR':        'rag',
            'OUTLIER':              'chain_of_thought',
        }
        return strategy_map.get(error_type, 'zero_shot')

    # ──────────────────────────────────────────────────────────────────
    # PROMPT BUILDERS
    # ──────────────────────────────────────────────────────────────────

    def _build_prompt(self, error: dict, strategy: str) -> list:
        """Return list of messages (OpenAI chat format)."""
        user_content = self._user_content(error, strategy)
        return [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_content},
        ]

    def _user_content(self, error: dict, strategy: str) -> str:
        col   = error.get('column', '')
        val   = error.get('observed_value', '')
        etype = error.get('error_type', '')
        ctx   = error.get('context', {})

        base = (
            f"Error type: {etype}\n"
            f"Column: {col}\n"
            f"Observed value: {val}\n"
            f"Record context: {json.dumps(ctx, default=str)}\n"
        )

        if strategy == 'zero_shot':
            return (
                base +
                f"\nCorrect the {etype} in the '{col}' column.\n"
                f"Valid values for this column: {VOCABULARY.get(col, 'any appropriate value')}\n"
                f"\nRespond ONLY with JSON: {{\"corrected_value\": ..., \"reasoning\": ..., \"confidence\": ...}}"
            )

        elif strategy == 'few_shot':
            examples = FEW_SHOT_EXAMPLES.get(etype, [])
            ex_text = "\n".join(
                f"  Dirty: {e['dirty']}  →  Clean: {e['clean']}"
                for e in examples
            )
            return (
                f"Here are correction examples:\n{ex_text}\n\n"
                + base +
                f"\nNow correct the typographical error in '{col}'.\n"
                f"Valid values: {VOCABULARY.get(col, [])}\n"
                f"\nRespond ONLY with JSON: {{\"corrected_value\": ..., \"reasoning\": ..., \"confidence\": ...}}"
            )

        elif strategy == 'chain_of_thought':
            return (
                base +
                "\nThink step by step:\n"
                "1. What information is available from surrounding fields?\n"
                "2. What is the most likely correct value based on context?\n"
                "3. How confident are you in this correction?\n\n"
                "Respond ONLY with JSON: {\"corrected_value\": ..., \"reasoning\": ..., \"confidence\": ...}"
            )

        elif strategy == 'rag':
            return (
                base +
                "\nUsing the record context provided, calculate or infer the correct value.\n"
                "For Total Sales: it should equal Price per Unit × Units Sold (with any applicable discount).\n"
                "Show your calculation in the reasoning field.\n\n"
                "Respond ONLY with JSON: {\"corrected_value\": ..., \"reasoning\": ..., \"confidence\": ...}"
            )

        return base + "\nRespond ONLY with JSON: {\"corrected_value\": ..., \"reasoning\": ..., \"confidence\": ...}"

    # ──────────────────────────────────────────────────────────────────
    # GPT-4 API CALL
    # ──────────────────────────────────────────────────────────────────

    def _call_gpt4(self, messages: list, retries: int = 3) -> str:
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=500,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    log_warn(f"API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    # ──────────────────────────────────────────────────────────────────
    # DEMO MODE (no API key required)
    # ──────────────────────────────────────────────────────────────────

    def _demo_correction(self, error: dict, strategy: str) -> dict:
        """
        Rule-based fallback used when no API key is available.
        Produces deterministic corrections for demonstration purposes.
        """
        col   = error.get('column', '')
        val   = error.get('observed_value', '')
        etype = error.get('error_type', '')
        ctx   = error.get('context', {})
        sugg  = error.get('suggestion', '')

        corrected_value = sugg if sugg and sugg != 'None' else val
        reasoning = f"[DEMO MODE] Applied rule-based correction for {etype} in '{col}'."
        confidence = 0.75

        if etype == 'TYPOGRAPHICAL_ERROR' and col in VOCABULARY:
            from fuzzywuzzy import process
            match, score = process.extractOne(val, VOCABULARY[col])
            corrected_value = match
            confidence = round(score / 100, 2)
            reasoning = f"[DEMO] Fuzzy matched '{val}' → '{match}' (score={score})."

        elif etype == 'FORMAT_INCONSISTENCY' and col == 'Price per Unit':
            numeric = val.replace('$', '').replace(' ', '').strip()
            corrected_value = f"${float(numeric):.2f}" if numeric else val
            confidence = 0.95
            reasoning = "[DEMO] Normalised price format to $XX.XX."

        elif etype == 'LOGICAL_ERROR' and col == 'Total Sales':
            try:
                price = float(str(ctx.get('price', 0)).replace('$','').replace(',',''))
                units = int(ctx.get('units', 0))
                corrected_value = f"${price * units:,.0f}"
                confidence = 0.90
                reasoning = f"[DEMO] Recalculated: ${price} × {units} = {corrected_value}."
            except Exception:
                pass

        elif etype == 'MISSING_VALUE':
            confidence = 0.55
            reasoning = f"[DEMO] Placeholder correction for missing value in '{col}'."

        return {
            'corrected_value': corrected_value,
            'reasoning':       reasoning,
            'confidence':      confidence,
            'strategy_used':   strategy,
        }

    def _fallback(self, error: dict) -> dict:
        return {
            'corrected_value': error.get('observed_value', ''),
            'reasoning':       'Fallback: GPT-4 response could not be parsed.',
            'confidence':      0.30,
            'strategy_used':   'fallback',
        }
