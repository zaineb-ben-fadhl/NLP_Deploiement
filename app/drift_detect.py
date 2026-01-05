import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


def _safe_numeric(arr):
    s = pd.Series(arr).replace([np.inf, -np.inf], np.nan).dropna()
    return s.values


def detect_drift(reference_file: str, production_file: str, threshold: float = 0.05):
    """
    Expected CSV columns:
      - text (required)
      - probability (optional)
    Drift features:
      - text_len_chars
      - text_len_words
      - probability (if present)
    Uses KS test for numeric distributions.
    """
    ref_df = pd.read_csv(reference_file)
    prod_df = pd.read_csv(production_file)

    if "text" not in ref_df.columns or "text" not in prod_df.columns:
        raise ValueError("Both reference and production CSV must contain a 'text' column.")

    # Build numeric features from text
    ref_df["text_len_chars"] = ref_df["text"].astype(str).apply(len)
    ref_df["text_len_words"] = ref_df["text"].astype(str).apply(lambda s: len(str(s).split()))

    prod_df["text_len_chars"] = prod_df["text"].astype(str).apply(len)
    prod_df["text_len_words"] = prod_df["text"].astype(str).apply(lambda s: len(str(s).split()))

    features = ["text_len_chars", "text_len_words"]

    # Optional: model probability if present
    if "probability" in ref_df.columns and "probability" in prod_df.columns:
        features.append("probability")

    results = {}

    for col in features:
        ref_values = _safe_numeric(ref_df[col])
        prod_values = _safe_numeric(prod_df[col])

        if len(ref_values) < 10 or len(prod_values) < 10:
            results[col] = {
                "type": "ks_test",
                "statistic": 0.0,
                "p_value": 1.0,
                "drift_detected": False,
                "note": "not_enough_samples"
            }
            continue

        stat, p_value = ks_2samp(ref_values, prod_values)
        results[col] = {
            "type": "ks_test",
            "statistic": float(stat),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < threshold),
        }

    return results
