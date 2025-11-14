import numpy as np
import pandas as pd
from typing import Optional

"""
This module implements **four heuristic-based portfolio weighting schemes**,
each accepting the same unified input and returning normalized portfolio weights.

----------------------------------------------------------------
INPUT:
    df : pandas.DataFrame
        Must contain at least:
            'price'  - current or predicted price
            'risk'   - asset volatility (σ)
        Optional additional columns:
            'mcap'        - market capitalization (for value weighting)
            'fundamental' - accounting or fundamental measure (E/P, CF/P, etc.)
    long_only : bool, default=True
        - True  → only long positions (w_i ≥ 0), weights sum to 1
        - False → long-short (w_i can be ±), weights sum to 0 and |w| sums to gross_exposure
    gross_exposure : float, default=1.0
        Total absolute exposure when long_only=False (e.g., 1 = 100/100 long-short)
----------------------------------------------------------------
METHODS:
    1. value_weighting(df, ...)
        - w_i ∝ Market Cap (MCAP_i)
        - More weight on larger firms

    2. equal_weighting(df, ...)
        - w_i = 1/N (equal allocation to all assets)

    3. risk_weighting(df, ...)
        - w_i ∝ 1 / σ_i (inverse volatility)
        - Emphasizes low-risk (stable) assets

    4. fundamental_weighting(df, ...)
        - w_i ∝ Fundamental measure (e.g., E/P, CF/P)
        - "Smart beta" style — emphasizes cheap or strong fundamentals

----------------------------------------------------------------
OUTPUT:
    DataFrame with new columns:
        'weight'        - normalized portfolio weight
----------------------------------------------------------------
"""


def _normalize_weights(raw: np.ndarray, long_only: bool = True, gross_exposure: float = 1.0) -> np.ndarray:
    """
    Normalize raw weight/signal vector into either:
      - long-only non-negative weights summing to 1 (if long_only=True)
      - long-short weights summing to 0 and abs-sum = gross_exposure (if long_only=False)
    raw: numpy array of floats (can be positive/negative)
    """
    w = np.array(raw, dtype=float)
    n = len(w)
    if long_only:
        # Force non-negative and scale to sum 1
        w = np.where(np.isfinite(w), w, 0.0)   # replace inf/nan -> 0
        w[w < 0] = 0.0
        s = w.sum()
        if s == 0 or np.isclose(s, 0.0):
            # fallback to equal weights if everything zero
            return np.ones(n) / n
        return w / s
    else:
        # center to zero mean
        w = np.where(np.isfinite(w), w, 0.0)
        w_centered = w - np.mean(w)
        abs_sum = np.sum(np.abs(w_centered))
        if np.isclose(abs_sum, 0.0):
            # fallback: equal long-short split
            pattern = np.concatenate([np.ones(n // 2), -np.ones(n - n // 2)])
            return (pattern / np.sum(np.abs(pattern))) * gross_exposure
        return (w_centered / abs_sum) * gross_exposure


# Unified signature: df must contain 'symbol' (optional), 'price', 'risk'; optional 'mcap', 'fundamental', 'signal'
def value_weighting(
    df: pd.DataFrame,
    price_col: str = "price",
    risk_col: str = "risk",
    mcap_col: str = "mcap",
    long_only: bool = True,
    gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """
    Market-cap weighting: uses 'mcap' column. If mcap not present, raises ValueError.
    Returns DataFrame with 'weight'
    """
    df = pd.DataFrame(df).copy()
    if mcap_col not in df.columns:
        raise ValueError(f"value_weighting requires column '{mcap_col}' in dataframe")
    raw = pd.to_numeric(df[mcap_col], errors="coerce").fillna(0).values
    weights = _normalize_weights(raw, long_only=long_only, gross_exposure=gross_exposure)
    df["weight"] = weights
    return df


def equal_weighting(
    df: pd.DataFrame,
    price_col: str = "price",
    risk_col: str = "risk",
    mcap_col: str = "mcap",            # unused but kept for identical signature
    long_only: bool = True,
    gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """
    Equal weighting. If long_only=True returns all weights = 1/N.
    If long_only=False returns centered long-short equal-magnitude weights (gross exposure scaled).
    """
    df = pd.DataFrame(df).copy()
    n = len(df)
    if n == 0:
        return df
    if long_only:
        weights = np.ones(n) / n
    else:
        # initial pattern: +1 for first half, -1 for second half (user may want to pre-sort df by signal)
        pattern = np.concatenate([np.ones(n // 2), -np.ones(n - n // 2)])
        weights = _normalize_weights(pattern, long_only=False, gross_exposure=gross_exposure)
    df["weight"] = weights
    return df


def risk_weighting(
    df: pd.DataFrame,
    price_col: str = "price",
    risk_col: str = "risk",
    mcap_col: str = "mcap",
    long_only: bool = True,
    gross_exposure: float = 1.0,
    min_risk: float = 1e-8,
) -> pd.DataFrame:
    """
    Risk weighting (inverse volatility). Uses the 'risk_col' (std dev). 
    Invalid/zero risk -> treated as very large (zero weight).
    """
    df = pd.DataFrame(df).copy()
    if risk_col not in df.columns:
        raise ValueError(f"risk_weighting requires column '{risk_col}' in dataframe")
    risk = pd.to_numeric(df[risk_col], errors="coerce").values.astype(float)
    # treat non-positive or nan risk as NaN => inv = 0 for that row
    safe = np.where(risk > 0, risk, np.nan)
    inv = np.divide(1.0, safe, out=np.zeros_like(safe, dtype=float), where=~np.isnan(safe))
    weights = _normalize_weights(inv, long_only=long_only, gross_exposure=gross_exposure)
    df["weight"] = weights
    return df


def fundamental_weighting(
    df: pd.DataFrame,
    price_col: str = "price",
    risk_col: str = "risk",
    mcap_col: str = "mcap",
    funda_col: str = "fundamental",
    long_only: bool = True,
    gross_exposure: float = 1.0,
    allow_negative: bool = False,
) -> pd.DataFrame:
    """
    Fundamental weighting: uses 'funda_col'. If allow_negative=False and long_only=True, negative fundamentals clipped to 0.
    """
    df = pd.DataFrame(df).copy()
    if funda_col not in df.columns:
        raise ValueError(f"fundamental_weighting requires column '{funda_col}' in dataframe")
    raw = pd.to_numeric(df[funda_col], errors="coerce").fillna(0).astype(float)
    if long_only and not allow_negative:
        raw = np.where(raw < 0, 0.0, raw)
    weights = _normalize_weights(raw, long_only=long_only, gross_exposure=gross_exposure)
    df["weight"] = weights
    return df
