from typing import List, Dict, Any

import numpy as np
import pandas as pd
from prophet import Prophet

# ---------------------------------------------------------------------------
def _robust_parse_dates(series: pd.Series) -> pd.Series:
    """Try ISO first, then day-first, then month-first – raise if any NaT."""
    dates = pd.to_datetime(series, errors="coerce")             # ISO
    if dates.isna().any():
        mask = dates.isna()
        dates[mask] = pd.to_datetime(series[mask], dayfirst=True, errors="coerce")
    if dates.isna().any():
        mask = dates.isna()
        dates[mask] = pd.to_datetime(series[mask], dayfirst=False, errors="coerce")
    if dates.isna().any():
        bad = series[dates.isna()].unique()[:5]
        raise ValueError(f"Un-parsable dates, e.g. {bad}")
    return dates


def _clean_numeric(series: pd.Series) -> pd.Series:
    """
    Remove currency symbols and thousand separators, keep . and - .
    Everything else becomes NaN.
    """
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d\.\-]", "", regex=True)   # drop ₹, $, commas …
        .replace("", np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


# ---------------------------------------------------------------------------
def run_forecast(
    sales_data: List[Dict[str, Any]],
    future_days: int,
    country_holidays: str = "US",
    extra_regressors: Dict[str, List[Dict[str, Any]]] | None = None,
) -> List[Dict[str, Any]]:
    """Return a Prophet forecast (ds, yhat, yhat_lower, yhat_upper)."""

    df = pd.DataFrame(sales_data, copy=True)
    if {"ds", "y"} - set(df.columns):
        raise ValueError("sales_data must have 'ds' and 'y' keys")

    # --- PARSE & CLEAN ------------------------------------------------------
    df["ds"] = _robust_parse_dates(df["ds"])
    df["y"] = _clean_numeric(df["y"])
    df = df.dropna(subset=["ds", "y"])
    if df.empty:
        raise ValueError("No valid rows after cleaning – check your CSV.")

    # keep ≤ 1 year (enough for seasonality but avoids tiny-history blow-ups)
    last_seen = df["ds"].max()
    df = df[df["ds"] >= last_seen - pd.DateOffset(days=365)]

    # clip 1st–99th percentile – keeps extreme spikes from dominating trend
    df["y"] = df["y"].clip(*df["y"].quantile([0.01, 0.99]))

    # collapse duplicate dates
    df = df.groupby("ds", as_index=False).sum().sort_values("ds")

    # -----------------------------------------------------------------------
    m = Prophet(
        seasonality_mode="multiplicative",  # raw scale – works for most sales series
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.2,
    )
    if country_holidays:
        m.add_country_holidays(country_name=country_holidays)

    if extra_regressors:
        for name, payload in extra_regressors.items():
            _reg_df = pd.DataFrame(payload)
            _reg_df["ds"] = _robust_parse_dates(_reg_df["ds"])
            df = df.merge(_reg_df[["ds", name]], on="ds", how="left")
            m.add_regressor(name)

    print(
        f"[Φ] rows used: {len(df):,}; "
        f"window: {df['ds'].min().date()} → {df['ds'].max().date()}"
    )

    m.fit(df)

    future = m.make_future_dataframe(
        periods=future_days, freq="D", include_history=False
    )
    if extra_regressors:
        for name, payload in extra_regressors.items():
            _reg_df = pd.DataFrame(payload)
            _reg_df["ds"] = _robust_parse_dates(_reg_df["ds"])
            future = future.merge(_reg_df[["ds", name]], on="ds", how="left")

    fc = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    # safety floor (no negatives)
    fc[["yhat", "yhat_lower", "yhat_upper"]] = fc[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].clip(lower=0.0)

    return fc.to_dict(orient="records")
