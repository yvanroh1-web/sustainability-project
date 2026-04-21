"""
io_utils.py
===========
Data loading and cleaning utilities for the SAAM project.

All raw data files are Datastream exports with the following layout:
  - Row 0 : Datastream error messages (skipped automatically)
  - Column 'ISIN' : firm identifier (set as index)
  - Column 'NAME' : firm name (dropped after static merge)
  - Remaining columns : date-indexed time series (monthly or annual)

Functions
---------
load_ts(filepath)
    Load a Datastream time-series Excel file.
clean_prices(ri_m, low_price_threshold=0.5)
    Apply low-price filter and intermediate forward-fill to the RI matrix.
compute_returns(ri_m)
    Compute simple monthly returns with delisting adjustment.
ffill_annual(df)
    Forward-fill annual carbon / revenue data per project rules.
filter_region(dataframes, static, regions)
    Restrict all DataFrames to firms in the specified regions.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_ts(filepath: str) -> pd.DataFrame:
    """Load a Datastream time-series Excel file.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the Excel file.

    Returns
    -------
    pd.DataFrame
        Firms as rows (ISIN index), dates as columns, values as float.
    """
    df = pd.read_excel(filepath)
    df = df.iloc[1:].reset_index(drop=True)   # drop Datastream error row
    df = df.set_index("ISIN")
    df = df.drop(columns=["NAME"], errors="ignore")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Region filtering
# ---------------------------------------------------------------------------

def filter_region(
    dataframes: dict[str, pd.DataFrame],
    static: pd.DataFrame,
    regions: list[str],
) -> dict[str, pd.DataFrame]:
    """Keep only firms belonging to the assigned regions.

    Parameters
    ----------
    dataframes : dict
        Mapping of name → DataFrame, all indexed by ISIN.
    static : pd.DataFrame
        Static firm table with columns ['ISIN', 'NAME', 'Country', 'Region'].
    regions : list of str
        Region codes to keep, e.g. ['AMER', 'EUR'].

    Returns
    -------
    dict
        Same keys, filtered DataFrames.
    """
    region_isins = set(static.loc[static["Region"].isin(regions), "ISIN"].values)
    return {
        name: df.loc[df.index.isin(region_isins)]
        for name, df in dataframes.items()
    }


# ---------------------------------------------------------------------------
# Price cleaning
# ---------------------------------------------------------------------------

def clean_prices(
    ri_m: pd.DataFrame,
    low_price_threshold: float = 0.5,
) -> pd.DataFrame:
    """Apply low-price filter and intermediate forward-fill to RI matrix.

    Steps
    -----
    1. Prices below *low_price_threshold* → NaN  (avoids extreme returns).
    2. Intermediate NaN (between first and last valid price) → forward-filled
       (misreporting / failure to report per project instructions).
    Prices at the beginning (firm not yet listed) and at the end (after
    delisting) are left as NaN.

    Parameters
    ----------
    ri_m : pd.DataFrame
        Monthly total return index, firms × months, columns sorted chronologically.
    low_price_threshold : float
        Prices strictly below this value are treated as missing (default 0.5).

    Returns
    -------
    pd.DataFrame
        Cleaned price matrix (same shape).
    """
    ri_clean = ri_m.copy()

    # Step 1 — low-price filter
    ri_clean[ri_clean < low_price_threshold] = np.nan

    # Step 2 — intermediate forward-fill
    for isin in ri_clean.index:
        p = ri_clean.loc[isin].values.astype(float)
        n = len(p)
        first_valid = next((j for j in range(n) if not np.isnan(p[j])), None)
        last_valid  = next((j for j in range(n - 1, -1, -1) if not np.isnan(p[j])), None)
        if first_valid is None or last_valid is None:
            continue
        for j in range(first_valid + 1, last_valid + 1):
            if np.isnan(p[j]):
                p[j] = p[j - 1]
        ri_clean.loc[isin] = p

    return ri_clean


# ---------------------------------------------------------------------------
# Return computation
# ---------------------------------------------------------------------------

def compute_returns(ri_m: pd.DataFrame) -> pd.DataFrame:
    """Compute simple monthly returns with delisting adjustment.

    Rules
    -----
    - NaN at t-1 (not yet listed) → return is NaN.
    - NaN at t while all subsequent prices are also NaN (delisting) →
      return = -1.0 (total loss, per project instructions).
    - Otherwise → R = P_t / P_{t-1} - 1.

    Parameters
    ----------
    ri_m : pd.DataFrame
        Cleaned monthly price matrix (firms × months), columns sorted.

    Returns
    -------
    pd.DataFrame
        Return matrix with one fewer column than ri_m.
    """
    date_cols = list(ri_m.columns)
    ret_cols  = date_cols[1:]
    returns   = pd.DataFrame(index=ri_m.index, columns=ret_cols, dtype=float)

    for isin in ri_m.index:
        p = ri_m.loc[isin].values.astype(float)
        r = np.full(len(p) - 1, np.nan)
        for t in range(1, len(p)):
            if np.isnan(p[t - 1]):
                r[t - 1] = np.nan
            elif np.isnan(p[t]):
                if all(np.isnan(p[t:])):
                    r[t - 1] = -1.0   # delisting
                else:
                    r[t - 1] = np.nan
            else:
                r[t - 1] = p[t] / p[t - 1] - 1.0
        returns.loc[isin] = r

    return returns.astype(float)


# ---------------------------------------------------------------------------
# Annual data forward-fill
# ---------------------------------------------------------------------------

def ffill_annual(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill annual carbon / revenue data per project instructions.

    Rules
    -----
    - Missing between two valid years or at the end → use previous year value.
    - Missing at the beginning → stays NaN (cannot invest in this firm yet).

    Parameters
    ----------
    df : pd.DataFrame
        Annual data matrix, firms × years.

    Returns
    -------
    pd.DataFrame
        Forward-filled copy of df.
    """
    df_filled = df.copy()
    for isin in df_filled.index:
        row = df_filled.loc[isin].values.astype(float)
        first_valid = next((j for j in range(len(row)) if not np.isnan(row[j])), None)
        if first_valid is not None:
            for j in range(first_valid + 1, len(row)):
                if np.isnan(row[j]):
                    row[j] = row[j - 1]
        df_filled.loc[isin] = row
    return df_filled
