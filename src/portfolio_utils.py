"""
portfolio_utils.py
==================
Portfolio construction utilities for the SAAM project.

Functions
---------
get_estimation_window(year_Y, ret_dates, window=120)
    Return the list of monthly return dates in the rolling estimation window.
build_investment_set(year_Y, returns, ri_m, co2_s1, co2_s2, rev, ...)
    Determine the eligible investment set for a given year.
pairwise_covariance(R_mat)
    Estimate the covariance matrix using pairwise complete observations.
solve_min_variance(Sigma, ...)
    Solve the long-only minimum-variance optimisation.
compute_mv_weights(investment_sets, returns, ret_dates, ...)
    Roll the MV optimisation over the full sample.
compute_mv_returns(investment_sets, mv_weights, returns, ret_dates)
    Compute ex-post monthly MV portfolio returns with weight drift.
compute_vw_returns(investment_sets, returns, mv_m, ret_dates)
    Compute ex-post monthly value-weighted portfolio returns.
compute_stats(ret, rf)
    Compute annualised portfolio summary statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp


# ---------------------------------------------------------------------------
# Estimation window
# ---------------------------------------------------------------------------

def get_estimation_window(
    year_Y: int,
    ret_dates: list,
    window: int = 120,
) -> list:
    """Return up to *window* monthly return dates ending at December of year_Y.

    Parameters
    ----------
    year_Y : int
        Rebalancing year (portfolio decided at end of this year).
    ret_dates : list of pd.Timestamp
        Full sorted list of monthly return dates.
    window : int
        Number of months in the rolling estimation window (default 120 = 10 yr).

    Returns
    -------
    list of pd.Timestamp
    """
    end_date   = pd.Timestamp(year=year_Y, month=12, day=31)
    valid_dates = [d for d in ret_dates if d <= end_date]
    return valid_dates[-window:] if len(valid_dates) >= window else valid_dates


# ---------------------------------------------------------------------------
# Investment set
# ---------------------------------------------------------------------------

def build_investment_set(
    year_Y: int,
    returns: pd.DataFrame,
    ri_m: pd.DataFrame,
    co2_s1: pd.DataFrame,
    co2_s2: pd.DataFrame,
    rev: pd.DataFrame,
    ret_dates: list,
    min_obs: int = 36,
    stale_threshold: float = 0.50,
) -> list[str]:
    """Determine the eligible investment set for year *year_Y*.

    Inclusion criteria (all must hold)
    ------------------------------------
    1. Valid price (RI ≥ 0.5, not NaN) at end of year_Y.
    2. At least *min_obs* valid monthly returns in the 10-year window.
    3. Proportion of zero returns ≤ *stale_threshold* (stale-price filter).
    4. Scope 1, Scope 2, and Revenue data available for year_Y.

    Parameters
    ----------
    year_Y : int
    returns : pd.DataFrame  — monthly return matrix (firms × months)
    ri_m    : pd.DataFrame  — cleaned price matrix (firms × months)
    co2_s1  : pd.DataFrame  — forward-filled Scope 1 CO₂ (firms × years)
    co2_s2  : pd.DataFrame  — forward-filled Scope 2 CO₂ (firms × years)
    rev     : pd.DataFrame  — forward-filled Revenue (firms × years)
    ret_dates : list of pd.Timestamp
    min_obs : int           — minimum valid return observations (default 36)
    stale_threshold : float — max proportion of zero returns (default 0.50)

    Returns
    -------
    list of str  — ISIN codes of eligible firms
    """
    window     = get_estimation_window(year_Y, ret_dates)
    date_cols  = sorted([c for c in ri_m.columns if hasattr(c, "year")])
    dec_dates  = [d for d in date_cols if d.year == year_Y and d.month == 12]
    eligible   = []

    for isin in returns.index:
        # 1. Valid price at end of year_Y
        if not dec_dates:
            continue
        dec = dec_dates[0]
        if dec not in ri_m.columns or pd.isna(ri_m.loc[isin, dec]):
            continue

        # 2. Sufficient observations
        valid_rets = returns.loc[isin, window].dropna()
        if len(valid_rets) < min_obs:
            continue

        # 3. Stale-price filter
        if len(valid_rets) > 0 and (valid_rets == 0).sum() / len(valid_rets) > stale_threshold:
            continue

        # 4. Carbon data available
        def _has(df: pd.DataFrame) -> bool:
            return (
                year_Y in df.columns
                and isin in df.index
                and not pd.isna(df.loc[isin, year_Y])
            )

        if not (_has(co2_s1) and _has(co2_s2) and _has(rev)):
            continue

        eligible.append(isin)

    return eligible


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def pairwise_covariance(R_mat: np.ndarray) -> np.ndarray:
    """Estimate the covariance matrix using pairwise complete observations.

    For each pair (i, j), only months where *both* firms have non-NaN returns
    are used.  This avoids discarding the entire time series of firms with
    NaN only at the beginning (pre-listing).

    After estimation, eigenvalue clipping (floor = 1e-8) ensures the matrix
    is positive semi-definite before passing to the QP solver.

    Parameters
    ----------
    R_mat : np.ndarray  shape (N, T)
        Return matrix; NaN allowed.

    Returns
    -------
    np.ndarray  shape (N, N)
        Symmetric, PSD covariance matrix.
    """
    N = R_mat.shape[0]
    Sigma = np.zeros((N, N))

    for i in range(N):
        for j in range(i, N):
            valid = ~np.isnan(R_mat[i, :]) & ~np.isnan(R_mat[j, :])
            n_valid = valid.sum()
            if n_valid > 1:
                ri_v = R_mat[i, valid]
                rj_v = R_mat[j, valid]
                cov_ij = np.sum(
                    (ri_v - ri_v.mean()) * (rj_v - rj_v.mean())
                ) / n_valid
                Sigma[i, j] = cov_ij
                Sigma[j, i] = cov_ij

    # Eigenvalue clipping → PSD
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 1e-8)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ---------------------------------------------------------------------------
# Minimum-variance optimisation
# ---------------------------------------------------------------------------

def solve_min_variance(
    Sigma: np.ndarray,
    *,
    max_iter: int = 20_000,
    eps: float = 1e-8,
) -> tuple[np.ndarray, str]:
    """Solve the long-only minimum-variance portfolio.

    Problem
    -------
        min  α' Σ α
        s.t. sum(α) = 1
             α_i ≥ 0  ∀ i

    Falls back to SCS if OSQP does not reach optimality.

    Parameters
    ----------
    Sigma    : np.ndarray shape (N, N) — PSD covariance matrix
    max_iter : int  — max OSQP iterations
    eps      : float — convergence tolerance

    Returns
    -------
    weights : np.ndarray shape (N,) — normalised, non-negative weights
    status  : str — solver status string
    """
    N     = Sigma.shape[0]
    alpha = cp.Variable(N)
    prob  = cp.Problem(
        cp.Minimize(cp.quad_form(alpha, cp.psd_wrap(Sigma))),
        [cp.sum(alpha) == 1, alpha >= 0],
    )
    prob.solve(solver=cp.OSQP, max_iter=max_iter, eps_abs=eps, eps_rel=eps)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        prob.solve(solver=cp.SCS, max_iters=50_000)

    w = np.maximum(alpha.value, 0)
    w = w / w.sum()
    return w, prob.status


# ---------------------------------------------------------------------------
# Rolling MV optimisation
# ---------------------------------------------------------------------------

def compute_mv_weights(
    investment_sets: dict[int, list[str]],
    returns: pd.DataFrame,
    ret_dates: list,
    window: int = 120,
) -> dict[int, pd.Series]:
    """Roll the MV optimisation over the full sample (2013 → 2024).

    Parameters
    ----------
    investment_sets : dict year → list of ISIN strings
    returns         : pd.DataFrame  monthly return matrix
    ret_dates       : list of pd.Timestamp
    window          : int  estimation window in months

    Returns
    -------
    dict year → pd.Series (index = ISIN, values = weights)
    """
    mv_weights = {}

    print(
        f"{'Year':<6s} {'N':>5s} {'Active':>7s} {'Top wt':>8s} "
        f"{'σ_p (ann.)':>11s} {'Status':>12s}"
    )
    print("-" * 56)

    for Y in sorted(investment_sets.keys()):
        isins = investment_sets[Y]
        N     = len(isins)
        win   = get_estimation_window(Y, ret_dates, window)

        R_mat = returns.loc[isins, win].values   # N × T
        Sigma = pairwise_covariance(R_mat)
        w, status = solve_min_variance(Sigma)

        mv_weights[Y] = pd.Series(w, index=isins)

        n_active  = (w > 1e-6).sum()
        vol_ann   = np.sqrt(w @ Sigma @ w) * np.sqrt(12) * 100
        print(
            f"{Y:<6d} {N:>5d} {n_active:>7d} {w.max()*100:>7.2f}% "
            f"{vol_ann:>10.2f}% {status:>12s}"
        )

    return mv_weights


# ---------------------------------------------------------------------------
# Ex-post portfolio returns
# ---------------------------------------------------------------------------

def compute_mv_returns(
    investment_sets: dict[int, list[str]],
    mv_weights: dict[int, pd.Series],
    returns: pd.DataFrame,
    ret_dates: list,
) -> pd.Series:
    """Compute ex-post monthly MV portfolio returns with intra-year weight drift.

    The weight vector α is set at end of year Y and updated each month:
        α_{i,t+1} = α_{i,t} × (1 + R_{i,t+1}) / (1 + R_{p,t+1})

    Parameters
    ----------
    investment_sets : dict year → list of ISIN strings
    mv_weights      : dict year → pd.Series of weights
    returns         : pd.DataFrame  monthly return matrix
    ret_dates       : list of pd.Timestamp

    Returns
    -------
    pd.Series  monthly portfolio returns indexed by date
    """
    mv_monthly_returns, mv_monthly_dates = [], []

    for Y in sorted(investment_sets.keys()):
        isins = investment_sets[Y]
        alpha = mv_weights[Y].values.copy()
        month_dates = [d for d in ret_dates if d.year == Y + 1]

        for mdate in month_dates:
            r       = returns.loc[isins, mdate].values.astype(float)
            r_clean = np.where(np.isnan(r), 0.0, r)
            rp      = np.dot(alpha, r_clean)
            mv_monthly_returns.append(rp)
            mv_monthly_dates.append(mdate)
            alpha   = alpha * (1 + r_clean) / (1 + rp)

    return pd.Series(mv_monthly_returns, index=mv_monthly_dates, name="MV")


def compute_vw_returns(
    investment_sets: dict[int, list[str]],
    returns: pd.DataFrame,
    mv_m: pd.DataFrame,
    ret_dates: list,
) -> pd.Series:
    """Compute ex-post monthly value-weighted portfolio returns.

    Weights are recomputed each month from end-of-previous-month market caps,
    restricted to the same investment set as the MV portfolio.

    Parameters
    ----------
    investment_sets : dict year → list of ISIN strings
    returns         : pd.DataFrame  monthly return matrix
    mv_m            : pd.DataFrame  monthly market capitalisation matrix
    ret_dates       : list of pd.Timestamp

    Returns
    -------
    pd.Series  monthly portfolio returns indexed by date
    """
    mv_m_dates = sorted([c for c in mv_m.columns if hasattr(c, "year")])
    vw_monthly_returns, vw_monthly_dates = [], []

    for Y in sorted(investment_sets.keys()):
        isins       = investment_sets[Y]
        month_dates = [d for d in ret_dates if d.year == Y + 1]

        for mdate in month_dates:
            prev_idx = ret_dates.index(mdate) - 1
            if prev_idx < 0:
                continue
            prev_date = ret_dates[prev_idx]

            # Match market-cap date (within ±5 calendar days)
            mv_cands = [d for d in mv_m_dates if abs((d - prev_date).days) < 5]
            mv_date  = mv_cands[0] if mv_cands else (
                prev_date if prev_date in mv_m.columns else None
            )
            if mv_date is None or mv_date not in mv_m.columns:
                continue

            caps      = mv_m.loc[isins, mv_date].fillna(0)
            total_cap = caps.sum()
            if total_cap <= 0:
                continue

            vw_w    = (caps / total_cap).values
            r       = returns.loc[isins, mdate].values.astype(float)
            r_clean = np.where(np.isnan(r), 0.0, r)
            rp      = np.dot(vw_w, r_clean)

            vw_monthly_returns.append(rp)
            vw_monthly_dates.append(mdate)

    return pd.Series(vw_monthly_returns, index=vw_monthly_dates, name="VW")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_stats(ret: pd.Series, rf: pd.Series) -> dict:
    """Compute annualised portfolio summary statistics.

    Parameters
    ----------
    ret : pd.Series  monthly portfolio returns
    rf  : pd.Series  monthly risk-free rate (aligned to same dates)

    Returns
    -------
    dict with keys:
        'Annualized average return', 'Annualized volatility',
        'Annualized cumulative return', 'Sharpe ratio',
        'Minimum monthly return', 'Maximum monthly return'
    """
    mu  = ret.mean() * 12
    vol = ret.std()  * np.sqrt(12)
    cum = (1 + ret).prod() - 1
    n_years = len(ret) / 12
    acr     = (1 + cum) ** (1 / n_years) - 1
    excess  = ret.values - rf.values
    sr      = excess.mean() * 12 / (excess.std() * np.sqrt(12))

    return {
        "Annualized average return"    : mu,
        "Annualized volatility"        : vol,
        "Annualized cumulative return" : acr,
        "Sharpe ratio"                 : sr,
        "Minimum monthly return"       : ret.min(),
        "Maximum monthly return"       : ret.max(),
    }
