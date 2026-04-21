"""
carbon_utils.py
===============
Carbon metrics and climate-constrained portfolio optimisation for Part II
of the SAAM project.

Functions
---------
compute_carbon_intensity(co2_s1, co2_s2, rev, scope='1+2')
    Compute firm-level carbon intensity CI_{i,Y} (tCO2e / $M revenue).
compute_waci(weights, ci, year_Y)
    Weighted-Average Carbon Intensity of a portfolio.
compute_carbon_footprint(weights, co2_total, mv_annual, portfolio_value, year_Y)
    Carbon footprint CF (tCO2e / $M invested).
compute_vw_carbon_footprint(investment_set, co2_total, mv_annual, year_Y)
    Carbon footprint of the value-weighted benchmark.
solve_mv_carbon_constrained(Sigma, footprint_vector, cf_budget, ...)
    Minimum-variance portfolio with a carbon footprint upper bound.
solve_te_carbon_constrained(Sigma, vw_weights, footprint_vector, cf_budget, ...)
    Tracking-error minimisation with a carbon footprint upper bound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp


# ---------------------------------------------------------------------------
# Carbon intensity
# ---------------------------------------------------------------------------

def compute_carbon_intensity(
    co2_s1: pd.DataFrame,
    co2_s2: pd.DataFrame,
    rev: pd.DataFrame,
    scope: str = "1+2",
) -> pd.DataFrame:
    """Compute firm-level carbon intensity CI_{i,Y} in tCO2e per $M revenue.

    Note: CO2 data is in tonnes; revenue is in thousands of USD.
    We divide revenue by 1 000 to convert to millions before dividing.

    Parameters
    ----------
    co2_s1 : pd.DataFrame  Scope 1 CO₂ emissions (firms × years), tonnes
    co2_s2 : pd.DataFrame  Scope 2 CO₂ emissions (firms × years), tonnes
    rev    : pd.DataFrame  Revenue (firms × years), thousands USD
    scope  : str           '1', '2', or '1+2' (default '1+2')

    Returns
    -------
    pd.DataFrame  Carbon intensity matrix (firms × years), tCO2e / $M revenue
    """
    if scope == "1":
        co2_total = co2_s1
    elif scope == "2":
        co2_total = co2_s2
    elif scope == "1+2":
        co2_total = co2_s1.add(co2_s2, fill_value=0)
    else:
        raise ValueError(f"scope must be '1', '2', or '1+2', got '{scope}'")

    # Revenue in $M (convert from $k)
    rev_m = rev / 1_000

    # CI = CO2 (tonnes) / Revenue ($M)  → tCO2e / $M
    ci = co2_total.div(rev_m).replace([np.inf, -np.inf], np.nan)
    return ci


# ---------------------------------------------------------------------------
# Portfolio carbon metrics
# ---------------------------------------------------------------------------

def compute_waci(
    weights: pd.Series,
    ci: pd.DataFrame,
    year_Y: int,
) -> float:
    """Weighted-Average Carbon Intensity of a portfolio.

    WACI^(p)_Y = Σ_i α_{i,Y} × CI_{i,Y}

    Parameters
    ----------
    weights : pd.Series  (index = ISIN, values = portfolio weights)
    ci      : pd.DataFrame  carbon intensity matrix (firms × years)
    year_Y  : int

    Returns
    -------
    float  WACI in tCO2e / $M revenue
    """
    if year_Y not in ci.columns:
        return np.nan
    isins  = weights.index
    ci_vec = ci.loc[isins, year_Y].fillna(0).values
    return float(np.dot(weights.values, ci_vec))


def compute_carbon_footprint(
    weights: pd.Series,
    co2_total: pd.DataFrame,
    mv_annual: pd.DataFrame,
    portfolio_value: float,
    year_Y: int,
) -> float:
    """Carbon Footprint attributed to the investor per $M invested.

    CF^(p)_Y = (1 / V_Y) × Σ_i o_{i,Y} × E_{i,Y}

    where o_{i,Y} = (α_{i,Y} × V_Y) / Cap_{i,Y}  (ownership fraction)

    Parameters
    ----------
    weights         : pd.Series  portfolio weights (ISIN indexed)
    co2_total       : pd.DataFrame  total CO₂ emissions (firms × years), tonnes
    mv_annual       : pd.DataFrame  year-end market cap (firms × years), $M
    portfolio_value : float  total portfolio value V_Y in $M
    year_Y          : int

    Returns
    -------
    float  CF in tCO2e / $M invested
    """
    if year_Y not in co2_total.columns or year_Y not in mv_annual.columns:
        return np.nan

    isins   = weights.index
    alpha   = weights.values
    E_vec   = co2_total.loc[isins, year_Y].fillna(0).values
    cap_vec = mv_annual.loc[isins, year_Y].fillna(np.nan).values

    # Ownership fraction o_{i,Y} = (α_i × V) / Cap_i
    V_i     = alpha * portfolio_value
    o_vec   = np.where(cap_vec > 0, V_i / cap_vec, 0.0)

    cf = np.dot(o_vec, E_vec) / portfolio_value
    return float(cf)


def compute_vw_carbon_footprint(
    investment_set: list[str],
    co2_total: pd.DataFrame,
    mv_annual: pd.DataFrame,
    year_Y: int,
) -> float:
    """Carbon footprint of the value-weighted benchmark.

    CF^(vw)_Y = (1 / Cap_Y) × Σ_i E_{i,Y}

    where Cap_Y = Σ_i Cap_{i,Y}

    Parameters
    ----------
    investment_set : list of ISIN strings
    co2_total      : pd.DataFrame  total CO₂ (firms × years), tonnes
    mv_annual      : pd.DataFrame  year-end market cap (firms × years), $M
    year_Y         : int

    Returns
    -------
    float  CF in tCO2e / $M invested
    """
    if year_Y not in co2_total.columns or year_Y not in mv_annual.columns:
        return np.nan

    isins   = [i for i in investment_set if i in co2_total.index and i in mv_annual.index]
    E_sum   = co2_total.loc[isins, year_Y].fillna(0).sum()
    cap_sum = mv_annual.loc[isins, year_Y].fillna(0).sum()

    if cap_sum <= 0:
        return np.nan
    return float(E_sum / cap_sum)


# ---------------------------------------------------------------------------
# Constrained optimisation helpers
# ---------------------------------------------------------------------------

def _footprint_constraint_vector(
    isins: list[str],
    co2_total: pd.DataFrame,
    mv_annual: pd.DataFrame,
    portfolio_value: float,
    year_Y: int,
) -> np.ndarray:
    """Return vector c such that c' α = CF^(p)_Y.

    CF = (1/V) Σ_i (α_i V / Cap_i) E_i = Σ_i α_i (E_i / Cap_i)

    So c_i = E_i / Cap_i  (in tCO2e / $M invested per unit weight).
    """
    E_vec   = co2_total.loc[isins, year_Y].fillna(0).values
    cap_vec = mv_annual.loc[isins, year_Y].fillna(np.nan).values
    c_vec   = np.where(cap_vec > 0, E_vec / cap_vec, 0.0)
    return c_vec


def solve_mv_carbon_constrained(
    Sigma: np.ndarray,
    footprint_vector: np.ndarray,
    cf_budget: float,
    *,
    max_iter: int = 20_000,
    eps: float = 1e-8,
) -> tuple[np.ndarray, str]:
    """Minimum-variance portfolio with a carbon footprint upper bound.

    Problem (Section 3.2)
    ---------------------
        min  α' Σ α
        s.t. c' α ≤ cf_budget
             sum(α) = 1
             α_i ≥ 0  ∀ i

    Parameters
    ----------
    Sigma            : np.ndarray (N, N)  PSD covariance matrix
    footprint_vector : np.ndarray (N,)    vector c (E_i / Cap_i per unit weight)
    cf_budget        : float              CF upper bound (tCO2e / $M)
    max_iter         : int
    eps              : float

    Returns
    -------
    weights : np.ndarray (N,)
    status  : str
    """
    N     = Sigma.shape[0]
    alpha = cp.Variable(N)
    constraints = [
        cp.sum(alpha) == 1,
        alpha >= 0,
        footprint_vector @ alpha <= cf_budget,
    ]
    prob = cp.Problem(cp.Minimize(cp.quad_form(alpha, cp.psd_wrap(Sigma))), constraints)
    prob.solve(solver=cp.OSQP, max_iter=max_iter, eps_abs=eps, eps_rel=eps)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        prob.solve(solver=cp.SCS, max_iters=50_000)

    if alpha.value is None:
        # Constraint infeasible — fall back to unconstrained
        from portfolio_utils import solve_min_variance
        w, status = solve_min_variance(Sigma)
        return w, "infeasible_fallback"

    w = np.maximum(alpha.value, 0)
    w = w / w.sum()
    return w, prob.status


def solve_te_carbon_constrained(
    Sigma: np.ndarray,
    vw_weights: np.ndarray,
    footprint_vector: np.ndarray,
    cf_budget: float,
    *,
    max_iter: int = 20_000,
    eps: float = 1e-8,
) -> tuple[np.ndarray, str]:
    """Tracking-error minimisation with a carbon footprint upper bound.

    Problem (Sections 3.3 and 4.1)
    --------------------------------
        min  (α - α_vw)' Σ (α - α_vw)
        s.t. c' α ≤ cf_budget
             sum(α) = 1
             α_i ≥ 0  ∀ i

    Parameters
    ----------
    Sigma            : np.ndarray (N, N)  PSD covariance matrix
    vw_weights       : np.ndarray (N,)    value-weighted benchmark weights
    footprint_vector : np.ndarray (N,)    vector c (E_i / Cap_i per unit weight)
    cf_budget        : float              CF upper bound
    max_iter         : int
    eps              : float

    Returns
    -------
    weights : np.ndarray (N,)
    status  : str
    """
    N     = Sigma.shape[0]
    alpha = cp.Variable(N)
    diff  = alpha - vw_weights
    constraints = [
        cp.sum(alpha) == 1,
        alpha >= 0,
        footprint_vector @ alpha <= cf_budget,
    ]
    prob = cp.Problem(cp.Minimize(cp.quad_form(diff, cp.psd_wrap(Sigma))), constraints)
    prob.solve(solver=cp.OSQP, max_iter=max_iter, eps_abs=eps, eps_rel=eps)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        prob.solve(solver=cp.SCS, max_iters=50_000)

    if alpha.value is None:
        # Constraint infeasible
        w = vw_weights.copy()
        w = np.maximum(w, 0)
        w = w / w.sum()
        return w, "infeasible_fallback"

    w = np.maximum(alpha.value, 0)
    w = w / w.sum()
    return w, prob.status
