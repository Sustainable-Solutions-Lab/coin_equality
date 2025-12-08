"""
Tax and redistribution policy functions.

This module implements different tax and redistribution policies for the
climate-economic model. Policies determine how costs are distributed across
the income distribution and how revenues are redistributed.

Tax Policies
------------
- uniform_fractional: Everyone pays the same fraction of income
- tax_richest: Tax only income above a threshold (progressive)
- uniform_utility_reduction: Tax so everyone experiences equal utility loss

Redistribution Policies
-----------------------
- uniform_dividend: Equal per-capita payment to everyone
- targeted_lowest_income: Benefits go only to lowest income individuals
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import bisect
from distribution_utilities import L_pareto_derivative


def y_post_damage(F, y_mean, G, damage_at_rank_fn):
    """
    Post-damage income at population rank F.

    y(F) = y_mean * dL/dF(F) - damage(F)

    Parameters
    ----------
    F : float
        Population rank in [0,1], poorest to richest.
    y_mean : float
        Mean income.
    G : float
        Gini index.
    damage_at_rank_fn : callable
        Function damage(F) returning damage at rank F.

    Returns
    -------
    float
        Income at rank F after damage.
    """
    return y_mean * L_pareto_derivative(F, G) - damage_at_rank_fn(F)


def _phi_tax_richest(Fc, y_mean, G, fract_gdp, damage_at_rank_fn):
    """
    Objective function for finding critical rank Fcrit.

    Phi(Fc) = integral from Fc to 1 of [y(F) - y(Fc)] dF - fract_gdp * y_mean

    We want Phi(Fc) = 0.

    Parameters
    ----------
    Fc : float
        Candidate cutoff rank in [0,1].
    y_mean : float
        Mean income.
    G : float
        Gini index.
    fract_gdp : float
        Target fraction of total GDP from excess income above y(Fc).
    damage_at_rank_fn : callable
        Function damage(F) returning damage at rank F.

    Returns
    -------
    float
        Value of Phi at Fc.
    """
    y_c = y_post_damage(Fc, y_mean, G, damage_at_rank_fn)
    integrand = lambda F: y_post_damage(F, y_mean, G, damage_at_rank_fn) - y_c
    integral_value, _ = quad(integrand, Fc, 1.0, epsabs=1e-9, epsrel=1e-9)
    return integral_value - fract_gdp * y_mean


def find_Fcrit_tax_richest(y_mean, G, fract_gdp, damage_at_rank_fn):
    """
    Find critical rank Fcrit for tax_richest policy.

    Finds Fcrit in [0,1) such that:
        integral from Fcrit to 1 of (y(F) - y(Fcrit)) dF = fract_gdp * y_mean

    Parameters
    ----------
    y_mean : float
        Mean income.
    G : float
        Gini index.
    fract_gdp : float
        Target fraction of GDP from excess income above y(Fcrit).
    damage_at_rank_fn : callable
        Function damage(F) returning damage at rank F.

    Returns
    -------
    float
        Critical rank Fcrit in [0,1).
    """
    eps = 1e-8
    a, b = 0.0, 1.0 - eps

    fa = _phi_tax_richest(a, y_mean, G, fract_gdp, damage_at_rank_fn)
    fb = _phi_tax_richest(b, y_mean, G, fract_gdp, damage_at_rank_fn)

    if fa * fb > 0:
        raise RuntimeError(
            f"No sign change in Phi between 0 and 1-eps; root not bracketed. "
            f"phi(0)={fa}, phi(1-eps)={fb}"
        )

    Fcrit = bisect(
        _phi_tax_richest,
        a,
        b,
        args=(y_mean, G, fract_gdp, damage_at_rank_fn),
        xtol=1e-9,
        rtol=1e-9,
        maxiter=100,
    )
    return Fcrit


def make_tax_function_richest(Fcrit, y_mean, G, damage_at_rank_fn):
    """
    Construct tax function for tax_richest policy.

    Returns a function tax(F) such that:
        tax(F) = 0                for F < Fcrit
        tax(F) = y(F) - y(Fcrit)  for F >= Fcrit

    This shaves off income from everyone above Fcrit down to the income at Fcrit.

    Parameters
    ----------
    Fcrit : float
        Cutoff rank, 0 <= Fcrit < 1.
    y_mean : float
        Mean income.
    G : float
        Gini index.
    damage_at_rank_fn : callable
        Function damage(F) returning damage at rank F.

    Returns
    -------
    callable
        tax(F): takes float or np.ndarray F and returns tax at that rank.
    """
    y_crit = y_post_damage(Fcrit, y_mean, G, damage_at_rank_fn)

    def tax(F):
        """Per-capita tax at rank F."""
        F_arr = np.asarray(F)
        yF = y_post_damage(F_arr, y_mean, G, damage_at_rank_fn)
        t = np.maximum(0.0, yF - y_crit)
        if np.isscalar(F):
            return float(t)
        return t

    return tax


def calculate_tax_richest(y_mean, G, fract_gdp, damage_at_rank_fn):
    """
    Main entry point for tax_richest policy.

    Computes the critical rank and returns the tax function.

    Parameters
    ----------
    y_mean : float
        Mean income.
    G : float
        Gini index.
    fract_gdp : float
        Target fraction of GDP to collect as tax.
    damage_at_rank_fn : callable
        Function damage(F) returning damage at rank F.

    Returns
    -------
    tax_fn : callable
        Function tax(F) returning tax at rank F.
    Fcrit : float
        Critical rank above which tax is applied.
    """
    Fcrit = find_Fcrit_tax_richest(y_mean, G, fract_gdp, damage_at_rank_fn)
    tax_fn = make_tax_function_richest(Fcrit, y_mean, G, damage_at_rank_fn)
    return tax_fn, Fcrit


# =============================================================================
# Uniform fractional tax
# =============================================================================

def calculate_tax_uniform_fractional(y_mean, G, fract_gdp, mean_damage, damage_dist_fn):
    """
    Uniform fractional tax: everyone pays the same fraction of post-damage income.

    The tax at rank F is:
        tax(F) = fract_gdp * y_mean * dL/dF(F) - fract_gdp * mean_damage * damage_dist_fn(F)

    This applies a uniform fractional tax to income, adjusted for damage at each rank.

    Parameters
    ----------
    y_mean : float
        Mean income.
    G : float
        Gini index.
    fract_gdp : float
        Fraction of GDP to collect as tax.
    mean_damage : float
        Mean aggregate damage across the population.
    damage_dist_fn : callable
        Function damage_dist_fn(F) returning relative damage factor at rank F.

    Returns
    -------
    tax_fn : callable
        Function tax(F) returning tax at rank F.
    """
    def tax(F):
        """Per-capita tax at rank F (uniform fraction of post-damage income)."""
        F_arr = np.asarray(F)
        # Income at rank F (before damage)
        Gini_index_part = fract_gdp * y_mean * ((1 - G) / (1 + G)) * (1 - F_arr) ** (-2 * G / (1 + G))
        # Damage adjustment
        damage_part = fract_gdp * mean_damage * damage_dist_fn(F_arr)
        t = Gini_index_part - damage_part
        if np.isscalar(F):
            return float(t)
        return t

    return tax


# =============================================================================
# Redistribution policies (placeholders for future implementation)
# =============================================================================

def calculate_redistribution_uniform_dividend(total_revenue, L):
    """
    Uniform dividend: equal per-capita payment to everyone.

    Parameters
    ----------
    total_revenue : float
        Total revenue to redistribute.
    L : float
        Population.

    Returns
    -------
    dividend_per_capita : float
        Per-capita dividend amount.
    """
    return total_revenue / L


def calculate_redistribution_targeted_lowest(total_revenue, L, G, target_fraction):
    """
    Targeted redistribution: benefits go only to lowest income individuals.

    Parameters
    ----------
    total_revenue : float
        Total revenue to redistribute.
    L : float
        Population.
    G : float
        Gini index.
    target_fraction : float
        Fraction of population (lowest income) to receive benefits.

    Returns
    -------
    dividend_per_capita : float
        Per-capita dividend for targeted recipients.
    Fcrit : float
        Rank below which people receive benefits.
    """
    Fcrit = target_fraction
    recipients = L * target_fraction
    dividend_per_capita = total_revenue / recipients
    return dividend_per_capita, Fcrit
