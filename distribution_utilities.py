"""
Income distribution functions and utility integration over Pareto-Lorenz distributions.

This module provides:
- Pareto-Lorenz income distribution calculations with taxation and redistribution
- CRRA utility integration over income distributions
- Climate damage integration over income distributions
- Stepwise interpolation and integration utilities for quadrature
"""

import math
import numpy as np
from scipy.optimize import root_scalar, fsolve
from constants import EPSILON, LOOSE_EPSILON, MAX_ITERATIONS


#========================================================================================
# Pareto-Lorenz distribution functions
#========================================================================================

def a_from_G(G):
    """Pareto index a from Gini coefficient G."""
    if not (0 < G < 1):
        raise ValueError("G must be in (0,1).")
    return (1.0 + 1.0/G) / 2.0


def G_from_a(a):
    """Gini coefficient G from Pareto index a (inverse of a_from_G)."""
    if a <= 1:
        raise ValueError("a must be > 1 for finite Gini.")
    return 1.0 / (2.0 * a - 1.0)


def L_pareto(F, G):
    """Lorenz curve at F for Pareto-Lorenz distribution with Gini coefficient G."""
    a = a_from_G(G)
    return 1.0 - (1.0 - F)**(1.0 - 1.0/a)


def F_pareto(L, G):
    """Rank F where Lorenz curve equals L for Pareto-Lorenz distribution with Gini coefficient G."""
    a = a_from_G(G)
    return 1.0 - (1.0 - L)**(a / (a - 1.0))


def L_pareto_derivative(F, G):
    """Derivative of Lorenz curve dL/dF at F for Pareto-Lorenz distribution with Gini coefficient G."""
    a = a_from_G(G)
    return (1.0 - 1.0/a) * (1.0 - F)**(-1.0/a)


def crossing_rank_from_G(Gini_initial, G2):
    """Find rank where two Pareto-Lorenz curves with different Gini coefficients cross."""
    if Gini_initial == G2:
        return 0.5
    r = ((1.0 - G2) * (1.0 + Gini_initial)) / ((1.0 + G2) * (1.0 - Gini_initial))
    s = ((1.0 + Gini_initial) * (1.0 + G2)) / (2.0 * (G2 - Gini_initial))
    return 1.0 - (r ** s)


def _phi(r):
    """Helper for bracketing cap; φ(r) = (r-1) r^{1/(r-1)-1}."""
    if r <= 0:
        return float("-inf")
    if abs(r - 1.0) < EPSILON:
        return 0.0
    sgn = 1.0 if r > 1.0 else -1.0
    log_abs = math.log(abs(r - 1.0)) + (1.0/(r - 1.0) - 1.0) * math.log(r)
    return sgn * math.exp(log_abs)


#========================================================================================
# Income at rank F with damage and redistribution
#========================================================================================

# Global call counter for diagnostics
_first_call_diagnostics_printed = False
_call_counter = 0


def y_of_F_after_damage(F, Fmin, Fmax, y_mean_before_damage, damage_prev_F, uniform_redistribution, gini):
    """
    Compute income y(F) from the equation:

        y(F) = y_mean_before_damage * dL/dF(F; gini) + uniform_redistribution - damage_prev_F

    where the Lorenz curve is Pareto with Gini index gini:

        L(F) = 1 - (1-F)^(1 - 1/a)
        a    = (1 + 1/gini)/2
        dL/dF(F) = (1 - 1/a) * (1 - F)^(-1/a)

    Parameters
    ----------
    F : float or array-like
        Population rank(s) in [0,1].
    Fmin : float
        Minimum population rank for clipping F to [Fmin, Fmax].
    Fmax : float
        Maximum population rank for clipping F to [Fmin, Fmax].
    y_mean_before_damage : float
        Mean income before damage.
    damage_prev_F : float or array-like
        Per-capita damage at rank F (must be same size as F or scalar).
    uniform_redistribution : float
        Additive constant redistribution amount.
    gini : float
        Gini index (0 < gini < 1).

    Returns
    -------
    y_of_F : float or ndarray
        Income y(F) evaluated at the given F values.
    """
    global _first_call_diagnostics_printed, _call_counter

    # Increment call counter and print progress periodically
    _call_counter += 1
    if _call_counter % 100000 == 0:
        print(f"  [y_of_F_after_damage call count = {_call_counter//100000} x 100,000]")

    F = np.clip(np.asarray(F), Fmin, Fmax)
    is_scalar = F.ndim == 0
    if is_scalar:
        F = F.reshape(1)

    # Pareto-Lorenz shape parameter from Gini
    a = (1.0 + 1.0 / gini) / 2.0

    # dL/dF(F) for Pareto-Lorenz
    dLdF = (1.0 - 1.0 / a) * (1.0 - F) ** (-1.0 / a)

    # Compute income
    result = y_mean_before_damage * dLdF + uniform_redistribution - damage_prev_F

    return result


#========================================================================================
# Taxation and redistribution functions
#========================================================================================

def segment_integral_with_cut(
    Flo,
    Fhi,
    Fcut,
    Fmin,
    Fmax_for_clip,
    y_mean_before_damage,
    omega_base,
    y_damage_distribution_exponent,
    y_net_reference,
    uniform_redistribution,
    gini,
    xi,
    wi,
    cut_at,
    branch=0,
):
    """
    Compute ∫_{Flo}^{Fhi} [ y(F; Fmin, Fmax_for_clip, ...) - y(Fcut; Fmin, Fmax_for_clip, ...) ] dF

    Generic function for computing integrals over income distribution segments with a reference cut.
    Used for both taxation (upper tail) and redistribution (lower tail) calculations.

    Parameters
    ----------
    Flo : float
        Lower integration bound.
    Fhi : float
        Upper integration bound.
    Fcut : float
        Rank where reference income y(Fcut) is evaluated.
    Fmin : float
        Minimum rank for clipping in y_of_F_after_damage.
    Fmax_for_clip : float
        Maximum rank for clipping in y_of_F_after_damage.
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_exponent : float
        Damage distribution coefficient parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    cut_at : str
        Semantic label: "upper" for taxation, "lower" for redistribution.
    branch : int, optional
        Lambert W branch (default 0).

    Returns
    -------
    float
        Integral value.
    """
    # Map Gauss-Legendre nodes from [-1, 1] to [Flo, Fhi]
    F_nodes = 0.5 * (Fhi - Flo) * xi + 0.5 * (Fhi + Flo)
    w_nodes = 0.5 * (Fhi - Flo) * wi

    # y(F) over the segment, using same Fmin and Fmax_for_clip
    y_vals = y_of_F_after_damage(
        F_nodes,
        Fmin,
        Fmax_for_clip,
        y_mean_before_damage,
        omega_base,
        y_damage_distribution_exponent,
        y_net_reference,
        uniform_redistribution,
        gini,
        branch=branch,
    )

    # Reference value y(Fcut)
    y_cut = y_of_F_after_damage(
        Fcut,
        Fmin,
        Fmax_for_clip,
        y_mean_before_damage,
        omega_base,
        y_damage_distribution_exponent,
        y_net_reference,
        uniform_redistribution,
        gini,
        branch=branch,
    )

    # For redistribution (lower tail): y(Fcut) - y(F) gives positive values (poor have less income)
    # For taxation (upper tail): y(F) - y(Fcut) gives positive values (rich have more income)
    if cut_at == "lower":
        integrand = y_cut - y_vals
    else:
        integrand = y_vals - y_cut
    integral_val = np.dot(w_nodes, integrand)

    return integral_val


def total_tax_top(
    Fmax,
    Fmin,
    y_mean_before_damage,
    omega_base,
    y_damage_distribution_exponent,
    y_net_reference,
    uniform_redistribution,
    gini,
    xi,
    wi,
    target_tax=0.0,
    branch=0,
):
    """
    Compute ∫_{Fmax}^{1} [ y(F; Fmin, 1, ...) - y(Fmax; Fmin, Fmax, ...) ] dF - target_tax

    This is the function we set to zero in root finding for taxation.

    Parameters
    ----------
    Fmax : float
        Upper boundary for taxation (income ranks above Fmax are taxed).
    Fmin : float
        Lower boundary for income distribution.
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_exponent : float
        Damage distribution coefficient parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    target_tax : float, optional
        Target tax amount to subtract (default 0.0).
    branch : int, optional
        Lambert W branch (default 0).

    Returns
    -------
    float
        Integral value minus target_tax (for root finding).
    """
    integral_val = segment_integral_with_cut(
        Flo=Fmax,
        Fhi=1.0,
        Fcut=Fmax,
        Fmin=Fmin,
        Fmax_for_clip=1.0,
        y_mean_before_damage=y_mean_before_damage,
        omega_base=omega_base,
        y_damage_distribution_exponent=y_damage_distribution_exponent,
        y_net_reference=y_net_reference,
        uniform_redistribution=uniform_redistribution,
        gini=gini,
        xi=xi,
        wi=wi,
        cut_at="upper",
        branch=branch,
    )

    return integral_val - target_tax


def total_tax_bottom(
    Fmin,
    y_mean_before_damage,
    omega_base,
    y_damage_distribution_exponent,
    y_net_reference,
    uniform_redistribution,
    gini,
    xi,
    wi,
    target_subsidy=0.0,
    branch=0,
):
    """
    Compute ∫_{0}^{Fmin} [ y(Fmin; 0, Fmin, ...) - y(F; 0, Fmin, ...) ] dF - target_subsidy

    This is the function we set to zero in root finding for redistribution.

    Parameters
    ----------
    Fmin : float
        Lower boundary for redistribution (income ranks below Fmin receive redistribution).
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_exponent : float
        Damage distribution coefficient parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    target_subsidy : float, optional
        Target subsidy amount to subtract (default 0.0).
    branch : int, optional
        Lambert W branch (default 0).

    Returns
    -------
    float
        Integral value minus target_subsidy (for root finding).
    """
    integral_val = segment_integral_with_cut(
        Flo=0.0,
        Fhi=Fmin,
        Fcut=Fmin,
        Fmin=0.0,
        Fmax_for_clip=Fmin,
        y_mean_before_damage=y_mean_before_damage,
        omega_base=omega_base,
        y_damage_distribution_exponent=y_damage_distribution_exponent,
        y_net_reference=y_net_reference,
        uniform_redistribution=uniform_redistribution,
        gini=gini,
        xi=xi,
        wi=wi,
        cut_at="lower",
        branch=branch,
    )

    return integral_val - target_subsidy


def find_Fmax(
    Fmin,
    y_mean_before_damage,
    omega_base,
    y_damage_distribution_exponent,
    y_net_reference,
    uniform_redistribution,
    gini,
    xi,
    wi,
    target_tax=0.0,
    branch=0,
    tol=LOOSE_EPSILON,
):
    """
    Find Fmax in [Fmin, 1) such that total_tax_top(Fmax) = target_tax.

    Uses a bracketing root-finder.

    Parameters
    ----------
    Fmin : float
        Lower boundary for income distribution.
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_exponent : float
        Damage distribution coefficient parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    target_tax : float, optional
        Target tax amount (default 0.0).
    branch : int, optional
        Lambert W branch (default 0).
    tol : float, optional
        Tolerance for root finding (default LOOSE_EPSILON).

    Returns
    -------
    float
        Fmax value such that total_tax_top(Fmax) = target_tax.
    """
    # Define a wrapper with all parameters bound
    def f(Fmax):
        return total_tax_top(
            Fmax,
            Fmin,
            y_mean_before_damage,
            omega_base,
            y_damage_distribution_exponent,
            y_net_reference,
            uniform_redistribution,
            gini,
            xi,
            wi,
            target_tax=target_tax,
            branch=branch,
        )

    # Bracket Fmax between Fmin and just below 1 (to avoid Pareto singularity at F=1)
    left = Fmin
    right = 1.0 - EPSILON

    f_left = f(left)
    f_right = f(right)

    # If both endpoints have the same sign, return the appropriate boundary
    if f_left * f_right > 0:
        if f_left > 0:
            # Both positive: target_tax is too small, return right (tax almost no one)
            return right
        else:
            # Both negative: target_tax is too large, return left (tax everyone above Fmin)
            return left

    sol = root_scalar(f, bracket=[left, right], method="brentq", xtol=tol)
    if not sol.converged:
        raise RuntimeError("root_scalar did not converge for find_Fmax")

    return sol.root


def find_Fmin(
    y_mean_before_damage,
    omega_base,
    y_damage_distribution_exponent,
    y_net_reference,
    uniform_redistribution,
    gini,
    xi,
    wi,
    target_subsidy=0.0,
    branch=0,
    tol=LOOSE_EPSILON,
):
    """
    Find Fmin in (0, 1) such that total_tax_bottom(Fmin) = target_subsidy.

    Uses a bracketing root-finder.

    Parameters
    ----------
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_exponent : float
        Damage distribution coefficient parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    target_subsidy : float, optional
        Target subsidy amount (default 0.0).
    branch : int, optional
        Lambert W branch (default 0).
    tol : float, optional
        Tolerance for root finding (default LOOSE_EPSILON).

    Returns
    -------
    float
        Fmin value such that total_tax_bottom(Fmin) = target_subsidy.
    """
    # Define a wrapper with all parameters bound
    def f(Fmin):
        return total_tax_bottom(
            Fmin,
            y_mean_before_damage,
            omega_base,
            y_damage_distribution_exponent,
            y_net_reference,
            uniform_redistribution,
            gini,
            xi,
            wi,
            target_subsidy=target_subsidy,
            branch=branch,
        )

    # Bracket Fmin between 0 and something less than 1
    left = 0.0
    right = 0.999999

    f_left = f(left)
    f_right = f(right)

    # If both endpoints have the same sign, return the appropriate boundary
    if f_left * f_right > 0:
        if f_left > 0:
            # Both positive: target_subsidy is too small, return left (redistribute to almost no one)
            return EPSILON
        else:
            # Both negative: target_subsidy is too large, return right (redistribute to almost everyone)
            return right

    sol = root_scalar(f, bracket=[left, right], method="brentq", xtol=tol)
    if not sol.converged:
        raise RuntimeError("root_scalar did not converge for find_Fmin")

    return sol.root


#========================================================================================
# Stepwise interpolation and integration utilities
#========================================================================================

def stepwise_interpolate(F, yi, Fi_edges):
    """
    Stepwise (piecewise constant) interpolation over quadrature intervals.

    Returns yi[i] for F in [Fi_edges[i], Fi_edges[i+1]).

    Parameters
    ----------
    F : float or ndarray
        Evaluation point(s) where interpolated values are desired.
    yi : ndarray
        Values at quadrature points (length N). yi[i] is the constant value
        in interval [Fi_edges[i], Fi_edges[i+1]).
    Fi_edges : ndarray
        Interval boundaries (length N+1), must be monotonically increasing.

    Returns
    -------
    float or ndarray
        Interpolated value(s) at F. Returns same shape as F input.

    Notes
    -----
    - For F < Fi_edges[0], returns yi[0]
    - For F >= Fi_edges[-1], returns yi[-1]
    - For Fi_edges[i] <= F < Fi_edges[i+1], returns yi[i]
    """
    F_array = np.atleast_1d(F)
    scalar_input = np.ndim(F) == 0

    # Find which interval each F value falls into
    # searchsorted returns index i such that Fi_edges[i-1] <= F < Fi_edges[i]
    # We want index for yi, so subtract 1 and clip to valid range
    indices = np.searchsorted(Fi_edges, F_array, side='right') - 1
    indices = np.clip(indices, 0, len(yi) - 1)

    result = yi[indices]

    return result[0] if scalar_input else result


def stepwise_integrate(F0, F1, yi, Fi_edges):
    """
    Integrate a stepwise (piecewise constant) function from F0 to F1.

    The stepwise function has value yi[i] over interval [Fi_edges[i], Fi_edges[i+1]).

    Parameters
    ----------
    F0 : float
        Lower integration bound.
    F1 : float
        Upper integration bound.
    yi : ndarray
        Values at quadrature points (length N). yi[i] is the constant value
        in interval [Fi_edges[i], Fi_edges[i+1]).
    Fi_edges : ndarray
        Interval boundaries (length N+1), must be monotonically increasing.

    Returns
    -------
    float
        Integral of the stepwise function from F0 to F1.

    Notes
    -----
    - Handles F0 and F1 outside [Fi_edges[0], Fi_edges[-1]] by extending boundary values
    - For F0 > F1, returns negative of integral from F1 to F0
    """
    # Handle reversed bounds
    if F1 < F0:
        return -stepwise_integrate(F1, F0, yi, Fi_edges)

    # Clip bounds to valid range
    F0_clip = np.clip(F0, Fi_edges[0], Fi_edges[-1])
    F1_clip = np.clip(F1, Fi_edges[0], Fi_edges[-1])

    # Find which intervals F0 and F1 fall into
    i0 = np.searchsorted(Fi_edges, F0_clip, side='right') - 1
    i1 = np.searchsorted(Fi_edges, F1_clip, side='right') - 1

    # Clip to valid indices
    i0 = np.clip(i0, 0, len(yi) - 1)
    i1 = np.clip(i1, 0, len(yi) - 1)

    # Same interval case
    if i0 == i1:
        return yi[i0] * (F1_clip - F0_clip)

    # Multiple intervals case
    integral = 0.0

    # Partial contribution from first interval
    integral += yi[i0] * (Fi_edges[i0 + 1] - F0_clip)

    # Full contributions from middle intervals
    for k in range(i0 + 1, i1):
        integral += yi[k] * (Fi_edges[k + 1] - Fi_edges[k])

    # Partial contribution from last interval
    integral += yi[i1] * (F1_clip - Fi_edges[i1])

    return integral


def invert_step_integral(Fi_edges, y, yint):
    """
    Given a step function defined by edges Fi_edges and heights y,
    return F such that integral_0^F y(x) dx = yint.

    Parameters
    ----------
    Fi_edges : ndarray
        Interval boundaries (length N+1).
    y : ndarray
        Values at quadrature points (length N).
    yint : float
        Target integral value.

    Returns
    -------
    float
        Value F such that ∫_0^F y(x) dx = yint.
    """
    Fi_edges = np.asarray(Fi_edges)
    y = np.asarray(y)

    # Bin widths
    widths = Fi_edges[1:] - Fi_edges[:-1]

    # Contribution of each bin to the total integral
    bin_integrals = widths * y

    # Cumulative integral
    cum = np.cumsum(bin_integrals)

    # Case: yint is 0 → answer is 0
    if yint <= 0:
        return Fi_edges[0]

    # Case: yint exceeds whole integral
    if yint >= cum[-1]:
        return Fi_edges[-1]

    # Find the bin where cumulative integral first exceeds yint
    k = np.searchsorted(cum, yint)

    # Integral up to the start of bin k
    prev = cum[k-1] if k > 0 else 0.0

    # Remaining integral needed inside bin k
    rem = yint - prev

    # Linear solve inside the bin
    F = Fi_edges[k] + rem / y[k]

    return F


#========================================================================================
# CRRA utility integration
#========================================================================================

def crra_utility_interval(F0, F1, c_mean, eta):
    """
    Utility of a constant consumption level c over the interval [F0, F1].

    u(c) = c^(1-eta)/(1-eta)    if eta != 1
           ln(c)                if eta == 1

    Parameters
    ----------
    F0 : float
        Lower rank bound.
    F1 : float
        Upper rank bound.
    c_mean : float
        Constant consumption level.
    eta : float
        CRRA coefficient.

    Returns
    -------
    float
        Utility value.
    """
    width = F1 - F0
    if width < 0 or F0 < 0 or F1 > 1:
        raise ValueError("Require 0 <= F0 <= F1 <= 1.")

    if eta == 1:
        return width * np.log(c_mean)
    else:
        return width * (c_mean**(1-eta)) / (1-eta)


def crra_utility_integral_with_damage(
    F0,
    F1,
    Fmin,
    Fmax_for_clip,
    y_mean_before_damage,
    omega_base,
    y_damage_distribution_exponent,
    y_net_reference,
    uniform_redistribution,
    gini,
    eta,
    s,
    xi,
    wi,
    branch=0,
):
    """
    Numerically integrate CRRA utility over rank F in [F0, F1] accounting for climate damage.

    Uses y_of_F_after_damage() to compute income at each rank, then integrates utility
    using Gauss-Legendre quadrature.

    Parameters
    ----------
    F0 : float
        Lower rank (0 <= F0 < F1 <= 1).
    F1 : float
        Upper rank (0 <= F0 < F1 <= 1).
    Fmin : float
        Minimum rank for clipping in y_of_F_after_damage.
    Fmax_for_clip : float
        Maximum rank for clipping in y_of_F_after_damage.
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_exponent : float
        Damage distribution coefficient parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    eta : float
        CRRA coefficient (eta = 1 is log utility, eta != 1 is power utility).
    s : float
        Savings rate (0 <= s < 1). Income after savings is (1-s)*income.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    branch : int, optional
        Lambert W branch (default 0).

    Returns
    -------
    U : float
        Integral of utility over F in [F0, F1].

    Notes
    -----
    CRRA utility:
        u(c) = c^(1-eta) / (1-eta)   if eta != 1
        u(c) = log(c)                if eta == 1

    Income at each rank is computed using y_of_F_after_damage() which accounts for:
    - Pareto-Lorenz income distribution
    - Climate damage (income-dependent via power-law)
    - Redistribution
    """
    F0 = float(F0)
    F1 = float(F1)
    eta = float(eta)

    if F1 <= F0:
        raise ValueError(f"Require F0 < F1, got F0={F0}, F1={F1}")
    if F0 < 0 or F1 > 1:
        raise ValueError(f"Require 0 <= F0 < F1 <= 1, got F0={F0}, F1={F1}")

    # Map Gauss-Legendre nodes from [-1, 1] to [F0, F1]
    F_mid = 0.5 * (F1 + F0)
    F_half = 0.5 * (F1 - F0)
    F_nodes = F_half * xi + F_mid

    # Evaluate income at quadrature nodes (accounting for climate damage)
    income_before_savings = y_of_F_after_damage(
        F_nodes,
        Fmin,
        Fmax_for_clip,
        y_mean_before_damage,
        omega_base,
        y_damage_distribution_exponent,
        y_net_reference,
        uniform_redistribution,
        gini,
        branch=branch,
    )

    # Apply savings rate to get consumption
    income_vals = income_before_savings * (1.0 - s)

    # Limit income to EPSILON to prevent negative or zero values in utility calculation
    income_vals = np.maximum(income_vals, EPSILON)

    # Compute utility at each node
    if abs(eta - 1.0) < LOOSE_EPSILON:
        # Log utility
        utility_vals = np.log(income_vals)
    else:
        # Power utility
        utility_vals = (income_vals ** (1.0 - eta)) / (1.0 - eta)

    # Weighted sum scaled by interval length
    integral = F_half * np.dot(wi, utility_vals)

    return float(integral)


def climate_damage_integral(
    F0,
    F1,
    Fmin,
    Fmax_for_clip,
    y_mean_before_damage,
    omega_base,
    y_damage_distribution_exponent,
    y_net_reference,
    uniform_redistribution,
    gini,
    xi,
    wi,
    branch=0,
):
    """
    Numerically integrate climate damage over rank F in [F0, F1].

    Uses y_of_F_after_damage() to compute income at each rank, then integrates damage
    using Gauss-Legendre quadrature.

    Parameters
    ----------
    F0 : float
        Lower rank (0 <= F0 < F1 <= 1).
    F1 : float
        Upper rank (0 <= F0 < F1 <= 1).
    Fmin : float
        Minimum rank for clipping in y_of_F_after_damage.
    Fmax_for_clip : float
        Maximum rank for clipping in y_of_F_after_damage.
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_exponent : float
        Damage distribution coefficient parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    branch : int, optional
        Lambert W branch (default 0).

    Returns
    -------
    D : float
        Integral of damage over F in [F0, F1].

    Notes
    -----
    Climate damage at each rank:
        damage(F) = omega_base * (income(F) / y_net_reference)**y_damage_distribution_exponent

    Income at each rank is computed using y_of_F_after_damage() which accounts for:
    - Pareto-Lorenz income distribution
    - Climate damage (income-dependent via power-law)
    - Redistribution
    """
    F0 = float(F0)
    F1 = float(F1)

    if F1 <= F0:
        raise ValueError(f"Require F0 < F1, got F0={F0}, F1={F1}")
    if F0 < 0 or F1 > 1:
        raise ValueError(f"Require 0 <= F0 < F1 <= 1, got F0={F0}, F1={F1}")

    # Map Gauss-Legendre nodes from [-1, 1] to [F0, F1]
    F_mid = 0.5 * (F1 + F0)
    F_half = 0.5 * (F1 - F0)
    F_nodes = F_half * xi + F_mid

    # Evaluate income at quadrature nodes (accounting for climate damage)
    income_vals = y_of_F_after_damage(
        F_nodes,
        Fmin,
        Fmax_for_clip,
        y_mean_before_damage,
        omega_base,
        y_damage_distribution_exponent,
        y_net_reference,
        uniform_redistribution,
        gini,
        branch=branch,
    )

    # Compute damage at each node
    damage_vals = omega_base * (income_vals / y_net_reference)**y_damage_distribution_exponent

    # Weighted sum scaled by interval length
    integral = F_half * np.dot(wi, damage_vals)

    return float(integral)
