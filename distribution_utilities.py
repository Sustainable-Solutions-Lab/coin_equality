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
    F_arr = np.asarray(F)
    scalar_input = np.ndim(F) == 0

    # Clip F to avoid numerical issues when F is very close to 1.0
    # When F = 1.0, L should equal 1.0 (everyone has accumulated all income)
    F_clipped = np.clip(F_arr, 0.0, 1.0 - EPSILON)
    exponent = 1.0 - 1.0/a
    result = 1.0 - (1.0 - F_clipped)**exponent

    return result[()] if scalar_input else result


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
# Analytical taxation and redistribution functions
#========================================================================================


def find_Fmax_analytical(
    Fmin,
    y_gross,
    gini,
    damage_yi,
    Fi_edges,
    uniform_redistribution,
    target_tax,
    tol=LOOSE_EPSILON,
):
    """
    Find Fmax in [Fmin, 1) such that progressive taxation yields target_tax.

    Uses analytical Lorenz curve integration instead of numerical quadrature,
    with stepwise interpolation for climate damage.

    Tax revenue = ∫_{Fmax}^1 [y(F) - y(Fmax)] dF
    where y(F) = y_gross * dL/dF(F) + uniform_redistribution - damage(F)

    The tax simplifies to:
    - Lorenz part: y_gross * [(1 - L(Fmax)) - (1 - Fmax) * dL/dF(Fmax)]
    - Damage part: ∫_{Fmax}^1 damage(F) dF - (1 - Fmax) * damage(Fmax)
    - Uniform redistribution cancels out

    Parameters
    ----------
    Fmin : float
        Lower boundary for income distribution (must have Fmin < Fmax).
    y_gross : float
        Gross income per capita before damage.
    gini : float
        Gini coefficient.
    damage_yi : ndarray
        Damage values at quadrature points (length N).
    Fi_edges : ndarray
        Interval boundaries for stepwise damage (length N+1).
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    target_tax : float
        Target tax amount to collect.
    tol : float, optional
        Tolerance for root finding (default LOOSE_EPSILON).

    Returns
    -------
    float
        Fmax value such that progressive taxation yields target_tax.
    """
    def tax_revenue_minus_target(Fmax):
        # Lorenz contribution from Pareto distribution
        lorenz_part = y_gross * (
            (1.0 - L_pareto(Fmax, gini)) -
            (1.0 - Fmax) * L_pareto_derivative(Fmax, gini)
        )

        # Damage contribution using stepwise functions
        damage_integral = stepwise_integrate(Fmax, 1.0, damage_yi, Fi_edges)
        damage_at_Fmax = stepwise_interpolate(Fmax, damage_yi, Fi_edges)
        damage_part = damage_integral - (1.0 - Fmax) * damage_at_Fmax

        # Tax revenue (uniform redistribution cancels out)
        tax_revenue = lorenz_part - damage_part

        return tax_revenue - target_tax

    # Bracket Fmax between Fmin and just below 1
    left = Fmin
    right = 1.0 - EPSILON

    f_left = tax_revenue_minus_target(left)
    f_right = tax_revenue_minus_target(right)

    # If both endpoints have the same sign, return the appropriate boundary
    if f_left * f_right > 0:
        if f_left > 0:
            # Both positive: target_tax is too small, return right (tax almost no one)
            return right
        else:
            # Both negative: target_tax is too large, return left (tax everyone above Fmin)
            return left

    sol = root_scalar(tax_revenue_minus_target, bracket=[left, right], method="brentq", xtol=tol)
    if not sol.converged:
        raise RuntimeError("root_scalar did not converge for find_Fmax_analytical")

    return sol.root


def find_Fmin_analytical(
    y_gross,
    gini,
    damage_yi,
    Fi_edges,
    uniform_redistribution,
    target_subsidy,
    tol=LOOSE_EPSILON,
):
    """
    Find Fmin in (0, 1) such that progressive redistribution yields target_subsidy.

    Uses analytical Lorenz curve integration instead of numerical quadrature,
    with stepwise interpolation for climate damage.

    Subsidy amount = ∫_0^Fmin [y(Fmin) - y(F)] dF
    where y(F) = y_gross * dL/dF(F) + uniform_redistribution - damage(F)

    The subsidy simplifies to:
    - Lorenz part: y_gross * [Fmin * dL/dF(Fmin) - L(Fmin)]
    - Damage part: Fmin * damage(Fmin) - ∫_0^Fmin damage(F) dF
    - Uniform redistribution cancels out

    Parameters
    ----------
    y_gross : float
        Gross income per capita before damage.
    gini : float
        Gini coefficient.
    damage_yi : ndarray
        Damage values at quadrature points (length N).
    Fi_edges : ndarray
        Interval boundaries for stepwise damage (length N+1).
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    target_subsidy : float
        Target subsidy amount to distribute.
    tol : float, optional
        Tolerance for root finding (default LOOSE_EPSILON).

    Returns
    -------
    float
        Fmin value such that progressive redistribution yields target_subsidy.
    """
    def subsidy_minus_target(Fmin):
        # Lorenz contribution from Pareto distribution
        lorenz_part = y_gross * (
            Fmin * L_pareto_derivative(Fmin, gini) - L_pareto(Fmin, gini)
        )

        # Damage contribution using stepwise functions
        damage_integral = stepwise_integrate(0.0, Fmin, damage_yi, Fi_edges)
        damage_at_Fmin = stepwise_interpolate(Fmin, damage_yi, Fi_edges)
        damage_part = Fmin * damage_at_Fmin - damage_integral

        # Subsidy amount (uniform redistribution cancels out)
        subsidy_amount = lorenz_part + damage_part

        return subsidy_amount - target_subsidy

    # Bracket Fmin between 0 and something less than 1
    left = 0.0
    right = 0.999999

    f_left = subsidy_minus_target(left)
    f_right = subsidy_minus_target(right)

    # If both endpoints have the same sign, return the appropriate boundary
    if f_left * f_right > 0:
        if f_left > 0:
            # Both positive: target_subsidy is too small, return left (redistribute to almost no one)
            return EPSILON
        else:
            # Both negative: target_subsidy is too large, return right (redistribute to almost everyone)
            return right

    sol = root_scalar(subsidy_minus_target, bracket=[left, right], method="brentq", xtol=tol)
    if not sol.converged:
        raise RuntimeError("root_scalar did not converge for find_Fmin_analytical")

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
