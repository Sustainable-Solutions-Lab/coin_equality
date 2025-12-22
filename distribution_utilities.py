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
from scipy.optimize import root_scalar, fsolve, newton
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


def y_net_of_F(F, Fmin, Fmax, y_gross, omega_yi_calc, Fi_edges, uniform_tax_rate, uniform_redistribution, gini):
    """
    Compute net income y_net(F) at population rank F after accounting for damage, tax, and redistribution.

    Formula:
        y_net(F) = y_gross * dL/dF(F; gini) * (1.0 - omega_prev_F) * (1.0 - uniform_tax_rate) + uniform_redistribution

        Order: Lorenz → Damage → Tax → Redistribution (untaxed)

    where:
        omega_prev_F = stepwise_interpolate(F, omega_yi_calc, Fi_edges)  [damage fraction at rank F]
        dL/dF(F) = (1 - 1/a) * (1 - F)^(-1/a)  [Pareto-Lorenz derivative]
        a = (1 + 1/gini)/2  [Pareto shape parameter]

    Parameters
    ----------
    F : float or array-like
        Population rank(s) in [0,1].
    Fmin : float
        Minimum population rank for clipping F to [Fmin, Fmax].
    Fmax : float
        Maximum population rank for clipping F to [Fmin, Fmax].
    y_gross : float
        Gross income per capita before damage.
    omega_yi_calc : ndarray
        Damage fractions at quadrature points from previous timestep.
    Fi_edges : ndarray
        Interval boundaries for stepwise damage interpolation.
    uniform_tax_rate : float
        Uniform tax rate (fraction of gross income).
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini index (0 < gini < 1).

    Returns
    -------
    y_net : float or ndarray
        Net income evaluated at the given F values.
    """
    global _first_call_diagnostics_printed, _call_counter

    # Increment call counter and print progress periodically
    _call_counter += 1
    if _call_counter % 100000 == 0:
        print(f"  [y_net_of_F call count = {_call_counter//100000} x 100,000]")

    F = np.clip(np.asarray(F), Fmin, Fmax)
    is_scalar = F.ndim == 0
    if is_scalar:
        F = F.reshape(1)

    # Get damage fraction at rank F via stepwise interpolation
    omega_prev_F = stepwise_interpolate(F, omega_yi_calc, Fi_edges)

    # Pareto-Lorenz shape parameter from Gini
    a = (1.0 + 1.0 / gini) / 2.0

    # dL/dF(F) for Pareto-Lorenz
    dLdF = (1.0 - 1.0 / a) * (1.0 - F) ** (-1.0 / a)

    # Compute net income: Lorenz → Damage → Tax → Redistribution (untaxed)
    y_net_F = y_gross * dLdF * (1.0 - omega_prev_F) * (1.0 - uniform_tax_rate) + uniform_redistribution

    return y_net_F


#========================================================================================
# Analytical taxation and redistribution functions
#========================================================================================


def find_Fmax(
    Fmin,
    y_gross,
    gini,
    Omega,
    omega_yi,
    redistribution_amount,
    abateCost_amount,
    Fi_edges,
    tol=LOOSE_EPSILON,
    initial_guess=None,
):
    """
    Find Fmax in [Fmin, 1) such that progressive taxation yields target tax amount.

    Fmax is calculated based on post-damage Lorenz income only (no taxes, no redistributions).
    This ensures that Fmax defines income thresholds based on the underlying distribution,
    avoiding circular dependencies with tax and redistribution calculations.

    uniform_redistribution is always 0.0 for this calculation.

    If we are here it is because there is income-dependent tax.

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
    Omega : float
        Aggregate climate damage fraction.
    omega_yi : ndarray
        Damage values at quadrature points (length N).
    redistribution_amount : float
        Per-capita redistribution amount.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    abateCost_amount : float
        Per-capita abatement cost amount.
    Fi_edges : ndarray
        Interval boundaries for stepwise damage (length N+1).
    tol : float, optional
        Tolerance for root finding (default LOOSE_EPSILON).
    initial_guess : float, optional
        Initial guess for Fmax from previous timestep (speeds up convergence).

    Returns
    -------
    float
        Fmax value such that progressive taxation yields target tax amount.
    """

    # Always use 0.0 for uniform_redistribution when finding Fmax (see README.md Tax and Redistribution Logic)
    uniform_redistribution = 0.0

    # Pre-compute cumulative damage integrals at bin edges for fast lookup
    # damage_cumulative[i] = integral of damage from 0 to Fi_edges[i]
    # Order: Lorenz → Damage → Tax (uniform tax rate is zero in this routine)
    # damage(F) = omega(F) * y_gross * dL/dF(F)
    # Since omega is stepwise constant in each bin, integrate dL/dF over each bin
    bin_widths = np.diff(Fi_edges)
    # Integral of dL/dF from Fi_edges[i] to Fi_edges[i+1] = L(Fi_edges[i+1]) - L(Fi_edges[i])
    lorenz_diff = np.diff(L_pareto(Fi_edges, gini))
    damage_per_bin = omega_yi * y_gross * lorenz_diff
    damage_cumulative = np.concatenate(([0.0], np.cumsum(damage_per_bin)))
    total_damage_integral = damage_cumulative[-1]

    def tax_revenue_minus_target(Fmax):
        # Lorenz contribution from Pareto distribution
        lorenz_part = y_gross * (
            (1.0 - L_pareto(Fmax, gini)) -
            (1.0 - Fmax) * L_pareto_derivative(Fmax, gini)
        )

        # Fast damage calculation using pre-computed cumulative integrals
        # Find which bin Fmax is in
        bin_idx = np.searchsorted(Fi_edges, Fmax, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, len(omega_yi) - 1)

        # Damage at Fmax is constant within bin (stepwise function)
        omega_at_Fmax = omega_yi[bin_idx]

        # Integral from Fmax to 1.0 = total - integral from 0 to Fmax
        if Fmax <= Fi_edges[0]:
            damage_integral_0_to_Fmax = 0.0
        elif Fmax >= Fi_edges[-1]:
            damage_integral_0_to_Fmax = total_damage_integral
        else:
            # Cumulative up to start of bin containing Fmax
            damage_integral_0_to_Fmax = damage_cumulative[bin_idx]
            # Add partial contribution from within the bin
            partial_width = (Fmax - Fi_edges[bin_idx]) / bin_widths[bin_idx]
            damage_integral_0_to_Fmax += (damage_cumulative[bin_idx+1] - damage_cumulative[bin_idx]) * partial_width

        damage_integral = total_damage_integral - damage_integral_0_to_Fmax

        # Damage at Fmax based on income at Fmax (uniform redistribution is excluded; tax is zero here)
        damage_at_Fmax = y_gross * L_pareto_derivative(Fmax, gini) * omega_at_Fmax
        damage_part = damage_integral - (1.0 - Fmax) * damage_at_Fmax

        # Tax revenue (uniform redistribution cancels out)
        tax_revenue = lorenz_part - damage_part

        target_tax = abateCost_amount + redistribution_amount
        return tax_revenue - target_tax

    # Use secant method if we have a good initial guess, otherwise fall back to brentq
    if initial_guess is not None and Fmin < initial_guess < 1.0 - EPSILON:
        # First, check if the initial guess is already very close to the solution
        f_guess = tax_revenue_minus_target(initial_guess)
        if abs(f_guess) < tol:
            return initial_guess

        # Use secant method with initial guess
        try:
            # Secant needs two starting points; use initial_guess and a small perturbation
            x0 = initial_guess
            x1 = initial_guess + 0.001 * (1.0 - EPSILON - Fmin)  # Small step towards the middle
            x1 = np.clip(x1, Fmin + EPSILON, 1.0 - EPSILON)

            sol = root_scalar(tax_revenue_minus_target, method='secant', x0=x0, x1=x1, xtol=tol, maxiter=50)
            if sol.converged and Fmin <= sol.root <= 1.0 - EPSILON:
                return sol.root
        except (ValueError, RuntimeError):
            pass  # Fall through to bracketing method

    # Fall back to bracketing method if secant fails or no initial guess
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


def find_Fmin(
    Fmax,
    y_gross,
    gini,
    Omega,
    omega_yi,
    redistribution_amount,
    uniform_tax_rate,
    Fi_edges,
    tol=LOOSE_EPSILON,
    initial_guess=None,
):
    """
    Find Fmin in (0, Fmax) such that progressive redistribution yields target subsidy amount.

    Fmin is calculated considering:
    - Everyone pays tax on their Lorenz income (including those below Fmin)
    - Redistribution subsidy is NOT taxed (it comes from taxes)
    - Fmin defines the threshold where subsidy reaches zero

    This ensures continuity at Fmin and no perverse incentives.

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
    Fmax : float
        Upper boundary for income distribution (must have Fmin < Fmax).
    y_gross : float
        Gross income per capita before damage.
    gini : float
        Gini coefficient.
    Omega : float
        Aggregate climate damage fraction.
    omega_yi : ndarray
        Damage values at quadrature points (length N).
    redistribution_amount : float
        Per-capita redistribution amount (target subsidy).
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    uniform_tax_rate : float
        Uniform tax rate (fraction of income).
    Fi_edges : ndarray
        Interval boundaries for stepwise damage (length N+1).
    tol : float, optional
        Tolerance for root finding (default LOOSE_EPSILON).
    initial_guess : float, optional
        Initial guess for Fmin from previous timestep (speeds up convergence).

    Returns
    -------
    float
        Fmin value such that progressive redistribution yields target subsidy amount.
    """

    # Redistribution subsidy is not taxed (it comes from taxes)
    # But everyone pays tax on their Lorenz income, so uniform_tax_rate is passed as parameter
    uniform_redistribution = 0.0

    # Pre-compute cumulative damage integrals at bin edges for fast lookup
    # damage_cumulative[i] = integral of damage from 0 to Fi_edges[i]
    # damage(F) = omega(F) * y_gross * dL/dF(F) * (1 - tax)
    # Redistribution is added AFTER damage, so it's not included here
    # Since omega is stepwise constant in each bin, integrate dL/dF over each bin
    bin_widths = np.diff(Fi_edges)
    # Integral of dL/dF from Fi_edges[i] to Fi_edges[i+1] = L(Fi_edges[i+1]) - L(Fi_edges[i])
    lorenz_diff = np.diff(L_pareto(Fi_edges, gini))  # L at each edge
    damage_per_bin = omega_yi * y_gross * lorenz_diff * (1.0 - uniform_tax_rate)
    damage_cumulative = np.concatenate(([0.0], np.cumsum(damage_per_bin)))

    def subsidy_minus_target(Fmin):


        # Lorenz contribution from Pareto distribution
        # difference if everyone ewere consuming at Fmin rate minus actural integrated to Fmin.
        lorenz_part = y_gross * (1.0 - uniform_tax_rate) * (
            Fmin * L_pareto_derivative(Fmin, gini) - L_pareto(Fmin, gini)
        )

        # Fast damage calculation using pre-computed cumulative integrals
        # Find which bin Fmin is in
        bin_idx = np.searchsorted(Fi_edges, Fmin, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, len(omega_yi) - 1)

        # Omega_yi at Fmin is constant within bin (stepwise function)
        # Order: Lorenz → Damage → Tax
        damage_at_Fmin = y_gross * L_pareto_derivative(Fmin, gini) * omega_yi[bin_idx] * (1.0 - uniform_tax_rate)

        # Integral from 0 to Fmin = cumulative up to bin start + partial bin
        if Fmin <= Fi_edges[0]:
            damage_integral = 0.0
        elif Fmin >= Fi_edges[-1]:
            damage_integral = damage_cumulative[-1]
        else:
            # Cumulative up to start of bin containing Fmin
            damage_integral = damage_cumulative[bin_idx]
            # Add partial contribution from within the bin
            partial_width = (Fmin - Fi_edges[bin_idx]) / bin_widths[bin_idx]
            damage_integral += (damage_cumulative[bin_idx+1] - damage_cumulative[bin_idx]) * partial_width

        damage_part = Fmin * damage_at_Fmin - damage_integral

        # Subsidy amount (uniform redistribution cancels out)
        subsidy_amount = lorenz_part + damage_part

        return subsidy_amount - redistribution_amount

    # Use secant method if we have a good initial guess, otherwise fall back to brentq
    if initial_guess is not None and 0.0 < initial_guess < Fmax - EPSILON:
        # First, check if the initial guess is already very close to the solution
        f_guess = subsidy_minus_target(initial_guess)
        if abs(f_guess) < tol:
            return initial_guess

        # Use secant method with initial guess
        try:
            # Secant needs two starting points; use initial_guess and a small perturbation
            x0 = initial_guess
            x1 = initial_guess + 0.001   # Small step towards the middle
            x1 = np.clip(x1, EPSILON, Fmax - EPSILON)

            sol = root_scalar(subsidy_minus_target, method='secant', x0=x0, x1=x1, xtol=tol, maxiter=50)
            if sol.converged and 0.0 <= sol.root <= Fmax:
                return sol.root
        except (ValueError, RuntimeError):
            pass  # Fall through to bracketing method

    # Fall back to bracketing method if secant fails or no initial guess
    left = 0.0
    right = Fmax - EPSILON
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
