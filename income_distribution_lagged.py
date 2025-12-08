"""
Lagged damage helper functions for explicit (non-iterative) income calculations.

These functions use climate damage from the previous timestep, enabling explicit
calculation of income distributions without iterative convergence loops.
"""

import numpy as np
from scipy.optimize import root_scalar
from constants import EPSILON, LOOSE_EPSILON


def compute_redistribution_integral_lagged(
    Fmin_current,
    Fmax_current,
    y_gross_current,
    uniform_redistribution_current,
    uniform_tax_rate_current,
    gini_current,
    Omega_base_prev,
    y_damage_distribution_exponent,
    y_net_reference,
    Fmin_prev,
    Fmax_prev,
    y_gross_prev,
    uniform_redistribution_prev,
    uniform_tax_rate_prev,
    gini_prev,
    xi,
    wi,
):
    """
    Compute total redistribution to bottom segment [0, Fmin] using lagged damage.

    Returns: integral from 0 to Fmin of [y(Fmin) - y(F)] dF
    where y uses lagged damage from previous timestep.
    """
    from distribution_utilities import y_of_F_lagged_damage

    if Fmin_current < EPSILON:
        return 0.0

    # Income at Fmin using lagged damage
    y_at_Fmin = y_of_F_lagged_damage(
        Fmin_current,
        Fmin_current, Fmax_current,
        y_gross_current, uniform_redistribution_current, uniform_tax_rate_current, gini_current,
        Omega_base_prev, y_damage_distribution_exponent, y_net_reference,
        Fmin_prev, Fmax_prev, y_gross_prev,
        uniform_redistribution_prev, uniform_tax_rate_prev, gini_prev,
    )

    # Integrate y(F) from 0 to Fmin using Gauss-Legendre quadrature
    F_nodes = 0.5 * (Fmin_current * (xi + 1.0))
    y_vals = y_of_F_lagged_damage(
        F_nodes,
        Fmin_current, Fmax_current,
        y_gross_current, uniform_redistribution_current, uniform_tax_rate_current, gini_current,
        Omega_base_prev, y_damage_distribution_exponent, y_net_reference,
        Fmin_prev, Fmax_prev, y_gross_prev,
        uniform_redistribution_prev, uniform_tax_rate_prev, gini_prev,
    )
    integral_y = 0.5 * Fmin_current * np.sum(wi * y_vals)

    # Total redistribution = Fmin * y(Fmin) - integral(y(F))
    return Fmin_current * y_at_Fmin - integral_y


def compute_tax_integral_lagged(
    Fmin_current,
    Fmax_current,
    y_gross_current,
    uniform_redistribution_current,
    uniform_tax_rate_current,
    gini_current,
    Omega_base_prev,
    y_damage_distribution_exponent,
    y_net_reference,
    Fmin_prev,
    Fmax_prev,
    y_gross_prev,
    uniform_redistribution_prev,
    uniform_tax_rate_prev,
    gini_prev,
    xi,
    wi,
):
    """
    Compute total tax from top segment [Fmax, 1] using lagged damage.

    Returns: integral from Fmax to 1 of [y(F) - y(Fmax)] dF
    where y uses lagged damage from previous timestep.
    """
    from distribution_utilities import y_of_F_lagged_damage

    if Fmax_current > 1.0 - EPSILON:
        return 0.0

    # Income at Fmax using lagged damage
    y_at_Fmax = y_of_F_lagged_damage(
        Fmax_current,
        Fmin_current, Fmax_current,
        y_gross_current, uniform_redistribution_current, uniform_tax_rate_current, gini_current,
        Omega_base_prev, y_damage_distribution_exponent, y_net_reference,
        Fmin_prev, Fmax_prev, y_gross_prev,
        uniform_redistribution_prev, uniform_tax_rate_prev, gini_prev,
    )

    # Integrate y(F) from Fmax to 1 using Gauss-Legendre quadrature
    interval_width = 1.0 - Fmax_current
    F_nodes = Fmax_current + 0.5 * interval_width * (xi + 1.0)
    y_vals = y_of_F_lagged_damage(
        F_nodes,
        Fmin_current, Fmax_current,
        y_gross_current, uniform_redistribution_current, uniform_tax_rate_current, gini_current,
        Omega_base_prev, y_damage_distribution_exponent, y_net_reference,
        Fmin_prev, Fmax_prev, y_gross_prev,
        uniform_redistribution_prev, uniform_tax_rate_prev, gini_prev,
    )
    integral_y = 0.5 * interval_width * np.sum(wi * y_vals)

    # Total tax = integral(y(F)) - (1 - Fmax) * y(Fmax)
    return integral_y - interval_width * y_at_Fmax


def find_Fmin_lagged(
    Fmax_current,
    y_gross_current,
    uniform_redistribution_current,
    uniform_tax_rate_current,
    gini_current,
    Omega_base_prev,
    y_damage_distribution_exponent,
    y_net_reference,
    Fmin_prev,
    Fmax_prev,
    y_gross_prev,
    uniform_redistribution_prev,
    uniform_tax_rate_prev,
    gini_prev,
    xi,
    wi,
    target_subsidy,
    tol=LOOSE_EPSILON,
):
    """
    Find Fmin such that redistribution to [0, Fmin] equals target_subsidy using lagged damage.
    """
    def objective(Fmin_test):
        redistribution = compute_redistribution_integral_lagged(
            Fmin_test, Fmax_current,
            y_gross_current, uniform_redistribution_current, uniform_tax_rate_current, gini_current,
            Omega_base_prev, y_damage_distribution_exponent, y_net_reference,
            Fmin_prev, Fmax_prev, y_gross_prev,
            uniform_redistribution_prev, uniform_tax_rate_prev, gini_prev,
            xi, wi,
        )
        return redistribution - target_subsidy

    left = 0.0
    right = 0.999999

    f_left = objective(left)
    f_right = objective(right)

    if f_left * f_right > 0:
        if f_left > 0:
            return EPSILON
        else:
            return right

    sol = root_scalar(objective, bracket=[left, right], method="brentq", xtol=tol)
    if not sol.converged:
        raise RuntimeError("root_scalar did not converge for find_Fmin_lagged")

    return sol.root


def find_Fmax_lagged(
    Fmin_current,
    y_gross_current,
    uniform_redistribution_current,
    uniform_tax_rate_current,
    gini_current,
    Omega_base_prev,
    y_damage_distribution_exponent,
    y_net_reference,
    Fmin_prev,
    Fmax_prev,
    y_gross_prev,
    uniform_redistribution_prev,
    uniform_tax_rate_prev,
    gini_prev,
    xi,
    wi,
    target_tax,
    tol=LOOSE_EPSILON,
):
    """
    Find Fmax such that tax from [Fmax, 1] equals target_tax using lagged damage.
    """
    def objective(Fmax_test):
        tax = compute_tax_integral_lagged(
            Fmin_current, Fmax_test,
            y_gross_current, uniform_redistribution_current, uniform_tax_rate_current, gini_current,
            Omega_base_prev, y_damage_distribution_exponent, y_net_reference,
            Fmin_prev, Fmax_prev, y_gross_prev,
            uniform_redistribution_prev, uniform_tax_rate_prev, gini_prev,
            xi, wi,
        )
        return tax - target_tax

    left = Fmin_current
    right = 1.0 - EPSILON

    f_left = objective(left)
    f_right = objective(right)

    if f_left * f_right > 0:
        if f_left > 0:
            return right
        else:
            return left

    sol = root_scalar(objective, bracket=[left, right], method="brentq", xtol=tol)
    if not sol.converged:
        raise RuntimeError("root_scalar did not converge for find_Fmax_lagged")

    return sol.root


def compute_damage_integral_lagged(
    F0,
    F1,
    Fmin_current,
    Fmax_current,
    y_gross_current,
    uniform_redistribution_current,
    uniform_tax_rate_current,
    gini_current,
    Omega_base_prev,
    y_damage_distribution_exponent,
    y_net_reference,
    Fmin_prev,
    Fmax_prev,
    y_gross_prev,
    uniform_redistribution_prev,
    uniform_tax_rate_prev,
    gini_prev,
    xi,
    wi,
):
    """
    Integrate climate damage over rank F in [F0, F1] using lagged damage.

    Returns: integral from F0 to F1 of damage(F) dF
    where damage uses lagged income from previous timestep.
    """
    from distribution_utilities import y_of_F_lagged_damage

    if F1 - F0 < EPSILON:
        return 0.0

    # Map Gauss-Legendre nodes from [-1, 1] to [F0, F1]
    interval_width = F1 - F0
    F_nodes = F0 + 0.5 * interval_width * (xi + 1.0)

    # Compute income at each node using lagged damage
    y_vals = y_of_F_lagged_damage(
        F_nodes,
        Fmin_current, Fmax_current,
        y_gross_current, uniform_redistribution_current, uniform_tax_rate_current, gini_current,
        Omega_base_prev, y_damage_distribution_exponent, y_net_reference,
        Fmin_prev, Fmax_prev, y_gross_prev,
        uniform_redistribution_prev, uniform_tax_rate_prev, gini_prev,
    )

    # Compute damage at each node
    if np.abs(y_damage_distribution_exponent) < EPSILON:
        # Uniform damage
        damage_vals = np.full_like(y_vals, Omega_base_prev)
    else:
        # Income-dependent damage - but using PREVIOUS income for damage calculation
        # Note: y_vals is already the income after applying lagged damage, but for
        # computing the damage fraction, we need income before current damage
        # Actually, for aggregate damage fraction, we want: damage / y_gross_current
        # The damage is computed from previous income, which is what y_of_F_lagged_damage does

        # Reconstruct previous income to compute damage fraction
        from distribution_utilities import y_of_F_after_damage
        y_prev_vals = y_of_F_after_damage(
            F_nodes, Fmin_prev, Fmax_prev,
            y_gross_prev, Omega_base_prev, y_damage_distribution_exponent,
            y_net_reference, uniform_redistribution_prev, gini_prev,
        )
        damage_vals = Omega_base_prev * (y_prev_vals / y_net_reference) ** y_damage_distribution_exponent

    # Integrate damage using Gauss-Legendre quadrature
    integral = 0.5 * interval_width * np.sum(wi * damage_vals)
    return integral
