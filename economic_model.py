"""
Functions for calculating economic production, climate impacts, and system tendencies.

This module implements the Solow-Swann growth model with climate damage
and emissions abatement costs.
"""

import numpy as np
import time
from scipy.special import roots_legendre
from distribution_utilities import (
    y_net_of_F,
    find_Fmax,
    find_Fmin,
    L_pareto,
    L_pareto_derivative,
    stepwise_interpolate,
    stepwise_integrate
)
from parameters import evaluate_params_at_time
from constants import EPSILON, LOOSE_EPSILON, NEG_BIGNUM

def gini_from_distribution(values_yi, Fi_edges, Fwi):
    """
    Calculate Gini coefficient from discretized distribution.

    Parameters
    ----------
    values_yi : np.ndarray
        Values at quadrature points (length N_QUAD)
    Fi_edges : np.ndarray
        Edges of bins in F space [0, 1] (length N_QUAD + 1)
    Fwi : np.ndarray
        Bin widths (length N_QUAD)

    Returns
    -------
    float
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)

    Notes
    -----
    Gini = 2 * integral from 0 to 1 of (F - L(F)) dF
    where L(F) is the Lorenz curve (cumulative fraction of total quantity
    held by bottom F fraction of population)
    """
    total = np.sum(Fwi * values_yi)

    if np.abs(total) <= EPSILON:
        return 0.0

    # Calculate Lorenz curve at each edge
    lorenz = np.zeros(len(Fi_edges))
    cumulative = 0.0

    for i in range(len(Fwi)):
        cumulative += Fwi[i] * values_yi[i]
        lorenz[i+1] = cumulative / total

    # Calculate Gini coefficient using trapezoidal rule
    # Gini = 2 * integral from 0 to 1 of (F - L(F)) dF
    gini = 0.0
    for i in range(len(Fi_edges) - 1):
        dF = Fi_edges[i+1] - Fi_edges[i]
        F_avg = (Fi_edges[i] + Fi_edges[i+1]) / 2.0
        L_avg = (lorenz[i] + lorenz[i+1]) / 2.0
        gini += (F_avg - L_avg) * dF

    gini *= 2.0

    return gini

# Global timing statistics
_timing_stats = {
    'call_count': 0,
    'total_time': 0.0,
    'setup_time': 0.0,
    'find_Fmax_time': 0.0,
    'find_Fmin_time': 0.0,
    'segment1_time': 0.0,
    'segment2_time': 0.0,
    'segment3_time': 0.0,
    'utility_time': 0.0,
    'finalize_time': 0.0,
}

def print_timing_stats():
    """Print timing statistics for calculate_tendencies."""
    stats = _timing_stats
    if stats['call_count'] == 0:
        return

    print(f"\n{'='*80}")
    print(f"TIMING STATISTICS (after {stats['call_count']} calls)")
    print(f"{'='*80}")
    print(f"Total time:          {stats['total_time']:8.2f} s  (100.0%)")
    print(f"  Setup:             {stats['setup_time']:8.2f} s  ({100*stats['setup_time']/stats['total_time']:5.1f}%)")
    print(f"  find_Fmax:         {stats['find_Fmax_time']:8.2f} s  ({100*stats['find_Fmax_time']/stats['total_time']:5.1f}%)")
    print(f"  find_Fmin:         {stats['find_Fmin_time']:8.2f} s  ({100*stats['find_Fmin_time']/stats['total_time']:5.1f}%)")
    print(f"  Segment 1:         {stats['segment1_time']:8.2f} s  ({100*stats['segment1_time']/stats['total_time']:5.1f}%)")
    print(f"  Segment 2:         {stats['segment2_time']:8.2f} s  ({100*stats['segment2_time']/stats['total_time']:5.1f}%)")
    print(f"  Segment 3:         {stats['segment3_time']:8.2f} s  ({100*stats['segment3_time']/stats['total_time']:5.1f}%)")
    print(f"  Utility calc:      {stats['utility_time']:8.2f} s  ({100*stats['utility_time']/stats['total_time']:5.1f}%)")
    print(f"  Finalize:          {stats['finalize_time']:8.2f} s  ({100*stats['finalize_time']/stats['total_time']:5.1f}%)")
    print(f"Avg time per call:   {stats['total_time']/stats['call_count']*1000:8.3f} ms")
    print(f"{'='*80}\n")


def calculate_tendencies(state, params, omega_yi_Omega_base_ratio_prev, Omega_Omega_base_ratio_prev, xi, xi_edges, wi, Fmax_prev=None, Fmin_prev=None, store_detailed_output=True):
    """
    Calculate time derivatives and all derived variables.

    Parameters
    ----------
    state : dict
        State variables:
        - 'K': Capital stock ($)
        - 'Ecum': Cumulative CO2 emissions (tCO2)
    params : dict
        Model parameters (all must be provided):
        - 'alpha': Output elasticity of capital
        - 'delta': Capital depreciation rate (yr^-1)
        - 's': Savings rate
        - 'psi1': Linear climate damage coefficient (°C⁻¹) [Barrage & Nordhaus 2023]
        - 'psi2': Quadratic climate damage coefficient (°C⁻²) [Barrage & Nordhaus 2023]
        - 'y_damage_distribution_exponent': Exponent for income-dependent damage distribution
        - 'y_net_reference': Reference income for power-law damage scaling ($/person)
        - 'k_climate': Temperature sensitivity (°C tCO2^-1)
        - 'eta': Coefficient of relative risk aversion
        - 'A': Total factor productivity (current)
        - 'L': Population (current)
        - 'sigma': Carbon intensity of GDP (current, tCO2 $^-1)
        - 'theta1': Abatement cost coefficient (current, $ tCO2^-1)
        - 'theta2': Abatement cost exponent
        - 'mu_max': Maximum allowed abatement fraction (cap on μ)
        - 'gini': Background Gini index (current, from time function)
        - 'Gini_fract': Fraction of Gini change as instantaneous step
        - 'Gini_restore': Rate of restoration to gini (yr^-1)
        - 'fract_gdp': Fraction of GDP available for redistribution and abatement
        - 'f': Fraction allocated to abatement vs redistribution
    omega_yi_Omega_base_ratio_prev : np.ndarray
        Ratio of climate damage fractions to base damage from previous timestep (length N_QUAD).
        Units: dimensionless. Multiply by current Omega_base to get current damage fractions.
        Used to compute current income distribution with lagged damage to avoid circular dependency.
    Omega_Omega_base_ratio_prev : float
        Ratio of aggregate damage to base damage from previous timestep.
        Multiply by current Omega_base to get current aggregate damage fraction.
    xi : np.ndarray
        Gauss-Legendre quadrature nodes on [-1, 1] (length N_QUAD)
    xi_edges : np.ndarray
        Edges of quadrature intervals on [-1, 1] (length N_QUAD + 1)
    wi : np.ndarray
        Gauss-Legendre quadrature weights (length N_QUAD)
    Fmax_prev : float, optional
        Previous timestep's Fmax value (used as initial guess for root finding, speeds up convergence)
    Fmin_prev : float, optional
        Previous timestep's Fmin value (used as initial guess for root finding, speeds up convergence)
    store_detailed_output : bool, optional
        Whether to compute and return all intermediate variables. Default: True

    Returns
    -------
    dict
        Dictionary containing:
        - Tendencies: 'dK_dt', 'dEcum_dt'
        - Climate damage ratios for next timestep: 'omega_yi_Omega_base_ratio', 'Omega_Omega_base_ratio'
        - All intermediate variables: Y_gross, delta_T, Omega, Y_net, y_net, redistribution,
          mu, Lambda, AbateCost, U, E

    Notes
    -----
    Calculation order follows equations 1.1-1.10, 2.1-2.2, 3.5, 4.3-4.4:
    1. Y_gross from K, L, A, α (Eq 1.1)
    2. ΔT from Ecum, k_climate (Eq 2.2)
    3. y_gross from Y_gross, L (mean per-capita gross income)
    4. Ω, G_climate from ΔT, Gini, y_gross, damage params (income-dependent damage)
    5. Y_damaged from Y_gross, Ω (Eq 1.3)
    6. y from Y_damaged, L, s (Eq 1.4)
    7. Δc from y, ΔL (Eq 4.3)
    8. E_pot from σ, Y_gross (Eq 2.1)
    9. AbateCost from f, Δc, L (Eq 1.5)
    10. μ from AbateCost, θ₁, θ₂, E_pot (Eq 1.6)
    11. Λ from AbateCost, Y_damaged (Eq 1.7)
    12. Y_net from Y_damaged, Λ (Eq 1.8)
    13. y_net from y, AbateCost, L (Eq 1.9)
    14. U from y_net, Gini, η (Eq 3.5)
    16. E from σ, μ, Y_gross (Eq 2.3)
    17. dK/dt from s, Y_net, δ, K (Eq 1.10)
    """
    t_start = time.time()
    _timing_stats['call_count'] += 1

    # Extract state variables
    K = state['K']
    Ecum = state['Ecum']

    # Extract parameters
    alpha = params['alpha']
    delta = params['delta']
    s = params['s']
    k_climate = params['k_climate']
    eta = params['eta']
    rho = params['rho']
    t = params['t']
    A = params['A'] # total factor productivity
    L = params['L'] # population
    sigma = params['sigma']
    theta1 = params['theta1']
    theta2 = params['theta2']
    mu_max = params['mu_max']
    fract_gdp = params['fract_gdp']
    gini = params['gini']
    f = params['f']
    y_damage_distribution_exponent = params['y_damage_distribution_exponent']
    y_net_reference = params['y_net_reference']
    psi1 = params['psi1']
    psi2 = params['psi2']

    # Policy switches
    income_dependent_aggregate_damage = params['income_dependent_aggregate_damage']
    income_dependent_damage_distribution = params['income_dependent_damage_distribution']
    income_dependent_tax_policy = params['income_dependent_tax_policy']
    income_redistribution = params['income_redistribution']
    income_dependent_redistribution_policy = params['income_dependent_redistribution_policy']



    # Transform xi into F space. Map [-1,1] to [0,1]
    Fi = (xi + 1.0)/2.0
    # compute edges in F space
    Fi_edges = (xi_edges + 1.0)/2.0
    # Transform quadrature weights to F space [0,1] (wi is for xi space [-1,1])
    Fwi = wi / 2.0

    #========================================================================================
    # Simplified damage calculation using aggregate damage from previous timestep
    # No iteration needed - uses temperature-based Omega_base with previous damage for budgeting

    # Initialize variables
    uniform_redistribution_amount = 0.0
    uniform_tax_rate = 0.0
    Fmin = 0.0
    Fmax = 1.0
    min_y_net = 0.0
    max_y_net = 0.0

    #========================================================================================
    # Main calculations

    # Eq 2.2: Temperature change from cumulative emissions
    delta_T = k_climate * Ecum

    # Base damage from temperature (capped just below 1.0 to avoid division by zero)
    # Be careful when used not to produce effective Omega values >= 1.0
    Omega_base = psi1 * delta_T + psi2 * (delta_T ** 2)

    if income_dependent_damage_distribution:
        # Reconstruct damage fractions from ratios stored at previous timestep
        # Multiply stored ratios by current Omega_base to get current damage estimates
        Omega_calc = Omega_Omega_base_ratio_prev * Omega_base
        omega_yi_calc = omega_yi_Omega_base_ratio_prev * Omega_base
        # Clip scaled values to ensure they remain valid damage fractions
        Omega_calc = np.clip(Omega_calc, 0.0, 1.0 - EPSILON)
        omega_yi_calc = np.clip(omega_yi_calc, 0.0, 1.0 - EPSILON)
    else:
        # No income-dependent damage distribution, use aggregate damage only
        Omega_calc = Omega_base
        omega_yi_calc = np.full_like(omega_yi_Omega_base_ratio_prev, Omega_base)
        y_damage_distribution_exponent = 0.0 # set damage exponent to zero if no income-dependent damage

    # Eq 1.1: Gross production per capita (Cobb-Douglas)
    # y_gross: gross production before climate damage and abatement cost
    if K > 0 and L > 0:
        y_gross = A * ((K / L) ** alpha)
    else:
        y_gross = 0.0

    # Use Omega from previous timestep for budgeting and damage calculations
    # y_damaged_calc: gross production net of climate damage (using previous timestep's damage)
    y_damaged_calc = y_gross * (1.0 - Omega_calc)
    climate_damage_calc = Omega_calc * y_gross

    t_setup = time.time()
    _timing_stats['setup_time'] += t_setup - t_start

    # -----------------------------------------------------------------------------------------
    #  Do redistribution, taxes, utility, etcc
    # -----------------------------------------------------------------------------------------

    if y_gross <= EPSILON or y_damaged_calc <= EPSILON:
        # Economy has collapsed - set all downstream variables to zero or appropriate values
        redistribution_amount = 0.0
        abateCost_amount = 0.0
        tax_amount = 0.0
        aggregate_utility = NEG_BIGNUM
        aggregate_damage_fraction = 0.0
        Omega = 0.0
        omega_yi = np.zeros_like(xi)
        lambda_abate = 0.0
        y_net = 0.0
        mu = 0.0
        U = NEG_BIGNUM
        e = 0.0
        dK_dt = -delta * K
        omega_yi = np.zeros_like(xi)
        y_net_yi = np.zeros_like(xi)
        utility_yi = np.zeros_like(xi)
    else:
        # Economy exists - proceed with calculations

        available_for_redistribution_and_abatement = fract_gdp *  y_damaged_calc 
        
        if income_redistribution:
            redistribution_amount = (1 - f) * available_for_redistribution_and_abatement
        else:
            redistribution_amount = 0.0

        abateCost_amount = f * available_for_redistribution_and_abatement # per capita
        lambda_abate = abateCost_amount / y_damaged_calc # fraction of damaged production

        tax_amount = abateCost_amount + redistribution_amount # per capita
        # tax amount can be less than amount available if redistribution is turned off.

        # Eq 1.8 & 1.9: Net production per capita after abatement cost and climate damage
        # y_net (aggregate): gross production net of climate damage and abatement cost
        # Note: consumption + savings = y_net
        y_net = (1.0 - lambda_abate) * y_damaged_calc 

        # Find uniform redistribution amount
        if income_redistribution and income_dependent_redistribution_policy:
            # No uniform distribution
            uniform_redistribution_amount = 0.0
        else:
            # Uniform redistribution
            uniform_redistribution_amount = redistribution_amount

        # Pre-compute uniform tax rate when policy is uniform so Fmin uses taxed income
        # Uniform tax always covers full budget (abatement + redistribution)
        # Fmin is calculated AFTER uniform tax to ensure equal consumption below Fmin
        if income_dependent_tax_policy:
            uniform_tax_rate = 0.0
        else:
            uniform_tax_rate = (abateCost_amount + redistribution_amount) / y_damaged_calc

        #------------------------------------------------------
        # Now we are going to do the income dependent part of the code
        # To simplify we are going to shift the calculation to discrete intervals of population
        # governed by the Gaussian Laegendre nodes and weights, xi and wi

        # Find Fmin using current Omega_base
        # For income-dependent redistribution, find Fmin such that redistribution matches target
        if income_redistribution and income_dependent_redistribution_policy:
            uniform_redistribution_amount = 0.0
            # Only find Fmin if there's actually something to redistribute
            if redistribution_amount > EPSILON:
                t_before_fmin = time.time()
                # Find Fmin: everyone pays uniform_tax_rate on Lorenz income,
                # then those below Fmin receive untaxed subsidy to reach Fmin income level
                # This ensures continuity at Fmin (no perverse incentives)
                Fmin = find_Fmin(1.0 - EPSILON,
                    y_gross, gini, Omega_calc, omega_yi_calc,
                    redistribution_amount, uniform_tax_rate, Fi_edges,
                    initial_guess=Fmin_prev,
                )
                _timing_stats['find_Fmin_time'] += time.time() - t_before_fmin
            else:
                # No redistribution amount, so no targeted redistribution
                Fmin = 0.0
        else:
            # Uniform redistribution
            uniform_redistribution_amount = redistribution_amount
            Fmin = 0.0

 
        # For income-dependent tax, find Fmax such that tax matches target
        if income_dependent_tax_policy:
            # Only find Fmax if there's actually something to tax
            if tax_amount > EPSILON:
                # We want to find the value of Fmax such that if everyone above Fmax made
                # the same amount of money as people at Fmax, that would generate the right amount of tax revenue.
                # The piece of resources that would come from the Lorenz curve above Fmax is:
                # lorenz_part = y_gross *( (1 - L(Fmax)) - (1.0 - Fmax) * (d L/dF)@Fmax )
                # damage_part = stepwise_integrate(Fmax, 1.0, omega_yi_prev, Fi_edges) - (1 - Fmax) * stepwise_interpolate(Fmax, omega_yi_prev, Fi_edges)
                # The challenge is to find Fmax such that:
                # tax_amount = lorenz_part - damage_part

                t_before_fmax = time.time()

                # Find Fmax based on post-damage Lorenz income only (no taxes, no redistributions)
                # This defines WHO pays progressive tax based on income rank thresholds
                # Note: find_Fmax always uses uniform_redistribution=0.0 because
                # Fmax determines "who pays" not "how much" (that's determined by tax rates)
                Fmax = find_Fmax(Fmin,
                    y_gross, gini, Omega_calc, omega_yi_calc,
                    redistribution_amount, abateCost_amount, Fi_edges,
                    initial_guess=Fmax_prev,
                )
                _timing_stats['find_Fmax_time'] += time.time() - t_before_fmax
            else:
                # No tax amount, so no income-dependent taxation
                Fmax = 1.0
        else:
            # Uniform tax already computed
            Fmax = 1.0
   
        # Compute consumption, aggregate utility for the Fmin and Fmax region, and at each of the Gauss-Legendre quadrature nodes
        # Also calculate climate damage for next time step.
        # Divide calculation into three segments: [0, Fmin], [Fmin, Fmax], [Fmax, 1]
        #
        # Summary of y variants (income flow):
        # 1. y_gross: gross production before climate damage and abatement cost
        # 2. y_damaged_calc: gross production net of climate damage (using previous timestep's damage)
        # 3. y_net (aggregate): gross production net of climate damage and abatement cost
        # 4. y_damaged_and_uniform: gross production net of climate damage, uniform distributions, and uniform taxes
        #                           (intermediate concept, calculated via y_of_F_after_damage for middle segment)
        # 5. y_net_yi (individual): individual income net of climate damage, abatement cost,
        #                           uniform distributions, uniform taxes, and income-dependent
        #                           redistributions and taxes
        # Note: consumption + savings = y_net
        lorenz_fractions_yi = L_pareto(Fi_edges[1:], gini) - L_pareto(Fi_edges[:-1],gini) # fraction of income in each bin
        lorenz_ratio_yi = lorenz_fractions_yi/Fwi # ratio of mean income in each bin to aggregate mean income

        y_net_yi = np.zeros_like(xi)
        consumption_yi = np.zeros_like(xi)
        utility_yi = np.zeros_like(xi)
        omega_yi = np.zeros_like(xi)

        # Segment 1: Low-income earners receiving income-dependent redistribution [0, Fmin]
        t_before_seg1 = time.time()
        if Fmin > EPSILON:
            # People below Fmin pay uniform tax on Lorenz income, then receive untaxed subsidy
            # The subsidy lifts them to the income level at Fmin
            # This ensures continuity: no perverse incentive to be just below Fmin
            min_y_net = y_net_of_F(
                Fmin, Fmin, Fmax,
                y_gross, omega_yi_calc, Fi_edges,
                uniform_tax_rate, uniform_redistribution_amount, gini,
            )
            min_consumption = min_y_net * (1 - s)
            omega_min = Omega_base * (min_y_net/y_net_reference)**(-y_damage_distribution_exponent)

            # Calculate utility for segment 1
            if eta == 1:
                min_utility = np.log(np.maximum(min_consumption, EPSILON))
            else:
                min_utility = (np.maximum(min_consumption, EPSILON) ** (1 - eta)) / (1 - eta)

            # Set y_net_yi, omega_yi, and utility_yi for bins below/containing Fmin
            for i in range(len(Fi_edges) - 1):
                if Fi_edges[i+1] <= Fmin:
                    # Bin completely below Fmin
                    y_net_yi[i] = min_y_net
                    omega_yi[i] = omega_min
                    utility_yi[i] = min_utility
                elif Fi_edges[i] < Fmin <= Fi_edges[i+1]:
                    # Bin containing Fmin - weight by fraction below Fmin
                    fraction_below = (Fmin - Fi_edges[i]) / Fwi[i]
                    y_net_yi[i] = min_y_net * fraction_below
                    omega_yi[i] = omega_min * fraction_below
                    utility_yi[i] = min_utility * fraction_below

        _timing_stats['segment1_time'] += time.time() - t_before_seg1

        # Segment 3: High-income earners paying income-dependent tax [Fmax, 1]
        t_before_seg3 = time.time()
        if 1.0 - Fmax > EPSILON:
            max_y_net = y_net_of_F(
                Fmax, Fmin, Fmax,
                y_gross, omega_yi_calc, Fi_edges,
                uniform_tax_rate, uniform_redistribution_amount, gini,
            )
            max_consumption = max_y_net * (1 - s)
            omega_max = Omega_base * (max_y_net/y_net_reference)**(-y_damage_distribution_exponent)

            # Calculate utility for segment 3
            if eta == 1:
                max_utility = np.log(np.maximum(max_consumption, EPSILON))
            else:
                max_utility = (np.maximum(max_consumption, EPSILON) ** (1 - eta)) / (1 - eta)

            # Set y_net_yi, omega_yi, and utility_yi for bins above/containing Fmax
            for i in range(len(Fi_edges) - 1):
                if Fi_edges[i] >= Fmax:
                    # Bin completely above Fmax
                    y_net_yi[i] = max_y_net
                    omega_yi[i] = omega_max
                    utility_yi[i] = max_utility
                elif Fi_edges[i] < Fmax <= Fi_edges[i+1]:
                    # Bin containing Fmax - weight by fraction above Fmax
                    fraction_above = (Fi_edges[i+1] - Fmax) / Fwi[i]
                    y_net_yi[i] = max_y_net * fraction_above
                    omega_yi[i] = omega_max * fraction_above
                    utility_yi[i] = max_utility * fraction_above

        _timing_stats['segment3_time'] += time.time() - t_before_seg3

        # Segment 2: Middle-income earners with uniform redistribution/tax [Fmin, Fmax]
        t_before_seg2 = time.time()
        if Fmax - Fmin > EPSILON:
            # Calculate y_net for each bin in the middle segment
            y_vals_Fi = y_net_of_F(
                Fi, Fmin, Fmax,
                y_gross, omega_yi_calc, Fi_edges,
                uniform_tax_rate, uniform_redistribution_amount, gini,
            )

            # Calculate climate damage at quadrature points for next timestep
            if np.abs(y_damage_distribution_exponent) < EPSILON:
                # Uniform damage
                omega_yi_mid = np.full_like(y_vals_Fi, Omega_base)
            else:
                # Income-dependent damage
                omega_yi_mid = Omega_base * (y_vals_Fi / y_net_reference) ** (-y_damage_distribution_exponent)

            # Calculate consumption and utility for middle segment
            consumption_vals = y_vals_Fi * (1 - s)
            if eta == 1:
                utility_vals = np.log(np.maximum(consumption_vals, EPSILON))
            else:
                utility_vals = (np.maximum(consumption_vals, EPSILON) ** (1 - eta)) / (1 - eta)

            # Set y_net_yi, omega_yi, and utility_yi for bins in [Fmin, Fmax]
            for i in range(len(Fi_edges) - 1):
                if Fi_edges[i] >= Fmin and Fi_edges[i+1] <= Fmax:
                    # Bin completely within [Fmin, Fmax]
                    y_net_yi[i] = y_vals_Fi[i]
                    omega_yi[i] = omega_yi_mid[i]
                    utility_yi[i] = utility_vals[i]
                elif Fi_edges[i] < Fmin <= Fi_edges[i+1] < Fmax:
                    # Bin contains Fmin - add weighted contribution for part above Fmin
                    fraction_above = (Fi_edges[i+1] - Fmin) / Fwi[i]
                    y_net_yi[i] += y_vals_Fi[i] * fraction_above
                    omega_yi[i] += omega_yi_mid[i] * fraction_above
                    utility_yi[i] += utility_vals[i] * fraction_above
                elif Fmin < Fi_edges[i] < Fmax <= Fi_edges[i+1]:
                    # Bin contains Fmax - add weighted contribution for part below Fmax
                    fraction_below = (Fmax - Fi_edges[i]) / Fwi[i]
                    y_net_yi[i] += y_vals_Fi[i] * fraction_below
                    omega_yi[i] += omega_yi_mid[i] * fraction_below
                    utility_yi[i] += utility_vals[i] * fraction_below
                elif Fi_edges[i] < Fmin and Fmax <= Fi_edges[i+1]:
                    # Bin contains both Fmin and Fmax - only the middle part
                    fraction_middle = (Fmax - Fmin) / Fwi[i]
                    y_net_yi[i] += y_vals_Fi[i] * fraction_middle
                    omega_yi[i] += omega_yi_mid[i] * fraction_middle
                    utility_yi[i] += utility_vals[i] * fraction_middle

        _timing_stats['segment2_time'] += time.time() - t_before_seg2

        # Calculate aggregate utility by summing over all bins
        aggregate_utility = np.sum(Fwi * utility_yi)

        if not income_dependent_aggregate_damage and income_dependent_damage_distribution:
            # Only rescale if we have income-dependent distribution but want aggregate to match temperature-based damage
            # Rescale to match Omega_base * y_net (consistent with uniform distribution case)
            # total_climate_damage_pre_scale = sum of (damage_fraction * income) across all groups
            total_climate_damage_pre_scale = np.sum(Fwi * y_net_yi * omega_yi)
            y_net_aggregate = np.sum(Fwi * y_net_yi)
            if total_climate_damage_pre_scale > 0 and y_net_aggregate > 0:
                omega_yi = omega_yi * (Omega_base * y_net_aggregate) / total_climate_damage_pre_scale

        # Clip omega_yi to valid range [0.0, 1.0 - EPSILON] since damage fractions cannot exceed 100%
        omega_yi = np.clip(omega_yi, 0.0, 1.0 - EPSILON)

        # Calculate aggregate damage fraction as weighted average across income distribution
        Omega = np.sum(Fwi * y_net_yi * omega_yi) / np.sum(Fwi * y_net_yi)  # Weighted average of damage fractions (dimensionless)

    #========================================================================================
    # mitigation carbon climate
    
        # Eq 2.1: Potential emissions per capita (unabated)
        epot = sigma * y_gross

        # Eq 1.6: Abatement fraction
        if epot > 0 and abateCost_amount > 0:
            mu = min(mu_max, (abateCost_amount * theta2 / (epot * theta1)) ** (1 / theta2))
        else:
            mu = 0.0

        # Eq 2.3: Actual emissions per capita (after abatement)
        e = sigma * (1 - mu) * y_gross
    



        # Eq 1.10: Capital tendency
        dK_dt = s * y_net * L - delta * K

        # aggregate utility
        U = aggregate_utility

    #========================================================================================

    # Prepare output
    results = {}

    if store_detailed_output:
        # Additional calculated variables for detailed output only
        marginal_abatement_cost = theta1 * mu ** (theta2 - 1)  # Social cost of carbon

        # Calculate total (upper-case) variables from per capita variables for recording output
        Y_gross = y_gross * L  # Total gross production
        Y_damaged = y_damaged_calc * L  # Total damaged production
        Y_net = y_net * L  # Total net production
        AbateCost = abateCost_amount * L  # Total abatement cost
        Lambda = lambda_abate  # Abatement cost fraction (same for total and per capita)
        E = e * L  # Total emissions
        Climate_damage = climate_damage_calc * L  # Total climate damage
        Consumption = (1 - s) * Y_net  # Total consumption
        consumption = (1 - s) * y_net  # Per capita consumption
        Savings = s * Y_net  # Total savings
        savings = s * y_net  # Per capita savings
        redistribution = redistribution_amount  # Per capita redistribution (same as redistribution_amount)
        Redistribution_amount = redistribution_amount * L  # Total redistribution amount

        # Calculate Gini coefficients from vector distributions
        if y_gross > EPSILON:
            # Calculate consumption_yi from y_net_yi for Gini calculation
            consumption_yi_full = y_net_yi * (1 - s)

            gini_consumption = gini_from_distribution(consumption_yi_full, Fi_edges, Fwi)

            delta_gini_consumption = gini_consumption - gini

            # Compute utility_yi from consumption_yi
            if eta == 1:
                utility_yi_full = np.log(np.maximum(consumption_yi_full, EPSILON))
            else:
                utility_yi_full = (np.maximum(consumption_yi_full, EPSILON) ** (1 - eta)) / (1 - eta)

            # Calculate utility Gini only when eta < 1
            if eta < 1:
                gini_utility = gini_from_distribution(utility_yi_full, Fi_edges, Fwi)
                delta_gini_utility = gini_utility - gini
            else:
                gini_utility = 0.0
                delta_gini_utility = 0.0
        else:
            gini_consumption = 0.0
            delta_gini_consumption = 0.0
            gini_utility = 0.0
            delta_gini_utility = 0.0

        # Return full diagnostics for CSV/PDF output
        results.update({
            'dK_dt': dK_dt,
            'dEcum_dt': E,
            'Gini': gini,  # Current Gini for plotting
            'gini': gini,  # Background Gini for reference
            'Y_gross': Y_gross,
            'delta_T': delta_T,
            'Omega': Omega,
            'Omega_base': Omega_base,  # Base damage from temperature before income adjustment
            'Omega_calc': Omega_calc,  # Damage from previous timestep (lagged)
            'Y_damaged': Y_damaged,
            'Y_net': Y_net,
            'y_net': y_net,
            'y_damaged': y_damaged_calc,  # Per capita gross production after climate damage
            'climate_damage': climate_damage_calc,  # Per capita climate damage
            'redistribution': redistribution,
            'redistribution_amount': redistribution_amount,  # Per capita redistribution amount
            'Redistribution_amount': Redistribution_amount,  # Total redistribution amount
            'uniform_redistribution_amount': uniform_redistribution_amount,  # Per capita uniform redistribution
            'tax_amount': tax_amount,  # Per capita tax amount (abatement + redistribution)
            'uniform_tax_rate': uniform_tax_rate,  # Uniform tax rate
            'Fmin': Fmin,  # Minimum income rank boundary
            'Fmax': Fmax,  # Maximum income rank boundary
            'min_y_net': min_y_net,  # Net income at Fmin
            'max_y_net': max_y_net,  # Net income at Fmax
            'aggregate_utility': aggregate_utility,  # Aggregate utility from integration
            'mu': mu,
            'Lambda': Lambda,
            'AbateCost': AbateCost,
            'marginal_abatement_cost': marginal_abatement_cost,
            'U': U,
            'E': E,
            'Climate_damage': Climate_damage,
            'Savings': Savings,
            'savings': savings,
            'Consumption': Consumption,
            'consumption': consumption,
            's': s,  # Savings rate (currently constant, may become time-dependent)
            'gini_consumption': gini_consumption,  # Gini coefficient of consumption distribution
            'gini_utility': gini_utility,  # Gini coefficient of utility distribution (0 when eta >= 1)
            'delta_gini_consumption': delta_gini_consumption,  # Change in Gini from input (consumption)
            'delta_gini_utility': delta_gini_utility,  # Change in Gini from input (utility, 0 when eta >= 1)
            'y_net_yi': y_net_yi,  # Per capita net income distribution across quadrature points
            'omega_yi': omega_yi,  # Climate damage fractions (dimensionless) at quadrature points
            'utility_yi': utility_yi,  # Utility distribution across quadrature points
        })

    # Return minimal variables needed for optimization
    results.update({
        'U': U,
        'dK_dt': dK_dt,
        'dEcum_dt': e * L,
    })

    # Always return climate damage ratios for use in next time step
    # Store ratios of damage to base damage, so next timestep can scale by new Omega_base
    if Omega_base > EPSILON:
        results['omega_yi_Omega_base_ratio'] = omega_yi / Omega_base
        results['Omega_Omega_base_ratio'] = Omega / Omega_base
    else:
        results['omega_yi_Omega_base_ratio'] = np.zeros_like(omega_yi)
        results['Omega_Omega_base_ratio'] = 0.0
    results['Fmax'] = Fmax  # Store maximum income rank for next timestep's initial guess
    results['Fmin'] = Fmin  # Store minimum income rank for next timestep's initial guess

    t_end = time.time()
    _timing_stats['total_time'] += t_end - t_start
    _timing_stats['finalize_time'] += t_end - t_setup

    # Print timing stats every 1000000 calls
    if _timing_stats['call_count'] % 1000000 == 0:
        print_timing_stats()

    return results


def integrate_model(config, store_detailed_output=True):
    """
    Integrate the model forward in time using Euler's method.

    Parameters
    ----------
    config : ModelConfiguration
        Complete model configuration including parameters and time-dependent functions
    store_detailed_output : bool, optional
        If True (default), stores all diagnostic variables for CSV/PDF output.
        If False, stores only t, U needed for optimization objective calculation.

    Returns
    -------
    dict
        Time series results with keys:
        - 't': array of time points
        - 'U': array of utility values (always stored)
        - 'L': array of population values (always stored, needed for objective function)

        If store_detailed_output=True, also includes:
        - 'K': array of capital stock values
        - 'Ecum': array of cumulative emissions values
        - 'Gini': array of Gini index values (from background)
        - 'gini': array of background Gini index values
        - 'A', 'sigma', 'theta1', 'f': time-dependent inputs
        - All derived variables: Y_gross, delta_T, Omega, Y_damaged, Y_net,
          redistribution, mu, Lambda, AbateCost, marginal_abatement_cost, y_net, E

    Notes
    -----
    Uses simple Euler integration: state(t+dt) = state(t) + dt * tendency(t)
    This ensures all functional relationships are satisfied exactly at output points.

    Initial conditions are computed automatically:
    - Ecum(0) = Ecum_initial (initial cumulative emissions from configuration)
    - K(0) = K_initial (from configuration)
    """
    # Extract integration parameters
    t_start = config.integration_params.t_start
    t_end = config.integration_params.t_end
    dt = config.integration_params.dt
    n_quad = config.integration_params.n_quad

    # Create time array
    t_array = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t_array)

    # Precompute Gauss-Legendre quadrature nodes and weights (used for all timesteps)
    xi, wi = roots_legendre(n_quad)
    # Create xi_edges: cumulative weights starting at -1, ending at +1
    # wi sums to 2 (integrating over [-1,1]), so cumsum(wi) goes from wi[0] to 2
    # We want edges from -1 to +1, so: -1 + cumsum(wi) goes from -1+wi[0] to 1
    xi_edges = np.concatenate(([-1.0], -1.0 + np.cumsum(wi)))  # length n_quad + 1

    # Initialize climate damage ratios for first timestep
    omega_yi_Omega_base_ratio_prev = np.ones(n_quad)  # Ratio of damage to base damage at quadrature points
    Omega_Omega_base_ratio_prev = 1.0  # Ratio of aggregate damage to base damage

    # Initialize Fmax and Fmin from previous timestep (None for first timestep)
    Fmax_prev = None
    Fmin_prev = None

    # take abatement cost and initial climate damage into account for initial capital
    state = {
        'K': config.scalar_params.K_initial,
        'Ecum': config.scalar_params.Ecum_initial,
    }

    # Initialize storage for variables
    results = {}

    if store_detailed_output:
        # Add storage for all diagnostic variables
        results.update({
            'A': np.zeros(n_steps),
            'sigma': np.zeros(n_steps),
            'theta1': np.zeros(n_steps),
            'f': np.zeros(n_steps),
            'Y_gross': np.zeros(n_steps),
            'delta_T': np.zeros(n_steps),
            'Omega': np.zeros(n_steps),
            'Omega_base': np.zeros(n_steps),
            'Omega_calc': np.zeros(n_steps),
            'Gini': np.zeros(n_steps),  # Total Gini (background + perturbation)
            'gini': np.zeros(n_steps),  # Background Gini
            'Y_damaged': np.zeros(n_steps),
            'Y_net': np.zeros(n_steps),
            'y_damaged': np.zeros(n_steps),
            'climate_damage': np.zeros(n_steps),
            'redistribution': np.zeros(n_steps),
            'redistribution_amount': np.zeros(n_steps),
            'Redistribution_amount': np.zeros(n_steps),
            'uniform_redistribution_amount': np.zeros(n_steps),
            'tax_amount': np.zeros(n_steps),
            'uniform_tax_rate': np.zeros(n_steps),
            'Fmin': np.zeros(n_steps),
            'Fmax': np.zeros(n_steps),
            'aggregate_utility': np.zeros(n_steps),
            'mu': np.zeros(n_steps),
            'Lambda': np.zeros(n_steps),
            'AbateCost': np.zeros(n_steps),
            'marginal_abatement_cost': np.zeros(n_steps),
            'y_net': np.zeros(n_steps),
            'E': np.zeros(n_steps),
            'dK_dt': np.zeros(n_steps),
            'dEcum_dt': np.zeros(n_steps),
            'Climate_damage': np.zeros(n_steps),
            'Savings': np.zeros(n_steps),
            'savings': np.zeros(n_steps),
            'Consumption': np.zeros(n_steps),
            'consumption': np.zeros(n_steps),
            's': np.zeros(n_steps),
            'gini_consumption': np.zeros(n_steps),
            'gini_utility': np.zeros(n_steps),
            'delta_gini_consumption': np.zeros(n_steps),
            'delta_gini_utility': np.zeros(n_steps),
            'min_y_net': np.zeros(n_steps),
            'max_y_net': np.zeros(n_steps),
            'y_net_yi': np.zeros((n_steps, n_quad)),  # Per capita net income distribution
            'omega_yi': np.zeros((n_steps, n_quad)),  # Climate damage fractions (dimensionless) at quadrature points
            'utility_yi': np.zeros((n_steps, n_quad)),  # Utility distribution
        })

    # Always store time, state variables, and objective function variables
    results.update({
        't': t_array,
        'K': np.zeros(n_steps),
        'Ecum': np.zeros(n_steps),
        'U': np.zeros(n_steps),
        'L': np.zeros(n_steps),  # Needed for objective function
    })

    # Store quadrature information (for xlsx output)
    if store_detailed_output:
        results.update({
            'xi': xi,
            'wi': wi,
            'xi_edges': xi_edges,
            'Fi': (xi + 1.0) / 2.0,
            'Fwi': wi / 2.0,
            'Fi_edges': (xi_edges + 1.0) / 2.0,
        })

    # Time stepping loop
    for i, t in enumerate(t_array):
        # Evaluate time-dependent parameters at current time
        params = evaluate_params_at_time(t, config)

        # Calculate all variables and tendencies at current time
        # Pass damage ratios from previous timestep to use lagged damage (avoids circular dependency)
        # Pass Fmax_prev and Fmin_prev to speed up root finding
        outputs = calculate_tendencies(state, params, omega_yi_Omega_base_ratio_prev, Omega_Omega_base_ratio_prev, xi, xi_edges, wi, Fmax_prev, Fmin_prev, store_detailed_output)

        # Always store variables needed for objective function
        results['U'][i] = outputs['U']
        results['L'][i] = params['L']

        if store_detailed_output:
            # Store state variables
            results['K'][i] = state['K']
            results['Ecum'][i] = state['Ecum']

            # Store time-dependent inputs
            results['A'][i] = params['A']
            results['sigma'][i] = params['sigma']
            results['theta1'][i] = params['theta1']
            results['f'][i] = params['f']

            # Store all derived variables
            results['Y_gross'][i] = outputs['Y_gross']
            results['delta_T'][i] = outputs['delta_T']
            results['Omega'][i] = outputs['Omega']
            results['Omega_base'][i] = outputs['Omega_base']
            results['Omega_calc'][i] = outputs['Omega_calc']
            results['Gini'][i] = outputs['Gini']  # Total Gini
            results['gini'][i] = outputs['gini']  # Background Gini
            results['Y_damaged'][i] = outputs['Y_damaged']
            results['Y_net'][i] = outputs['Y_net']
            results['y_damaged'][i] = outputs['y_damaged']
            results['climate_damage'][i] = outputs['climate_damage']
            results['redistribution'][i] = outputs['redistribution']
            results['redistribution_amount'][i] = outputs['redistribution_amount']
            results['Redistribution_amount'][i] = outputs['Redistribution_amount']
            results['uniform_redistribution_amount'][i] = outputs['uniform_redistribution_amount']
            results['tax_amount'][i] = outputs['tax_amount']
            results['uniform_tax_rate'][i] = outputs['uniform_tax_rate']
            results['Fmin'][i] = outputs['Fmin']
            results['Fmax'][i] = outputs['Fmax']
            results['aggregate_utility'][i] = outputs['aggregate_utility']
            results['mu'][i] = outputs['mu']
            results['Lambda'][i] = outputs['Lambda']
            results['AbateCost'][i] = outputs['AbateCost']
            results['marginal_abatement_cost'][i] = outputs['marginal_abatement_cost']
            results['y_net'][i] = outputs['y_net']
            results['E'][i] = outputs['E']
            results['dK_dt'][i] = outputs['dK_dt']
            results['dEcum_dt'][i] = outputs['dEcum_dt']
            results['Climate_damage'][i] = outputs['Climate_damage']
            results['Savings'][i] = outputs['Savings']
            results['savings'][i] = outputs['savings']
            results['Consumption'][i] = outputs['Consumption']
            results['consumption'][i] = outputs['consumption']
            results['s'][i] = outputs['s']
            results['gini_consumption'][i] = outputs['gini_consumption']
            results['gini_utility'][i] = outputs['gini_utility']
            results['delta_gini_consumption'][i] = outputs['delta_gini_consumption']
            results['delta_gini_utility'][i] = outputs['delta_gini_utility']
            results['min_y_net'][i] = outputs['min_y_net']
            results['max_y_net'][i] = outputs['max_y_net']
            results['y_net_yi'][i, :] = outputs['y_net_yi']
            results['omega_yi'][i, :] = outputs['omega_yi']
            results['utility_yi'][i, :] = outputs['utility_yi']

        # Euler step: update state for next iteration (skip on last step)
        if i < n_steps - 1:
            state['K'] = state['K'] + dt * outputs['dK_dt']
            # do not allow cumulative emissions to go negative, making it colder than the initial condition
            state['Ecum'] = max(0.0, state['Ecum'] + dt * outputs['dEcum_dt'])

            # Update climate damage ratios for next time step (lagged damage approach)
            omega_yi_Omega_base_ratio_prev = outputs['omega_yi_Omega_base_ratio']
            Omega_Omega_base_ratio_prev = outputs['Omega_Omega_base_ratio']

            # Update Fmax and Fmin for next time step (speeds up root finding)
            Fmax_prev = outputs.get('Fmax')
            Fmin_prev = outputs.get('Fmin')

    # Print final timing statistics only when called directly (not during optimization)
    # print_timing_stats()

    return results
