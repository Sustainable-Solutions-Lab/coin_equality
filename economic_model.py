"""
Functions for calculating economic production, climate impacts, and system tendencies.

This module implements the Solow-Swann growth model with climate damage
and emissions abatement costs.
"""

import numpy as np
from scipy.special import roots_legendre
from income_distribution import (
    y_of_F_after_damage,
    segment_integral_with_cut,
    total_tax_top,
    total_tax_bottom,
    find_Fmax,
    find_Fmin,
    L_pareto,
    L_pareto_derivative
)
from parameters import evaluate_params_at_time
from utility_integrals import (
    crra_utility_interval,
    crra_utility_integral_with_damage,
    climate_damage_integral
)
from constants import EPSILON, LOOSE_EPSILON, NEG_BIGNUM, MAX_ITERATIONS, N_QUAD, INVERSE_EPSILON


def calculate_tendencies(state, params, previous_step_values, store_detailed_output=True):
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
        - 'Gini_background': Background Gini index (current, from time function)
        - 'Gini_fract': Fraction of Gini change as instantaneous step
        - 'Gini_restore': Rate of restoration to Gini_background (yr^-1)
        - 'fract_gdp': Fraction of GDP available for redistribution and abatement
        - 'f': Fraction allocated to abatement vs redistribution
    previous_step_values : dict
        Income distribution from the previous time step, used for damage/tax/redistribution
        calculations to avoid circular dependency. Contains:
        - 'y_mean': Mean income from previous time step ($)
        - 'gini': Gini coefficient from previous time step
    store_detailed_output : bool, optional
        Whether to compute and return all intermediate variables. Default: True

    Returns
    -------
    dict
        Dictionary containing:
        - Tendencies: 'dK_dt', 'dEcum_dt'
        - Income distribution: 'current_income_dist' with {'y_mean': float, 'gini': float}
          for use as previous_step_values in the next time step
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
    A = params['A']
    L = params['L']
    sigma = params['sigma']
    theta1 = params['theta1']
    theta2 = params['theta2']
    mu_max = params['mu_max']
    fract_gdp = params['fract_gdp']
    Gini_background = params['Gini_background']
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

    #========================================================================================
    # Calculate quantities that don't require iteration

    # Eq 1.1: Gross production (Cobb-Douglas)
    if K > 0 and L > 0:
        Y_gross = A * (K ** alpha) * (L ** (1 - alpha))
        y_gross = Y_gross / L
    else:
        Y_gross = 0.0
        y_gross = 0.0

    # Eq 2.2: Temperature change from cumulative emissions
    delta_T = k_climate * Ecum

    # Base damage from temperature (capped just below 1.0 to avoid division by zero)
    Omega = min(psi1 * delta_T + psi2 * (delta_T ** 2), 1.0 - EPSILON)

    # Target Omega: only iterate if we have income-dependent damage distribution
    # AND income_dependent_aggregate_damage is False
    if income_dependent_damage_distribution and not income_dependent_aggregate_damage:
        Omega_target = Omega
    else:
        # No iteration needed: either damage is uniform, or we use the aggregate directly
        Omega_target = None  # Signal that we won't iterate

    # Initialize Omega using base damage as starting guess
    Omega_base = Omega
    # if income_dependent_damage_distribution and not income_dependent_aggregate_damage,
    # we will be updating Omega_base to match aggregate damage
    # start with a higher value to help convergence
    if income_dependent_damage_distribution and not income_dependent_aggregate_damage:
        Omega_base = Omega_target * (y_gross / y_net_reference)**y_damage_distribution_exponent

    # Initialize Omega_base_prev (only actually used when income_dependent_damage_distribution
    # is true and income_dependent_aggregate_damage is false, but initialize here to avoid
    # UnboundLocalError in convergence checks)
    Omega_base_prev = Omega_base

    # Current Gini coefficient from background
    gini = Gini_background

    #========================================================================================
    # Iterative convergence loop for climate damage (Section 6 of IMPLEMENTATION_PLAN.md)
    # We iterate to achieve consistency between climate damage and income distribution

    # Precompute Gauss-Legendre quadrature nodes and weights for numerical integration
    xi, wi = roots_legendre(N_QUAD)



    uniform_redistribution_amount = 0.0  # uniform per capita redistribution, will get updated in loop
    uniform_tax_rate = 0.0  # uniform tax rate, will get updated in loop
    Fmin = 0.0  # minimum income boundary for redistribution, will get updated in loop
    Fmax = 1.0  # maximum income boundary for redistribution, will get updated in loop

    # If y_gross is effectively zero, there's no economy to iterate on - skip the loop
    if y_gross > EPSILON:
        converged = False
    else:
        converged = True
        # Predefine variables needed after loop
        redistribution_amount = 0.0
        abateCost_amount = 0.0
        aggregate_utility = NEG_BIGNUM  # Penalty for pathological state

    n_damage_iterations = 0

    # Track convergence history for diagnostics
    omega_history = [Omega]
    omega_base_history = [Omega_base]

    # Track bracketing points for secant/regula falsi method
    lower_bound = None  # (Omega_base, Omega) where Omega < Omega_target
    upper_bound = None  # (Omega_base, Omega) where Omega > Omega_target

    while not converged:
        n_damage_iterations += 1
        if n_damage_iterations > MAX_ITERATIONS:
            omega_base_diff = abs(Omega_base - Omega_base_prev)
            omega_base_frac_diff = abs(Omega_base / Omega_base_prev - 1.0) if Omega_base_prev != 0 else 0.0
            # Print full convergence history for offline analysis
            print(f"\nConvergence failure diagnostics (full history):")
            print(f"  Iteration | Omega          | Omega_base     | Omega_diff     | Omega_base_diff")
            print(f"  ----------|----------------|----------------|----------------|----------------")
            for i in range(len(omega_history)):
                omega_diff_i = abs(omega_history[i] - omega_history[i-1]) if i > 0 else 0.0
                omega_base_diff_i = abs(omega_base_history[i] - omega_base_history[i-1]) if i > 0 else 0.0
                print(f"  {i+1:9d} | {omega_history[i]:14.10f} | {omega_base_history[i]:14.10f} | {omega_diff_i:14.2e} | {omega_base_diff_i:14.2e}")
            raise RuntimeError(
                f"Climate damage calculation failed to converge after {MAX_ITERATIONS} iterations. "
                f"Omega_diff = {abs(Omega - Omega_prev):.2e}, "
                f"Omega_base_frac_diff = {omega_base_frac_diff:.2e} (tolerance: {LOOSE_EPSILON:.2e})"
            )

        # total redistribution amount
        # Use Omega_target for iterative case to avoid lag effects
        # (only when income_dependent_damage_distribution and not income_dependent_aggregate_damage)
        if income_dependent_damage_distribution and not income_dependent_aggregate_damage:
            omega_for_budget = min(Omega_target, 1.0 - EPSILON)
        else:
            omega_for_budget = min(Omega, 1.0 - EPSILON)
        available_for_redistribution_and_abatement = fract_gdp * y_gross * (1 - omega_for_budget)

        if income_redistribution:
            redistribution_amount = (1 - f) * available_for_redistribution_and_abatement  # total redistribution amount
        else:
            redistribution_amount = 0.0

        abateCost_amount = f * available_for_redistribution_and_abatement  # total abatement cost amount

        # Now we calculate Fmin and uniform_redistribution amount, which is the minimum income rank that covers abatement or redistribution costs
        if income_redistribution and income_dependent_redistribution_policy:
            uniform_redistribution_amount = 0.0 # income dependent redistribution
            Fmin = find_Fmin(y_gross , Omega_base, y_damage_distribution_exponent, y_net_reference, uniform_redistribution_amount, gini,xi,wi,target_subsidy = redistribution_amount)
        else: # uniform redistribution
            uniform_redistribution_amount = redistribution_amount   # per capita uniform redistribution amount
            Fmin = 0.0

        # Now we calculate Fmax, which is the minimum income rank that covers abatement or redistribution costs
        if income_dependent_tax_policy:
            tax_amount = abateCost_amount + redistribution_amount  # total tax amount available for redistribution and abatement; be able to handle case when redistribution is not allowed but abatement is limited
            uniform_tax_rate = 0.0 # income dependent tax
            Fmax = find_Fmax(Fmin, y_gross, Omega_base, y_damage_distribution_exponent, y_net_reference, uniform_redistribution_amount, gini,xi,wi,target_tax = tax_amount)
        else:
            uniform_tax_rate = (abateCost_amount + redistribution_amount) / (y_gross * (1 - omega_for_budget))  # uniform tax rate; be able to handle case when redistribution is not allowed but abatement is limited
            Fmax = 1.0

        # Now we know the taxes and redistribution amounts, we can calculate the climate damage as a function of income rank F,
        # and from that calculate the new Omega, and also aggregate utility and the utilty distribution

        # we will divide this calculation into three segments:
        # 0 to Fmin: bottom income earners who receive income-dependent redistribution and maybe uniform redistribution and uniform tax
        # Fmin to Fmax: middle income earners who may receive uniform distribution and pay uniform tax
        # Fmax to 1: top income earners who may pay income-dependent tax and maybe uniform tax and maybe  uniform redistribution
        aggregate_damage_fraction = 0.0
        aggregate_utility = 0.0

        

        if Fmin > EPSILON: #  Low-income: This is the folks who are receiving income-dependent redistribution
            min_y_net_before_savings = y_of_F_after_damage(Fmin, Fmin, Fmax, y_gross * (1 - uniform_tax_rate), Omega_base, y_damage_distribution_exponent, y_net_reference, uniform_redistribution_amount, gini)
            min_consumption = min_y_net_before_savings * (1 - s)
            Omega_min = Omega_base * (min_y_net_before_savings / y_net_reference)**y_damage_distribution_exponent  # everyone below Fmin has the same income
            aggregate_damage_fraction = aggregate_damage_fraction + Fmin * Omega_min # so we can just multiply that damage by Fmin
            aggregate_utility = aggregate_utility + crra_utility_interval(0, Fmin, min_consumption, eta)

        if Fmax - Fmin > EPSILON:  #  Middle-income: This is the folks in the middle who may receive uniform redistribution and pay uniform tax
            aggregate_damage_fraction = aggregate_damage_fraction + climate_damage_integral(Fmin, Fmax, Fmin, Fmax, y_gross * (1 - uniform_tax_rate), Omega_base, y_damage_distribution_exponent, y_net_reference, uniform_redistribution_amount, gini, xi, wi)
            aggregate_utility = aggregate_utility + crra_utility_integral_with_damage(Fmin, Fmax, Fmin, Fmax, y_gross * (1 - uniform_tax_rate), Omega_base, y_damage_distribution_exponent, y_net_reference, uniform_redistribution_amount, gini, eta, s, xi, wi)

        if 1.0 - Fmax > EPSILON: # High-income: This is the folks who are paying income-dependent tax
            max_y_net_before_savings = y_of_F_after_damage(Fmax, Fmin, Fmax, y_gross* (1 - uniform_tax_rate), Omega_base, y_damage_distribution_exponent, y_net_reference, uniform_redistribution_amount, gini)
            max_consumption = max_y_net_before_savings * (1 - s)
            Omega_max = Omega_base * (max_y_net_before_savings / y_net_reference)**y_damage_distribution_exponent
            aggregate_damage_fraction = aggregate_damage_fraction + (1 - Fmax) * Omega_max
            aggregate_utility = aggregate_utility + crra_utility_interval(Fmax, 1.0, max_consumption, eta)

        # Update Omega from aggregate damage (cap at physical maximum)
        Omega_prev = Omega
        Omega = min(aggregate_damage_fraction, 1.0 - EPSILON)

        # If we have income-dependent damage distribution but NOT income-dependent aggregate damage,
        # iterate to find Omega_base that produces target aggregate damage
        if income_dependent_damage_distribution and not income_dependent_aggregate_damage:
            # if we do not have income_dependent aggregate damage we want to scale omega base to match the aggregate damage

            if Omega_base < EPSILON:
                # No base damage - should have no aggregate damage either
                if Omega > EPSILON:
                    raise RuntimeError(f"Logic error: Omega_base = {Omega_base} is near zero but Omega = {Omega} is not")
                else:
                    Omega_base_prev = Omega_base
                    Omega = 0.0
            else:
                Omega_base_prev = Omega_base

                # If Omega is near catastrophic damage, declare convergence
                # This is an extreme/pathological case that won't be optimal anyway
                if Omega >= 1.0 - LOOSE_EPSILON:
                    break

                #----------------------------------------------------------------------------------
                # This next chunk of code implements a secant-like method to update Omega_base
                # It is somewhat involved because a naive approach can take a long time to converge
                # or perhaps not converge at all.

                # We will use a combination of secant method and multiplicative updates

                # Update bracketing bounds
                if Omega < Omega_target:
                    if lower_bound is None or Omega > lower_bound[1]:
                        lower_bound = (Omega_base, Omega)
                elif Omega > Omega_target:
                    if upper_bound is None or Omega < upper_bound[1]:
                        upper_bound = (Omega_base, Omega)

                # Choose update method based on available bracketing information
                if lower_bound is not None and upper_bound is not None:
                    # We have bracketing - use secant method with safeguards
                    omega_base_lower, omega_lower = lower_bound
                    omega_base_upper, omega_upper = upper_bound

                    # Linear interpolation (secant method)
                    if abs(omega_upper - omega_lower) > EPSILON:
                        Omega_base_new = omega_base_lower + (Omega_target - omega_lower) * (omega_base_upper - omega_base_lower) / (omega_upper - omega_lower)

                        # Safeguard: don't allow update to go outside bracketing interval
                        Omega_base_new = np.clip(Omega_base_new, min(omega_base_lower, omega_base_upper), max(omega_base_lower, omega_base_upper))

                        # Apply update - use full step since we're already safeguarded by bracketing
                        Omega_base = Omega_base_new
                    else:
                        # Bounds are too close - we've converged
                        Omega_base = omega_base_lower

                elif n_damage_iterations >= 2 and len(omega_base_history) >= 2:
                    # No bracketing yet, but we have history - use secant method on last two points
                    omega_base_prev2 = omega_base_history[-2]
                    omega_prev2 = omega_history[-2]

                    if abs(Omega - omega_prev2) > EPSILON:
                        # Secant step from last two iterations - trust it fully to encourage overshooting
                        Omega_base_new = Omega_base + (Omega_target - Omega) * (Omega_base - omega_base_prev2) / (Omega - omega_prev2)

                        # Use full step to encourage bracketing, but cap to prevent overflow
                        Omega_base = min(Omega_base_new, INVERSE_EPSILON)
                    else:
                        # Denominatoris too small, fall back to multiplicative update
                        if Omega > LOOSE_EPSILON:
                            Omega_base = min(Omega_base * (Omega_target / Omega), INVERSE_EPSILON)
                        else:
                            Omega_base = min(Omega_base + 0.5 * (Omega_target - Omega_base), INVERSE_EPSILON)

                else:
                    # First iteration or no history - use aggressive multiplicative/additive update to encourage bracketing
                    if Omega < LOOSE_EPSILON:
                        # Omega is very small - use additive update, but cap to prevent overflow
                        Omega_base = min(Omega_base + 0.7 * (Omega_target - Omega_base), INVERSE_EPSILON)
                    else:
                        # Normal multiplicative update - full step to encourage overshooting, but cap to prevent overflow
                        Omega_base = min(Omega_base * (Omega_target / Omega), INVERSE_EPSILON)

                #----end of iteration -------------------------------------------------------------------

        # Record convergence history for diagnostics
        omega_history.append(Omega)
        omega_base_history.append(Omega_base)

        # Check convergence
        # Primary criterion: Omega should be close to target (what we actually care about)
        # Secondary criterion: Omega_base should be changing slowly (indicates stability)
        omega_converged = abs(Omega - Omega_prev) < LOOSE_EPSILON

        # For Omega_base, use a looser fractional tolerance since it's a means to an end
        # Also check absolute convergence in case we're close to target from one side
        omega_base_frac_converged = abs(Omega_base / Omega_base_prev - 1.0) < LOOSE_EPSILON if Omega_base_prev != 0 else True
        omega_base_abs_converged = abs(Omega_base - Omega_base_prev) < LOOSE_EPSILON

        # If Omega itself is very close to target, accept convergence even if Omega_base is still changing slightly
        # (only relevant when iterating: income_dependent_damage_distribution and not income_dependent_aggregate_damage)
        omega_near_target = abs(Omega - Omega_target) < LOOSE_EPSILON if (income_dependent_damage_distribution and not income_dependent_aggregate_damage) else True

        if omega_converged and (omega_base_frac_converged or omega_base_abs_converged or omega_near_target):
            converged = True


    #========================================================================================
    # Calculate downstream economic variables

    # Eq 1.3: Production after climate damage
    Climate_Damage = Omega * Y_gross
    climate_damage = Omega * y_gross # per capita climate damage
    
    Y_damaged = Y_gross *(1 - Omega)
    y_damaged = y_gross * (1 - Omega)  # per capita gross production after climate damage

    AbateCost = abateCost_amount * L  # total abatement cost
    # Eq 1.7: Abatement cost as fraction of damaged production
    # If Y_damaged is 0 (catastrophic climate damage), set Lambda = 1 (not in optimal state)
    if Y_damaged == 0:
        Lambda = 1.0
    else:
        Lambda = AbateCost / Y_damaged  

    Y_net = Y_damaged - AbateCost # Eq 1.8: Net production after abatement cost
    y_net = y_damaged - abateCost_amount  # Eq 1.9: per capita income after abatement cost

    Consumption = (1-s) * Y_net
    consumption = (1-s) * y_net  # per capita consumption

    Savings = s * Y_net  # Total savings

    # Redistribution tracking
    redistribution = redistribution_amount  # Per capita redistribution (same as redistribution_amount)
    Redistribution_amount = redistribution_amount * L  # total redistribution amount

    # Eq 2.1: Potential emissions (unabated)
    Epot = sigma * Y_gross

    # Eq 1.6: Abatement fraction
    if Epot > 0 and AbateCost > 0:
        mu = min(mu_max, (AbateCost * theta2 / (Epot * theta1)) ** (1 / theta2))
    else:
        mu = 0.0

    # Eq 2.3: Actual emissions (after abatement)
    E = sigma * (1 - mu) * Y_gross

    # Eq 1.10: Capital tendency
    dK_dt = s * Y_net - delta * K

    # aggregate utility
    U = aggregate_utility

    #========================================================================================

    # Handle edge cases where economy has collapsed
    if y_gross <= 0 or Y_gross <= 0:
        Omega = 0.0
        Climate_Damage = 0.0
        Y_damaged = 0.0
        Savings = 0.0
        Lambda = 0.0
        AbateCost = 0.0
        Y_net = 0.0
        Redistribution_amount = 0.0
        Consumption = 0.0
        y_net = 0.0
        redistribution = 0.0
        mu = 0.0
        U = NEG_BIGNUM
        E = 0.0
        dK_dt = -delta * K
        
    # Prepare output
    results = {}

    if store_detailed_output:
        # Additional calculated variables for detailed output only
        marginal_abatement_cost = theta1 * mu ** (theta2 - 1)  # Social cost of carbon
        discounted_utility = U * np.exp(-rho * t)  # Discounted utility

        # Return full diagnostics for CSV/PDF output
        results.update({
            'dK_dt': dK_dt,
            'dEcum_dt': E,
            'Gini': gini,  # Current Gini for plotting
            'Gini_background': Gini_background,  # Background Gini for reference
            'Y_gross': Y_gross,
            'delta_T': delta_T,
            'Omega': Omega,
            'Omega_base': Omega_base,  # Base damage from temperature before income adjustment
            'Y_damaged': Y_damaged,
            'Y_net': Y_net,
            'y_net': y_net,
            'y_damaged': y_damaged,  # Per capita gross production after climate damage
            'climate_damage': climate_damage,  # Per capita climate damage
            'redistribution': redistribution,
            'redistribution_amount': redistribution_amount,  # Per capita redistribution amount
            'Redistribution_amount': Redistribution_amount,  # Total redistribution amount
            'uniform_redistribution_amount': uniform_redistribution_amount,  # Per capita uniform redistribution
            'uniform_tax_rate': uniform_tax_rate,  # Uniform tax rate
            'Fmin': Fmin,  # Minimum income rank boundary
            'Fmax': Fmax,  # Maximum income rank boundary
            'n_damage_iterations': n_damage_iterations,  # Number of convergence iterations
            'aggregate_utility': aggregate_utility,  # Aggregate utility from integration
            'mu': mu,
            'Lambda': Lambda,
            'AbateCost': AbateCost,
            'marginal_abatement_cost': marginal_abatement_cost,
            'U': U,
            'E': E,
            'Climate_Damage': Climate_Damage,
            'Savings': Savings,
            'Consumption': Consumption,
            'discounted_utility': discounted_utility,
            's': s,  # Savings rate (currently constant, may become time-dependent)
        })

    # Return minimal variables needed for optimization
    results.update({
        'U': U,
        'dK_dt': dK_dt,
        'dEcum_dt': E,
    })

    # Always return current income distribution for use as previous_step_values in next time step
    # Use y_net (net per-capita income)
    results['current_income_dist'] = {
        'y_mean': y_net
    }

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
        - 'Gini_background': array of background Gini index values
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

    # Create time array
    t_array = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t_array)

    # Calculate initial state
    A0 = config.time_functions['A'](t_start)
    L0 = config.time_functions['L'](t_start)
    delta = config.scalar_params.delta
    alpha = config.scalar_params.alpha
    fract_gdp = config.scalar_params.fract_gdp

    # take abatement cost and initial climate damage into account for initial capital
    Ecum_initial = config.scalar_params.Ecum_initial
    params = evaluate_params_at_time(t_start, config)

    Gini = config.time_functions['Gini_background'](t_start)
    k_climate = params['k_climate']
    delta_T = k_climate * Ecum_initial

    # iterate to find K0 that is consistent with climate damage from initial emissions
    Omega_prev = 1.0
    Omega_current = 0.0
    n_iterations = 0

    """
    # get time-dependent parameters at t_start
    s0 = params['s']
    f0 = params['f']
    k_climate = params['k_climate']
    lambda0 = (1-s0) * f0 * fract_gdp

    while np.abs(Omega_current - Omega_prev) > EPSILON:
        n_iterations += 1
        if n_iterations > MAX_ITERATIONS:
            raise RuntimeError(
                f"Initial capital stock failed to converge after {MAX_ITERATIONS} iterations. "
                f"Omega_prev = {Omega_prev:.10f}, Omega_current = {Omega_current:.10f}, "
                f"difference = {np.abs(Omega_current - Omega_prev):.2e} (tolerance: {EPSILON:.2e})"
            )
        Omega_prev = Omega_current
        K0 = ((s0 * (1 - Omega_prev) * (1 - lambda0) * A0 / delta) ** (1 / (1 - alpha))) * L0
        y_gross = A0 * (K0 ** alpha) * (L0 ** (1 - alpha)) / L0
        Omega_current, _ = calculate_climate_damage_and_gini_effect(
            delta_T, Gini, y_gross, params
        )

    """
    state = {
        'K': config.scalar_params.K_initial,
        'Ecum': config.scalar_params.Ecum_initial,
    }

    # Initialize previous_step_values for first time step
    # Use initial gross income and background Gini as starting point
    L0 = config.time_functions['L'](t_start)
    A0 = config.time_functions['A'](t_start)
    K0 = config.scalar_params.K_initial
    alpha = config.scalar_params.alpha
    Y_gross_initial = A0 * (K0 ** alpha) * (L0 ** (1 - alpha))
    y_gross_initial = Y_gross_initial / L0 if L0 > 0 else 0.0
    Gini_initial = config.time_functions['Gini_background'](t_start)

    previous_step_values = {
        'y_mean': y_gross_initial,
        'gini': Gini_initial,
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
            'Gini': np.zeros(n_steps),  # Total Gini (background + perturbation)
            'Gini_background': np.zeros(n_steps),  # Background Gini
            'Y_damaged': np.zeros(n_steps),
            'Y_net': np.zeros(n_steps),
            'y_damaged': np.zeros(n_steps),
            'climate_damage': np.zeros(n_steps),
            'redistribution': np.zeros(n_steps),
            'redistribution_amount': np.zeros(n_steps),
            'Redistribution_amount': np.zeros(n_steps),
            'uniform_redistribution_amount': np.zeros(n_steps),
            'uniform_tax_rate': np.zeros(n_steps),
            'Fmin': np.zeros(n_steps),
            'Fmax': np.zeros(n_steps),
            'n_damage_iterations': np.zeros(n_steps),
            'aggregate_utility': np.zeros(n_steps),
            'mu': np.zeros(n_steps),
            'Lambda': np.zeros(n_steps),
            'AbateCost': np.zeros(n_steps),
            'marginal_abatement_cost': np.zeros(n_steps),
            'y_net': np.zeros(n_steps),
            'E': np.zeros(n_steps),
            'dK_dt': np.zeros(n_steps),
            'dEcum_dt': np.zeros(n_steps),
            'Climate_Damage': np.zeros(n_steps),
            'Savings': np.zeros(n_steps),
            'Consumption': np.zeros(n_steps),
            'discounted_utility': np.zeros(n_steps),
            's': np.zeros(n_steps),
        })

    # Always store time, state variables, and objective function variables
    results.update({
        't': t_array,
        'K': np.zeros(n_steps),
        'Ecum': np.zeros(n_steps),
        'U': np.zeros(n_steps),
        'L': np.zeros(n_steps),  # Needed for objective function
    })

    # Time stepping loop
    for i, t in enumerate(t_array):
        # Evaluate time-dependent parameters at current time
        params = evaluate_params_at_time(t, config)

        # Calculate all variables and tendencies at current time
        # Pass previous_step_values to avoid circular dependency in damage calculations
        outputs = calculate_tendencies(state, params, previous_step_values, store_detailed_output)

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
            results['Gini'][i] = outputs['Gini']  # Total Gini
            results['Gini_background'][i] = outputs['Gini_background']  # Background Gini
            results['Y_damaged'][i] = outputs['Y_damaged']
            results['Y_net'][i] = outputs['Y_net']
            results['y_damaged'][i] = outputs['y_damaged']
            results['climate_damage'][i] = outputs['climate_damage']
            results['redistribution'][i] = outputs['redistribution']
            results['redistribution_amount'][i] = outputs['redistribution_amount']
            results['Redistribution_amount'][i] = outputs['Redistribution_amount']
            results['uniform_redistribution_amount'][i] = outputs['uniform_redistribution_amount']
            results['uniform_tax_rate'][i] = outputs['uniform_tax_rate']
            results['Fmin'][i] = outputs['Fmin']
            results['Fmax'][i] = outputs['Fmax']
            results['n_damage_iterations'][i] = outputs['n_damage_iterations']
            results['aggregate_utility'][i] = outputs['aggregate_utility']
            results['mu'][i] = outputs['mu']
            results['Lambda'][i] = outputs['Lambda']
            results['AbateCost'][i] = outputs['AbateCost']
            results['marginal_abatement_cost'][i] = outputs['marginal_abatement_cost']
            results['y_net'][i] = outputs['y_net']
            results['E'][i] = outputs['E']
            results['dK_dt'][i] = outputs['dK_dt']
            results['dEcum_dt'][i] = outputs['dEcum_dt']
            results['Climate_Damage'][i] = outputs['Climate_Damage']
            results['Savings'][i] = outputs['Savings']
            results['Consumption'][i] = outputs['Consumption']
            results['discounted_utility'][i] = outputs['discounted_utility']
            results['s'][i] = outputs['s']

        # Euler step: update state for next iteration (skip on last step)
        if i < n_steps - 1:
            state['K'] = state['K'] + dt * outputs['dK_dt']
            # do not allow cumulative emissions to go negative, making it colder than the initial condition
            state['Ecum'] = max(0.0, state['Ecum'] + dt * outputs['dEcum_dt'])

            # Update previous_step_values for next time step
            previous_step_values = outputs['current_income_dist']

    return results
