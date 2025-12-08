"""
Functions for calculating economic production, climate impacts, and system tendencies.

This module implements the Solow-Swann growth model with climate damage
and emissions abatement costs.
"""

import numpy as np
from scipy.special import roots_legendre
from income_distribution import (
    y_of_F_after_damage,
    y_of_F_lagged_damage,
    segment_integral_with_cut,
    total_tax_top,
    total_tax_bottom,
    find_Fmax,
    find_Fmin,
    L_pareto,
    L_pareto_derivative
)
from income_distribution_lagged import (
    find_Fmin_lagged,
    find_Fmax_lagged,
    compute_damage_integral_lagged,
)
from parameters import evaluate_params_at_time
from utility_integrals import (
    crra_utility_interval,
    crra_utility_integral_with_damage,
    climate_damage_integral
)
from constants import EPSILON, LOOSE_EPSILON, NEG_BIGNUM, MAX_ITERATIONS, N_QUAD, INVERSE_EPSILON


def calculate_tendencies(state, params, previous_step_values, xi, wi, store_detailed_output=True):
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

    #========================================================================================
    # Explicit lagged damage calculation (replaces iterative convergence loop)
    # Uses damage from previous timestep to compute current income distribution explicitly

    # Extract previous timestep values for lagged damage calculation
    Omega_base_prev = previous_step_values['Omega_base']
    Fmin_prev = previous_step_values['Fmin']
    Fmax_prev = previous_step_values['Fmax']
    y_gross_prev = previous_step_values['y_gross']
    uniform_redistribution_prev = previous_step_values['uniform_redistribution_amount']
    uniform_tax_rate_prev = previous_step_values['uniform_tax_rate']
    gini_prev = previous_step_values['gini']
    yi_prev = previous_step_values.get('yi', None)  # Income at quadrature points from previous timestep
    Climate_Damage_prev = previous_step_values.get('Climate_Damage', 0.0)  # Total climate damage from previous timestep

    # Initialize variables
    uniform_redistribution_amount = 0.0
    uniform_tax_rate = 0.0
    Fmin = 0.0
    Fmax = 1.0
    n_damage_iterations = 1  # Lagged approach: single pass, no iteration

    # Handle edge case: no economy
    if y_gross <= EPSILON:
        redistribution_amount = 0.0
        abateCost_amount = 0.0
        aggregate_utility = NEG_BIGNUM
        aggregate_damage_fraction = 0.0
        Omega = 0.0
        yi = None
    else:
        # Compute redistribution and abatement amounts using Omega_target (from temperature)
        # For lagged approach, we use Omega_target directly without iteration
        if income_dependent_damage_distribution and not income_dependent_aggregate_damage:
            omega_for_budget = min(Omega_target, 1.0 - EPSILON)
        else:
            omega_for_budget = min(Omega, 1.0 - EPSILON)

        available_for_redistribution_and_abatement = fract_gdp * y_gross * (1 - omega_for_budget)

        if income_redistribution:
            redistribution_amount = (1 - f) * available_for_redistribution_and_abatement
        else:
            redistribution_amount = 0.0

        abateCost_amount = f * available_for_redistribution_and_abatement

        # Find Fmin and Fmax using lagged damage
        # For income-dependent redistribution, find Fmin such that redistribution matches target
        if income_redistribution and income_dependent_redistribution_policy:
            uniform_redistribution_amount = 0.0
            Fmin = find_Fmin_lagged(
                Fmax, y_gross, uniform_redistribution_amount, uniform_tax_rate, gini,
                Omega_base_prev, y_damage_distribution_exponent, y_net_reference,
                Fmin_prev, Fmax_prev, y_gross_prev,
                uniform_redistribution_prev, uniform_tax_rate_prev, gini_prev,
                xi, wi, target_subsidy=redistribution_amount,
            )
        else:
            # Uniform redistribution
            uniform_redistribution_amount = redistribution_amount
            Fmin = 0.0

        # For income-dependent tax, find Fmax such that tax matches target
        if income_dependent_tax_policy:
            tax_amount = abateCost_amount + redistribution_amount
            uniform_tax_rate = 0.0
            Fmax = find_Fmax_lagged(
                Fmin, y_gross, uniform_redistribution_amount, uniform_tax_rate, gini,
                Omega_base_prev, y_damage_distribution_exponent, y_net_reference,
                Fmin_prev, Fmax_prev, y_gross_prev,
                uniform_redistribution_prev, uniform_tax_rate_prev, gini_prev,
                xi, wi, target_tax=tax_amount,
            )
        else:
            # Uniform tax
            uniform_tax_rate = (abateCost_amount + redistribution_amount) / (y_gross * (1 - omega_for_budget))
            Fmax = 1.0

        # Compute aggregate damage and utility using lagged damage
        # Divide calculation into three segments: [0, Fmin], [Fmin, Fmax], [Fmax, 1]
        aggregate_damage_fraction = 0.0
        aggregate_utility = 0.0

        # Segment 1: Low-income earners receiving income-dependent redistribution [0, Fmin]
        if Fmin > EPSILON:
            min_y_net = y_of_F_lagged_damage(
                Fmin,
                Fmin, Fmax,
                y_gross * (1 - uniform_tax_rate), uniform_redistribution_amount, uniform_tax_rate, gini,
                Omega_base_prev, y_damage_distribution_exponent, y_net_reference,
                Fmin_prev, Fmax_prev, y_gross_prev,
                uniform_redistribution_prev, uniform_tax_rate_prev, gini_prev,
            )
            min_consumption = min_y_net * (1 - s)

            # Damage for this segment (everyone has same income at Fmin)
            # Use previous income to compute damage fraction
            y_prev_at_Fmin = y_of_F_after_damage(
                Fmin, Fmin_prev, Fmax_prev, y_gross_prev, Omega_base_prev,
                y_damage_distribution_exponent, y_net_reference,
                uniform_redistribution_prev, gini_prev,
            )
            if np.abs(y_damage_distribution_exponent) < EPSILON:
                Omega_min = Omega_base_prev
            else:
                Omega_min = Omega_base_prev * (y_prev_at_Fmin / y_net_reference) ** y_damage_distribution_exponent

            aggregate_damage_fraction += Fmin * Omega_min
            aggregate_utility += crra_utility_interval(0, Fmin, min_consumption, eta)

        # Segment 2: Middle-income earners with uniform redistribution/tax [Fmin, Fmax]
        yi = None  # Income at quadrature points for next timestep
        if Fmax - Fmin > EPSILON:
            damage_mid = compute_damage_integral_lagged(
                Fmin, Fmax,
                Fmin, Fmax,
                y_gross * (1 - uniform_tax_rate), uniform_redistribution_amount, uniform_tax_rate, gini,
                Omega_base_prev, y_damage_distribution_exponent, y_net_reference,
                Fmin_prev, Fmax_prev, y_gross_prev,
                uniform_redistribution_prev, uniform_tax_rate_prev, gini_prev,
                xi, wi,
            )
            aggregate_damage_fraction += damage_mid

            # Utility for middle segment - integrate over income distribution
            # Map quadrature nodes to [Fmin, Fmax]
            interval_width = Fmax - Fmin
            F_nodes = Fmin + 0.5 * interval_width * (xi + 1.0)
            y_vals = y_of_F_lagged_damage(
                F_nodes,
                Fmin, Fmax,
                y_gross * (1 - uniform_tax_rate), uniform_redistribution_amount, uniform_tax_rate, gini,
                Omega_base_prev, y_damage_distribution_exponent, y_net_reference,
                Fmin_prev, Fmax_prev, y_gross_prev,
                uniform_redistribution_prev, uniform_tax_rate_prev, gini_prev,
            )

            # Store income values at quadrature points for next timestep
            yi = y_vals.copy()

            consumption_vals = y_vals * (1 - s)
            if eta == 1:
                utility_vals = np.log(consumption_vals)
            else:
                utility_vals = (consumption_vals ** (1 - eta)) / (1 - eta)
            aggregate_utility += 0.5 * interval_width * np.sum(wi * utility_vals)

        # Segment 3: High-income earners paying income-dependent tax [Fmax, 1]
        if 1.0 - Fmax > EPSILON:
            max_y_net = y_of_F_lagged_damage(
                Fmax,
                Fmin, Fmax,
                y_gross * (1 - uniform_tax_rate), uniform_redistribution_amount, uniform_tax_rate, gini,
                Omega_base_prev, y_damage_distribution_exponent, y_net_reference,
                Fmin_prev, Fmax_prev, y_gross_prev,
                uniform_redistribution_prev, uniform_tax_rate_prev, gini_prev,
            )
            max_consumption = max_y_net * (1 - s)

            # Damage for this segment (everyone has same income at Fmax)
            y_prev_at_Fmax = y_of_F_after_damage(
                Fmax, Fmin_prev, Fmax_prev, y_gross_prev, Omega_base_prev,
                y_damage_distribution_exponent, y_net_reference,
                uniform_redistribution_prev, gini_prev,
            )
            if np.abs(y_damage_distribution_exponent) < EPSILON:
                Omega_max = Omega_base_prev
            else:
                Omega_max = Omega_base_prev * (y_prev_at_Fmax / y_net_reference) ** y_damage_distribution_exponent

            aggregate_damage_fraction += (1 - Fmax) * Omega_max
            aggregate_utility += crra_utility_interval(Fmax, 1.0, max_consumption, eta)

        # Set Omega from aggregate damage (no iteration needed!)
        Omega = min(aggregate_damage_fraction, 1.0 - EPSILON)

        # Compute Omega_base for next timestep
        # If income_dependent_aggregate_damage is False, we scale Omega_base to match Omega_target
        # With lagged approach, Omega might not exactly equal Omega_target, but that's OK
        # For next timestep, we compute Omega_base from current Omega
        if income_dependent_damage_distribution and not income_dependent_aggregate_damage:
            # Compute expected value of (y/y_ref)^alpha to scale Omega_base
            # This gives us Omega_base such that E[Omega_base * (y/y_ref)^alpha] ≈ Omega_target at next timestep
            # For now, use simple scaling based on aggregate damage
            if Omega > EPSILON:
                # Omega_base for next step = Omega_target * Omega_base_current / Omega_current
                # But we want to use the target for next step, which is Omega (from temperature)
                # Actually, for next step we want Omega_base such that aggregate gives Omega_target_next
                # Simplest approach: scale current Omega_base by ratio
                Omega_base = Omega_target * Omega_base_prev / Omega if Omega > EPSILON else Omega_target
            else:
                Omega_base = Omega_target

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
            'gini': gini,  # Background Gini for reference
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

    # Always return current state for use as previous_step_values in next time step
    results['current_income_dist'] = {
        'y_mean': y_gross,
        'y_net': y_net,
        'gini': gini,
        'Omega_base': Omega_base,
        'Fmin': Fmin,
        'Fmax': Fmax,
        'uniform_redistribution_amount': uniform_redistribution_amount,
        'uniform_tax_rate': uniform_tax_rate,
        'y_gross': y_gross,
        'y_net_reference': y_net_reference,
        'yi': yi,  # Income at quadrature points for next timestep's lagged damage
        'Climate_Damage': Climate_Damage,  # Total climate damage for next timestep
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

    # Create time array
    t_array = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t_array)

    # Precompute Gauss-Legendre quadrature nodes and weights (used for all timesteps)
    xi, wi = roots_legendre(N_QUAD)

    # Calculate initial state
    A0 = config.time_functions['A'](t_start)
    L0 = config.time_functions['L'](t_start)
    delta = config.scalar_params.delta
    alpha = config.scalar_params.alpha
    fract_gdp = config.scalar_params.fract_gdp

    # take abatement cost and initial climate damage into account for initial capital
    Ecum_initial = config.scalar_params.Ecum_initial
    params = evaluate_params_at_time(t_start, config)

    Gini = config.time_functions['gini'](t_start)
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
    Gini_initial = config.time_functions['gini'](t_start)

    # Compute initial damage from cumulative emissions
    delta_T_initial = k_climate * config.scalar_params.Ecum_initial
    psi1 = params['psi1']
    psi2 = params['psi2']
    Omega_initial = psi1 * delta_T_initial + psi2 * delta_T_initial**2
    Omega_base_initial = Omega_initial  # No income-dependent adjustment initially

    # Compute initial climate damage
    Y_gross_initial_damage = Y_gross_initial * Omega_initial

    previous_step_values = {
        'y_mean': y_gross_initial,
        'gini': Gini_initial,
        'Omega_base': Omega_base_initial,
        'Fmin': 0.0,
        'Fmax': 1.0,
        'uniform_redistribution_amount': 0.0,
        'uniform_tax_rate': 0.0,
        'y_gross': y_gross_initial,
        'y_net': y_gross_initial * (1.0 - Omega_initial),  # Approximate initial net income
        'y_net_reference': config.scalar_params.y_net_reference,
        'yi': None,  # Income at quadrature points (not available for first timestep)
        'Climate_Damage': Y_gross_initial_damage,  # Initial total climate damage
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
            'gini': np.zeros(n_steps),  # Background Gini
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
        outputs = calculate_tendencies(state, params, previous_step_values, xi, wi, store_detailed_output)

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
            results['gini'][i] = outputs['gini']  # Background Gini
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
