# COIN_equality

A simple-as-possible stylized representation of the tradeoff between investment in income redistribution versus investment in emissions abatement.

## Table of Contents

- [Overview](#overview)
- [Model Structure](#model-structure)
  - [Objective Function](#objective-function)
  - [Calculation Order](#calculation-order)
  - [Core Components](#core-components)
- [Key Parameters](#key-parameters)
  - [Scalar Parameters (Time-Invariant)](#scalar-parameters-time-invariant)
  - [Time-Dependent Functions](#time-dependent-functions)
  - [Control Variables](#control-variables)
  - [Integration Parameters](#integration-parameters)
- [Model Features](#model-features)
  - [Simplifying Assumptions](#simplifying-assumptions)
  - [Key Insights](#key-insights)
- [Implementation: Key Functions](#implementation-key-functions)
- [Parameter Organization](#parameter-organization)
  - [Configuration File Structure](#configuration-file-structure)
  - [Example Configuration](#example-configuration)
  - [Loading Configuration](#loading-configuration)
  - [Testing the Forward Model](#testing-the-forward-model)
  - [Running Optimizations with Parameter Overrides](#running-optimizations-with-parameter-overrides)
  - [Running Multiple Optimizations in Parallel](#running-multiple-optimizations-in-parallel)
  - [Comparing Multiple Optimization Results](#comparing-multiple-optimization-results)
- [Time Integration](#time-integration)
  - [Integration Function](#integration-function)
  - [Implementation Notes](#implementation-notes)
  - [Performance Optimizations](#performance-optimizations)
  - [Output Variables](#output-variables)
- [Output and Visualization](#output-and-visualization)
  - [Saving Results](#saving-results)
  - [Output Files](#output-files)
  - [Example Workflow](#example-workflow)
- [Optimization Configuration](#optimization-configuration)
  - [Direct Multi-Point Optimization](#direct-multi-point-optimization)
  - [Iterative Refinement Optimization](#iterative-refinement-optimization)
  - [Optimization Stopping Criteria](#optimization-stopping-criteria)
  - [Dual Optimization (f and s)](#dual-optimization-f-and-s)
- [Dual Optimization of Savings Rate and Abatement Allocation](#dual-optimization-of-savings-rate-and-abatement-allocation)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)
- [Authors](#authors)

## Overview

This project develops a highly stylized model of an economy with income inequality, where a specified fraction of gross production is allocated to social good. The central question is how to optimally allocate resources between two competing objectives:

1. **Income redistribution** - reducing inequality by transferring income from high-income to low-income individuals
2. **Emissions abatement** - reducing carbon emissions to mitigate future climate damage

The model extends the COIN framework presented in [Caldeira et al. (2023)](https://doi.org/10.1088/1748-9326/acf949) to incorporate income inequality and diminishing marginal utility of income.

## Model Structure

### Objective Function

The model optimizes the time-integral of aggregate utility by choosing the allocation fraction `f(t)` between emissions abatement and income redistribution:

```
max∫₀^∞ e^(-ρt) · U(t) · L(t) dt,  subject to 0 ≤ f(t) ≤ 1
```

where:
- `ρ` = pure rate of time preference
- `U(t)` = mean utility of the population at time t
- `L(t)` = population at time t
- `f(t)` = fraction of resources allocated to abatement (control variable)

### Calculation Order

The differential equation solver uses climate damage from the previous timestep to avoid circular dependencies. Variables are calculated in this order:

**Setup and Previous Damage:**
1. **Fwi** = wi / 2.0 (transform quadrature weights from xi-space [-1,1] to F-space [0,1])
2. **Climate_Damage_prev** from previous timestep's Climate_Damage_yi using Fwi-weighted sum
3. **climate_damage_yi_prev** = Climate_Damage_prev / L (per-capita damage at each quadrature point)

**Current Timestep Production:**
4. **Y_gross** from K, L, A, α (Eq 1.1: Cobb-Douglas production)
5. **ΔT** from Ecum, k_climate (Eq 2.2: temperature from cumulative emissions)
6. **y_gross** from Y_gross, L (mean per-capita gross income before climate damage)
7. **Omega** = Climate_Damage_prev / Y_gross (climate damage fraction using previous damage)
8. **Omega_base** from ΔT, psi1, psi2 (base climate damage from temperature for next timestep)

**Tax and Redistribution (using previous damage):**
9. **redistribution_amount** from fract_gdp, f, y_gross, Omega (total per-capita redistribution)
10. **Fmin, uniform_redistribution_amount** from income_dependent_redistribution_policy:
   - If income-dependent: find_Fmin_analytical() computes critical rank using analytical Lorenz integrals
   - If uniform: Fmin = 0, uniform_redistribution = redistribution_amount
11. **Fmax, uniform_tax_rate** from income_dependent_tax_policy:
   - If income-dependent: find_Fmax_analytical() computes critical rank using analytical Lorenz integrals
   - If uniform: Fmax = 1.0, uniform_tax_rate = (abateCost + redistribution) / (y_gross * (1 - Omega))

**Segment-wise Integration (F-space with Fi_edges):**
12. **Fi, Fi_edges** = transform xi, xi_edges from [-1,1] to [0,1]
13. **Segment 1 [0, Fmin]**: Low-income earners receiving redistribution
   - Set y_net_yi for bins below/containing Fmin with proper weighting
   - Calculate aggregate_utility using crra_utility_interval()
   - Calculate climate_damage_yi for next timestep
14. **Segment 3 [Fmax, 1]**: High-income earners paying progressive tax
   - Set y_net_yi for bins above/containing Fmax with proper weighting
   - Calculate aggregate_utility using crra_utility_interval()
   - Calculate climate_damage_yi for next timestep
15. **Segment 2 [Fmin, Fmax]**: Middle-income earners with uniform tax/redistribution
   - Calculate y_vals_Fi at quadrature points using y_of_F_after_damage()
   - Set y_net_yi and climate_damage_yi for bins in/overlapping [Fmin, Fmax]
   - Calculate aggregate_utility using Gauss-Legendre quadrature

**Downstream Calculations:**
13. **Y_damaged** from Y_gross, Omega (Eq 1.3: production after climate damage)
14. **y_damaged** from y_gross, Omega (per-capita gross production after climate damage)
15. **AbateCost** from f, fract_gdp, Y_damaged (Eq 1.5: abatement expenditure)
16. **Y_net** from Y_damaged, AbateCost (Eq 1.8: production after abatement costs)
17. **y_net** from y_damaged, abateCost_mean (Eq 1.9: effective per-capita income)
18. **E_pot** from σ, Y_gross (Eq 2.1: potential emissions)
19. **μ** from AbateCost, θ₁, θ₂, E_pot (Eq 1.6: fraction of emissions abated, capped at μ_max)
20. **E** from σ, μ, Y_gross (Eq 2.3: actual emissions after abatement)
21. **U** from aggregate_utility (mean utility from integration)
22. **dK/dt** from s, Y_net, δ, K (Eq 1.10: capital tendency)
23. **current_income_dist** with y_mean = y_net (for next time step's damage calculation)

### Core Components

#### 1. Economic Model (Solow-Swann Growth)

**Eq. (1.1) - Production Function (Cobb-Douglas):**
```
Y_gross(t) = A(t) · K(t)^α · L(t)^(1-α)
```

**Eq. (1.2) - Income-Dependent Climate Damage:**

**Income Distribution:**
For a Pareto income distribution with parameter `a > 1`:
```
y(F) = ȳ · (1 - 1/a) · (1-F)^(-1/a),  F ∈ [0,1]
```
where `F` is the population fraction (poorest), `ȳ` is mean income, and pre-damage Gini is `G₀ = 1/(2a-1)`.

**Damage Function (Power-Law Model):**
```
Ω_base(ΔT) = psi1 · ΔT + psi2 · ΔT²  [Barrage & Nordhaus 2023]
ω(y) = Ω_base · (y / y_net_reference)^y_damage_distribution_exponent
```
where:
- `Ω_base` is the base damage fraction from temperature (before income-dependent adjustment)
- `y` is per-capita income
- `y_net_reference` is the reference income level for damage normalization
- `y_damage_distribution_exponent` is the power-law exponent controlling income-dependent damage scaling
- At income `y = y_net_reference`: damage = `Ω_base` (reference damage level)
- For `y_damage_distribution_exponent > 0`: damage increases with income (progressive damage)
- For `y_damage_distribution_exponent < 0`: damage decreases with income (regressive damage)
- For `y_damage_distribution_exponent = 0`: damage is uniform across income levels

**Aggregate Damage Calculation:**
The aggregate damage fraction is computed numerically by integrating over the income distribution:
```
aggregate_damage_fraction = ∫₀¹ ω(y(F)) dF
Ω = aggregate_damage_fraction
```
This integration is performed using three-segment Gauss-Legendre quadrature over [0, Fmin), [Fmin, Fmax), and [Fmax, 1], where:
- Fmin = maximum income rank receiving targeted redistribution
- Fmax = minimum income rank paying progressive taxation

**Lagged Damage Approach:**
The model uses climate damage from the previous timestep to avoid circular dependencies:
1. Start with Omega_base from temperature: Ω_base = psi1·ΔT + psi2·ΔT²
2. Use previous timestep's income distribution to compute current damage
3. This allows explicit (non-iterative) calculation of income distribution
4. Climate damage is updated each timestep based on the previous period's economic state

**Physical Interpretation:**
- As `y_damage_distribution_exponent → 0`: damage becomes uniform across income (no income effect)
- For `y_damage_distribution_exponent > 0`: higher-income populations experience more damage (progressive)
- For `y_damage_distribution_exponent < 0`: lower-income populations experience more damage (regressive)
- As `ΔT → 0`: `Ω_base → 0` and `Ω → 0` (no damage)

**Implementation:**
The damage integrals are computed using Gauss-Legendre quadrature (N_QUAD = 32 points) with analytical Lorenz curve integration and stepwise functions from `distribution_utilities.py`. Income at each quadrature point is computed using `y_of_F_lagged_damage()`, which uses damage from the previous timestep, eliminating the need for iterative convergence. The critical ranks Fmin and Fmax are found using `find_Fmin_analytical()` and `find_Fmax_analytical()`, which employ closed-form Lorenz integrals for ~1000× speedup over numerical quadrature.

**Eq. (1.3) - Damaged Production:**
```
Y_damaged(t) = (1 - Ω(t)) · Y_gross(t)
```
This is production after accounting for climate damage but before abatement costs.

**Eq. (1.4) - Mean Per-Capita Income:**
```
y(t) = (1 - s) · Y_damaged(t) / L(t)
```

**Eq. (1.5) - Abatement Cost:**
```
AbateCost(t) = f · c_redist(t) · L(t)
```
This is the total amount society allocates to emissions abatement, where:
- `f` = fraction of redistributable resources allocated to abatement (0 ≤ f ≤ 1)
- `c_redist(t)` = per-capita amount of income available for redistribution
- `L(t)` = population

**Eq. (1.6) - Abatement Fraction:**
```
μ(t) = min(μ_max, [AbateCost(t) · θ₂ / (E_pot(t) · θ₁(t))]^(1/θ₂))
```
The fraction of potential emissions that are abated, where:
- `E_pot(t) = σ(t) · Y_gross(t)` = potential (unabated) emissions
- `θ₁(t)` = marginal cost of abatement as μ→1 ($ tCO₂⁻¹)
- `θ₂` = abatement cost exponent (θ₂=2 gives quadratic cost function)
- `μ_max` = maximum allowed abatement fraction (cap on μ)

The calculated μ is capped at μ_max. Values of μ_max > 1 allow for carbon dioxide removal (negative emissions). If μ_max is not specified in the configuration, it defaults to INVERSE_EPSILON (effectively no cap).

This formulation differs from Nordhaus in that reducing carbon intensity σ(t) reduces the cost of abating remaining emissions, since there are fewer emissions to abate.

**Eq. (1.7) - Abatement Cost Fraction:**
```
Λ(t) = AbateCost(t) / Y_damaged(t)
```
This represents the fraction of damaged production allocated to emissions abatement.

**Eq. (1.8) - Net Production:**
```
Y_net(t) = (1 - Λ(t)) · Y_damaged(t)
```
Production after both climate damage and abatement costs.

**Eq. (1.9) - Effective Per-Capita Income:**
```
y_net(t) = y(t) - AbateCost(t) / L(t)
```
This is the per-capita income after subtracting abatement costs, used for utility calculations.

**Eq. (1.10) - Capital Accumulation:**
```
dK/dt = s · Y_net(t) - δ · K(t)
```

#### 2. Climate Model

**Eq. (2.1) - Potential Emissions:**
```
E_pot(t) = σ(t) · Y_gross(t)
```
This is the emissions rate without any abatement.

**Eq. (2.2) - Temperature Change:**
```
ΔT(t) = k_climate · ∫₀^t E(t') dt'
       = k_climate · Ecum(t)
```
Temperature change is proportional to cumulative carbon dioxide emissions.

**Eq. (2.3) - Actual Emissions:**
```
E(t) = σ(t) · (1 - μ(t)) · Y_gross(t)
     = (1 - μ(t)) · E_pot(t)
```
This is the actual emissions rate after abatement.

#### 3. Income Distribution and Utility

**Eq. (3.1) - Pareto-Lorenz Distribution:**
```
ℒ(F) = 1 - (1 - F)^(1-1/a)
```

where `F` is the fraction of the population with the lowest incomes.

**Eq. (3.2) - Gini Index:**
```
G = 1/(2a - 1)
a = (1 + 1/G)/2
```

**Eq. (3.3) - Income at Rank F:**
```
c(F) = y · (1 - 1/a) · (1 - F)^(-1/a)
```

**Eq. (3.4) - Isoelastic Utility Function (CRRA):**
```
u(c) = (c^(1-η) - 1)/(1 - η)  for η ≠ 1
u(c) = ln(c)                   for η = 1
```

where `η` is the coefficient of relative risk aversion.

**Eq. (3.5) - Mean Population Utility:**
```
U = [y^(1-η)/(1-η)] · [(1+G)^η(1-G)^(1-η)/(1+G(2η-1))]^(1/(1-η))  for η ≠ 1
U = ln(y) + ln((1-G)/(1+G)) + 2G/(1+G)                              for η = 1
```

#### 4. Redistribution Mechanics

The model considers allocation of resources between income redistribution and emissions abatement. The key parameters are:
- `G₁` = initial Gini index
- `f_gdp` = fraction of total income to be redistributed (specified exogenously)
- `f` = fraction of redistributable resources allocated to abatement (0 ≤ f ≤ 1)

**Eq. (4.1) - Fraction of Income Redistributed:**

Given `f_gdp` and `G₁`, we numerically solve for `G₂` (the Gini index after full redistribution) using the relationship:
```
f_gdp(F*) = [2(G₁-G₂)/(1-G₁)(1+G₂)] · [((1+G₁)(1-G₂))/((1-G₁)(1+G₂))]^((1+G₁)(1-G₂)/(2(G₂-G₁)))
```
where `F*` is the crossing rank (see Eq. 4.2).

**Eq. (4.2) - Crossing Rank:**

The population rank where income remains unchanged during redistribution:
```
F* = 1 - [((1+G₁)(1-G₂))/((1-G₁)(1+G₂))]^(((1+G₁)(1+G₂))/(2(G₂-G₁)))
```

**Eq. (4.3) - Per-Capita Amount Redistributed:**
```
c_redist = y · f_gdp
```
where `y` is mean per-capita income.

**Fraction of Emissions Abated:**

See Eq. (1.6) above. The abatement fraction is determined by the amount society allocates to abatement relative to potential emissions and the marginal abatement cost.

**Enhanced Redistribution Mode (f_gdp >= 1) - Partial Implementation**

The model now supports `f_gdp >= 1` with special handling that disables redistribution and allows pure abatement optimization.

**Current Behavior (f_gdp < 1):**
- Redistribution operates via income-dependent or uniform policies
- Control variable `f` determines allocation between abatement and redistribution
- Climate damage calculations integrate over the income distribution using power-law damage function
- Critical income ranks (Fmin, Fmax) define segments for progressive taxation/redistribution

**Implementation Details:**
The model uses a lagged damage approach for explicit (non-iterative) calculation:

1. **Climate Damage with Income Distribution**:
   - Uses previous timestep's damage to compute current income distribution
   - Integrates damage over three income segments: [0, Fmin), [Fmin, Fmax), [Fmax, 1]
   - Updates Omega for use in next timestep, eliminating within-timestep convergence loops

2. **Critical Income Ranks**:
   - **Fmin**: Maximum rank receiving income-dependent redistribution (if enabled)
   - **Fmax**: Minimum rank paying income-dependent tax (if enabled)
   - Computed via root-finding to match budget constraints

3. **Abatement Budget Mechanics**:
   - Available budget: `redistribution = y * delta_L` (Line 136 of `economic_model.py`)
   - With `delta_L >= 1`, this creates `redistribution >= y` (budget at least equals full per-capita income)
   - Abatement expenditure: `AbateCost = f * redistribution * L` (Line 142)
   - Effective income: `y_net = y - AbateCost/L = y - f * redistribution`

4. **Optimizer Behavior**:
   - The optimizer chooses `f` to maximize utility over time
   - **Naturally selects `f << 1`** because:
     - Large `f` would make `y_net = y - f * redistribution` very small or negative
     - This would result in terrible current utility (Consumption crash)
     - Optimizer balances current Consumption vs. future climate benefits
   - **Equivalence**: Optimization of `f` becomes equivalent to optimizing the abatement/Consumption tradeoff
   - No redistribution component in utility calculation (since `G_eff = Gini_climate`)

5. **Physical Interpretation**:
   - `f_gdp >= 1` represents a model mode where redistribution is turned off
   - Allows studying pure abatement policy without redistribution considerations
   - Budget parameter `f_gdp` scales the available resources, but optimizer self-limits via utility constraints
   - Climate damage treated as uniform across income levels (first-order approximation)

**Implementation Status**:
- ✓ Redistribution disabled in `economic_model.py` when income_redistribution = False
- ✓ Uniform damage when y_damage_distribution_exponent → 0 (power-law → 1)
- ✓ Uses `INVERSE_EPSILON` constant from `constants.py` (no hardcoded values)
- ✓ All existing unit tests pass

#### 5. Gini Index Dynamics and Persistence

The Gini index consists of two components: an **exogenous background** `gini(t)` and an **endogenous perturbation** `delta_Gini(t)` that evolves over time.

**Decomposition:**
```
Gini(t) = gini(t) + delta_Gini(t)
```
where:
- `gini(t)`: Exogenously specified time function (e.g., demographic trends, structural inequality)
- `delta_Gini(t)`: Endogenous state variable representing policy-driven perturbations from background

**State Variable:**
```
delta_Gini(t) - Perturbation from background Gini index
```

**Perturbation Evolution:**

The perturbation evolves through two mechanisms:

1. **Instantaneous Step Change** (fraction of policy effect applied immediately):
```
delta_Gini_step_change = Gini_fract · (G_eff - Gini)
```
where:
- `Gini = gini(t) + delta_Gini` is the current total Gini
- `G_eff` is the effective Gini from current policy (redistribution/abatement allocation)
- `Gini_fract` is the fraction of the change applied as an immediate step (0 ≤ Gini_fract ≤ 1)
- `Gini_fract = 0`: no immediate effect (fully persistent system)
- `Gini_fract = 1`: full immediate effect (no persistence)
- `Gini_fract = 0.1`: 10% of policy effect occurs immediately

2. **Continuous Restoration** (gradual return to background):
```
d(delta_Gini)/dt = -Gini_restore · delta_Gini
```
where:
- `Gini_restore` is the restoration rate (yr⁻¹)
- `Gini_restore = 0`: no restoration (persistent policy effects)
- `Gini_restore > 0`: exponential decay toward background inequality
- `Gini_restore = 0.1`: 10% per year restoration rate (timescale ~10 years)

**Combined Update Rule:**
```
delta_Gini(t+dt) = delta_Gini(t) + dt · d(delta_Gini)/dt + delta_Gini_step_change
```

**Physical Interpretation:**

This formulation cleanly separates exogenous and endogenous inequality dynamics:
- **Exogenous background** (`gini(t)`): Captures structural inequality trends (demographics, technology, institutions) specified externally
- **Endogenous perturbation** (`delta_Gini`): Captures policy-driven deviations from background that restore to zero
- **Policy pressure** (via `delta_Gini_step_change`): Redistribution policies create perturbations from background
- **Structural restoration** (via `d(delta_Gini)/dt`): Absent continued intervention, perturbations decay back to zero

The `Gini_fract` parameter controls the **speed of policy effect**:
- Small `Gini_fract`: Policy effects build up gradually (high persistence/inertia)
- Large `Gini_fract`: Policy effects manifest quickly (low persistence/inertia)

The `Gini_restore` parameter controls the **persistence of achieved changes**:
- Small `Gini_restore`: Perturbations are long-lasting
- Large `Gini_restore`: Perturbations decay quickly without continued policy pressure

**Initial Condition:**
```
delta_Gini(0) = 0
```
The system begins at the background inequality level.

**Climate Damage Interaction:**

Climate damage is computed using the income distribution from the previous timestep to avoid circular dependencies. The algorithm integrates damage over the income distribution accounting for:
- Income-dependent damage function: damage(y) = omega_base * (y / y_net_reference)^y_damage_distribution_exponent
- Progressive taxation and targeted redistribution via critical ranks (Fmin, Fmax)
- Analytical Lorenz integrals for finding critical ranks (Fmin, Fmax) with ~1000× speedup over numerical quadrature

The effective income distribution evolves through:
```
previous y_net → (damage + tax + redistribution) → current y_net → (save for next timestep)
```

## Key Parameters

Parameters are organized into groups as specified in the JSON configuration file.

### Scalar Parameters (Time-Invariant)

Economic parameters:

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| `α` | Output elasticity of capital (capital share of income) | - | `alpha` |
| `δ` | Capital depreciation rate | yr⁻¹ | `delta` |
| `s` | Savings rate (fraction of net production saved) | - | `s` |

Climate and abatement parameters:

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| `psi1` | Linear climate damage coefficient [Barrage & Nordhaus 2023] | °C⁻¹ | `psi1` |
| `psi2` | Quadratic climate damage coefficient [Barrage & Nordhaus 2023] | °C⁻² | `psi2` |
| `y_net_reference` | Reference income level for climate damage normalization | $ | `y_net_reference` |
| `y_damage_distribution_exponent` | Power-law exponent for income-dependent damage scaling (>0: progressive, <0: regressive, =0: uniform) | - | `y_damage_distribution_exponent` |
| `k_climate` | Temperature sensitivity to cumulative emissions | °C tCO₂⁻¹ | `k_climate` |
| `θ₂` | Abatement cost exponent (controls cost curve shape) | - | `theta2` |
| `μ_max` | Maximum allowed abatement fraction (cap on μ). Values >1 allow carbon removal. Defaults to INVERSE_EPSILON (no cap) if omitted. | - | `mu_max` |
| `Ecum_initial` | Initial cumulative CO2 emissions. Defaults to 0.0 (no prior emissions) if omitted. | tCO₂ | `Ecum_initial` |

Utility and inequality parameters:

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| `η` | Coefficient of relative risk aversion (CRRA) | - | `eta` |
| `ρ` | Pure rate of time preference | yr⁻¹ | `rho` |
| `Gini_fract` | Fraction of effective Gini change as instantaneous step (0 = no step, 1 = full step) | - | `Gini_fract` |
| `Gini_restore` | Rate at which delta_Gini restores to zero (0 = no restoration) | yr⁻¹ | `Gini_restore` |
| `f_gdp` | Fraction of income available for redistribution (<1: active redistribution; >=1: redistribution disabled, pure abatement mode) | - | `fract_gdp` |

### Time-Dependent Functions

These functions are evaluated at each time step:

| Function | Description | Units | JSON Key |
|----------|-------------|-------|----------|
| `A(t)` | Total factor productivity | - | `A` |
| `L(t)` | Population | people | `L` |
| `σ(t)` | Carbon intensity of GDP | tCO₂ $⁻¹ | `sigma` |
| `θ₁(t)` | Marginal abatement cost as μ→1 | $ tCO₂⁻¹ | `theta1` |
| `gini(t)` | Background Gini index (exogenous inequality baseline) | - | `gini` |

Each function is specified by `type` and type-specific parameters (e.g., `exponential_scaling`, `growth_rate`). Six function types are available: `constant`, `exponential_growth`, `logistic_growth`, `piecewise_linear`, `double_exponential_growth` (Barrage & Nordhaus 2023), and `gompertz_growth` (Barrage & Nordhaus 2023). See the Configuration section below for detailed specifications.

### Control Variables

The model supports two control variables that can be optimized:

| Variable | Description | Units | JSON Key | Status |
|----------|-------------|-------|----------|---------|
| `f(t)` | Fraction of redistributable resources allocated to abatement | - | `control_function` | Required |
| `s(t)` | Savings rate (fraction of net production invested) | - | `s` in `time_functions` or `s_control_function` | Required |

**f(t) - Abatement Allocation:** Determines the allocation between emissions abatement and income redistribution (0 = all to redistribution, 1 = all to abatement). Always specified via `control_function` in the configuration.

**s(t) - Savings Rate:** Determines the fraction of net output allocated to investment vs. Consumption. Can be specified in two ways:
- **Fixed/Prescribed s(t):** Defined in `time_functions['s']` using any time function type (constant, piecewise_linear, etc.)
- **Optimized s(t):** Defined in `s_control_function` to enable dual optimization of both f(t) and s(t)

When both `control_function` and `s_control_function` are present, the model operates in **dual optimization mode**, allowing simultaneous optimization of the abatement-redistribution tradeoff and the Consumption-investment tradeoff.

### Policy Switches

The model has 5 boolean switches controlling different policy features. Three are **primary switches** and two are **sub-switches** that are only meaningful when their parent switch is enabled:

#### Primary Switches

1. **`income_dependent_damage_distribution`** (boolean, default: true)
   - Controls whether climate damage varies by income level
   - When true: Poor people experience higher damage (exponential in income)
   - When false: Everyone experiences the same damage (uniform)
   - Parent switch for: `income_dependent_aggregate_damage`

2. **`income_redistribution`** (boolean, default: true)
   - Controls whether income redistribution is enabled
   - When true: Redistributes from high to low income according to `redistribution_amount`
   - When false: No redistribution occurs
   - Parent switch for: `income_dependent_redistribution_policy`

3. **`income_dependent_tax_policy`** (boolean, default: false)
   - Controls whether taxation is progressive or uniform
   - When true: Progressive tax using `find_Fmax()` to determine tax threshold
   - When false: Uniform tax rate applied to all income

#### Sub-Switches (Only Meaningful with Parent Enabled)

4. **`income_dependent_aggregate_damage`** (boolean, default: true)
   - **Parent switch: `income_dependent_damage_distribution`**
   - **Only meaningful when parent is true**
   - Controls how aggregate damage is calculated when damage varies by income
   - When true: Aggregate damage computed directly from income distribution
   - When false: Use uniform `Omega_base` from temperature (no income-dependent adjustment)
   - Code enforces: Only checked when `income_dependent_damage_distribution` is true

5. **`income_dependent_redistribution_policy`** (boolean, default: false)
   - **Parent switch: `income_redistribution`**
   - **Only meaningful when parent is true**
   - Controls whether redistribution is targeted or uniform
   - When true: Targeted to lowest incomes using `find_Fmin()` to determine threshold
   - When false: Uniform per-capita redistribution to all
   - Code enforces: Only checked when `income_redistribution` is true

**Important**: The code automatically enforces these dependencies. Sub-switches are only evaluated when their parent switch is enabled, ensuring logically consistent behavior.

### Integration Parameters

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| `t_start` | Start time for integration | yr | `t_start` |
| `t_end` | End time for integration | yr | `t_end` |
| `dt` | Time step for Euler integration | yr | `dt` |
| `rtol` | Relative tolerance (reserved for future use) | - | `rtol` |
| `atol` | Absolute tolerance (reserved for future use) | - | `atol` |

## Model Features

### Simplifying Assumptions

To maintain analytical tractability:
- Fixed Pareto-Lorenz income distribution (parameterized by Gini index)
- Proportional relationship between temperature and cumulative emissions
- Power-law relationships for climate damage and abatement costs
- No distinction between population and labor force
- Exogenous technological progress `A(t)` and population `L(t)`

### Key Insights

1. **Redistribution vs. Climate Action Tradeoff**: Resources allocated to income redistribution provide immediate utility gains (especially with high `η`), while emissions abatement provides future benefits by reducing climate damage.

2. **Diminishing Marginal Utility**: Higher values of `η` mean that redistributing income from rich to poor has greater utility benefits, favoring redistribution over abatement.

3. **Time Preference**: Higher discount rates (`ρ`) favor immediate redistribution over long-term climate benefits.

4. **Income Distribution Mechanics**: Taxing the wealthy reduces the Gini index even if revenues are allocated to abatement rather than redistribution, but only redistribution increases current aggregate utility.

## Implementation: Key Functions

The `distribution_utilities.py` module provides the core mathematical functions for calculating income distribution, utility integration, and stepwise interpolation over quadrature intervals.

### Basic Conversion Functions

- **`a_from_G(G)`** - Converts Gini index to Pareto distribution parameter `a` using equation (4)
- **`L_pareto(F, G)`** - Calculates Lorenz curve value at population fraction `F` for a given Gini index (equation 2)

### Redistribution Mechanics

- **`crossing_rank_from_G(Gini_initial, G2)`** - Computes the population rank `F*` where income remains unchanged during redistribution from `Gini_initial` to `G2` (equation 10)

### Inverse Problem: Finding G2 from f_gdp

- **`_phi(r)`** - Helper function for numerical root finding; computes `φ(r) = (r-1) · r^(1/(r-1)-1)` with proper handling of edge cases

- **`G2_from_deltaL(deltaL, Gini_initial)`** - **Solves the inverse problem**: given an initial Gini `Gini_initial` and a desired redistribution amount `f_gdp`, numerically finds the target Gini `G2` that would result from full redistribution. Uses `scipy.optimize.root_scalar` with Brent's method. Returns `(G2, remainder)` where remainder is non-zero if `f_gdp` exceeds the maximum possible for the Pareto family (caps at G2=0).

### Climate Damage and Income Distribution Functions

**Income After Damage:**
- **`y_of_F_after_damage(F, Fmin, Fmax, y_mean_before_damage, omega_base, y_damage_distribution_exponent, y_net_reference, uniform_redistribution, gini, branch=0)`** - Computes income at rank F accounting for climate damage. For exponent = 0.5, uses analytic solution via quadratic formula. For other exponents, uses numerical root finding (scipy.optimize.fsolve) to solve the implicit equation where damage depends on income which depends on damage. Returns array of incomes corresponding to input ranks.

**Critical Rank Finding (Analytical):**
- **`find_Fmax_analytical(Fmin, y_gross, gini, damage_yi, Fi_edges, uniform_redistribution, target_tax, tol=LOOSE_EPSILON)`** - Finds the minimum income rank that pays progressive taxation using analytical Lorenz curve integration. Uses stepwise_interpolate() and stepwise_integrate() for damage terms. Much faster than quadrature-based approach. Returns Fmax ∈ [0, 1].

- **`find_Fmin_analytical(y_gross, gini, damage_yi, Fi_edges, uniform_redistribution, target_subsidy, tol=LOOSE_EPSILON)`** - Finds the maximum income rank that receives targeted redistribution using analytical Lorenz curve integration. Uses stepwise functions for damage terms. Returns Fmin ∈ [0, 1].

**Stepwise Interpolation and Integration:**
- **`stepwise_interpolate(F, yi, Fi_edges)`** - Evaluates a stepwise (piecewise constant) function at point(s) F. Returns yi[i] for F in [Fi_edges[i], Fi_edges[i+1]).

- **`stepwise_integrate(F0, F1, yi, Fi_edges)`** - Integrates a stepwise function from F0 to F1. Handles bins that partially overlap the integration range.

**Numerical Integration:**
- **`crra_utility_integral_with_damage(F0, F1, Fmin, Fmax_for_clip, y_mean_before_damage, omega_base, y_damage_distribution_exponent, y_net_reference, uniform_redistribution, gini, eta, s, xi, wi, branch=0)`** - Integrates CRRA utility over income ranks [F0, F1] using Gauss-Legendre quadrature. Accounts for climate damage via y_of_F_after_damage().

- **`climate_damage_integral(F0, F1, Fmin, Fmax_for_clip, y_mean_before_damage, omega_base, y_damage_distribution_exponent, y_net_reference, uniform_redistribution, gini, xi, wi, branch=0)`** - Integrates climate damage over income ranks [F0, F1] using Gauss-Legendre quadrature. Damage function: omega_base * (income / y_net_reference)^y_damage_distribution_exponent.

- **`crra_utility_interval(F0, F1, c_mean, eta)`** - Utility of constant consumption c over interval [F0, F1]. Used for flat segments (below Fmin or above Fmax) where all individuals have same income.

### Usage Example

```python
from distribution_utilities import (
    y_of_F_after_damage,
    find_Fmax_analytical,
    find_Fmin_analytical,
    stepwise_interpolate,
    stepwise_integrate,
    crra_utility_integral_with_damage
)
from scipy.special import roots_legendre
import numpy as np

# Setup quadrature
xi, wi = roots_legendre(32)
xi_edges = -1.0 + np.concatenate(([0.0], np.cumsum(wi)))
Fi_edges = (xi_edges + 1.0) / 2.0

# Find critical ranks using analytical method
Fmax = find_Fmax_analytical(Fmin, y_gross, gini, damage_yi, Fi_edges,
                            uniform_redist, target_tax=1000.0)
Fmin = find_Fmin_analytical(y_gross, gini, damage_yi, Fi_edges,
                            uniform_redist, target_subsidy=500.0)

# Compute utility in middle segment
U_middle = crra_utility_integral_with_damage(Fmin, Fmax, Fmin, Fmax, y_mean,
                                            Omega_base, y_scale, uniform_redist, gini, eta, xi, wi)
```

## Parameter Organization

The model uses JSON configuration files to specify all parameters. Configuration is loaded via `load_configuration(config_path)` in `parameters.py`.

### Configuration File Structure

Each JSON configuration file must contain:

1. **`run_name`** - String identifier used for output directory naming
2. **`description`** - Optional description of the scenario
3. **`scalar_parameters`** - Time-invariant model constants:
   - Economic: `alpha`, `delta`, `s`
   - Climate: `psi1`, `psi2`, `y_net_reference`, `y_damage_distribution_exponent`, `k_climate`
   - Utility: `eta`, `rho`
   - Distribution: `Gini_initial`, `Gini_fract`, `Gini_restore`, `fract_gdp`

4. **`time_functions`** - Time-dependent functions (A, L, sigma, theta1), each specified with:
   - `type`: One of six available function types (see details below)
   - Type-specific parameters (e.g., `initial_value`, `growth_rate`)

   **Available Time Function Types:**

   a. **`constant`** - Returns fixed value for all times
      - Parameters: `value`
      - Equation: `f(t) = value`

   b. **`exponential_growth`** - Exponential growth or decay
      - Parameters: `initial_value`, `growth_rate`
      - Equation: `f(t) = initial_value · exp(growth_rate · t)`

   c. **`logistic_growth`** - S-curve growth approaching asymptotic limit
      - Parameters: `L0` (initial), `L_inf` (limit), `growth_rate`
      - Equation: `f(t) = L_inf / (1 + ((L_inf/L0) - 1) · exp(-growth_rate · t))`

   d. **`piecewise_linear`** - Linear interpolation between discrete points
      - Parameters: `time_points` (array), `values` (array)
      - Equation: Linear interpolation between (time_points, values)

   e. **`double_exponential_growth`** - Weighted sum of two exponentials (Barrage & Nordhaus 2023)
      - Parameters: `initial_value`, `growth_rate_1`, `growth_rate_2`, `fract_1`
      - Equation: `f(t) = initial_value · (fract_1 · exp(growth_rate_1 · t) + (1 - fract_1) · exp(growth_rate_2 · t))`
      - **Purpose**: Models carbon intensity (sigma) with fast initial decline transitioning to slower long-term decline
      - **Typical values**: Curve fit to DICE2023 parameters:
        - `growth_rate_1 = -0.015` (fast initial decarbonization)
        - `growth_rate_2 = -0.005` (slower asymptotic decline)
        - `fract_1 = 0.70` (70% weight on fast decline)

   f. **`gompertz_growth`** - Gompertz growth function (continuous form of Barrage & Nordhaus 2023 finite-difference model)
      - Parameters: `initial_value`, `final_value`, `adjustment_coefficient`
      - Equation: `L(t) = final_value · exp(ln(initial_value / final_value) · exp(adjustment_coefficient · t))`
      - **Purpose**: Models population growth approaching asymptotic limit
      - **Properties**: At t=0: L(0) = initial_value; as t→∞: L(t) → final_value (for negative adjustment_coefficient)
      - **Note**: This form using exp/log has better numerical properties than the equivalent power form `(initial_value / final_value)^exp(...)`
      - **Typical values**: Based on DICE2023 parameters:
        - `initial_value = 7.0e9` (7 billion people)
        - `final_value = 10.0e9` (10 billion asymptotic limit)
        - `adjustment_coefficient = -0.02` (controls approach rate to limit)

5. **`integration_parameters`** - Solver configuration:
   - `t_start`, `t_end`, `dt`, `rtol`, `atol`

6. **`control_function`** - Allocation policy f(t):
   - `type`: "constant" or "piecewise_constant"
   - Type-specific parameters (e.g., `value` for constant)

See `config_baseline.json` for extensive examples of documentation.

### Example Configuration

See `config_baseline.json` for a complete example. To create new scenarios, copy and modify this file.

**Note**: `config_DICE_000.json` provides a configuration for simulations close to the parameters and setup presented in Barrage & Nordhaus (2023), including Gompertz population growth, double exponential functions for carbon intensity and abatement costs, and settings that replicate DICE2023 behavior (income_redistribution = false for pure abatement mode, gini = 0.0 for no inequality).

**Example: Population with Gompertz growth**
```json
"L": {
  "type": "gompertz_growth",
  "initial_value": 7.0e9,
  "final_value": 10.0e9,
  "adjustment_coefficient": -0.02
}
```

**Example: Carbon intensity with double exponential decline**
```json
"sigma": {
  "type": "double_exponential_growth",
  "initial_value": 0.0005,
  "growth_rate_1": -0.015,
  "growth_rate_2": -0.005,
  "fract_1": 0.70
}
```

**Example: TFP with simple exponential growth**
```json
"A": {
  "type": "exponential_growth",
  "initial_value": 454.174,
  "growth_rate": 0.01
}
```

### Loading Configuration

```python
from parameters import load_configuration

config = load_configuration('config_baseline.json')
# config.run_name contains the run identifier
# config.scalar_params, config.time_functions, etc. are populated
```

The `evaluate_params_at_time(t, config)` helper combines all parameters into a dict for use with `calculate_tendencies()`.

### Testing the Forward Model

The project includes a comprehensive test script to verify the forward model integration and demonstrate the complete workflow from configuration loading through output generation.

#### Quick Start

To test the model with the baseline configuration:

```bash
python test_integration.py config_baseline.json
```

This command will:
1. Load the baseline configuration from `config_baseline.json`
2. Display key model parameters and setup information
3. Run the forward integration over the specified time period
4. Show detailed results summary (initial state, final state, changes)
5. Generate timestamped output directory with CSV data and PDF plots

#### Command Line Usage

The test script requires a configuration file argument:

```bash
python test_integration.py <config_file>
```

**Examples:**
```bash
# Test with baseline scenario
python test_integration.py config_baseline.json

# Test with high inequality scenario
python test_integration.py config_high_inequality.json

# Test with custom configuration
python test_integration.py my_custom_config.json
```

If you run the script without arguments, it will display usage instructions.

#### Understanding the Output

The test script provides detailed console output including:

- **Configuration Summary**: Run name, time span, key parameters
- **Integration Progress**: Confirmation of successful model execution
- **Results Summary**:
  - Initial state (t=0): all key variables at start
  - Final state (t=end): all key variables at end of simulation
  - Changes: percentage and absolute changes over simulation period
- **Output Files**: Paths to generated CSV and PDF files

#### Generated Files

Each test run creates a timestamped directory:
```
./data/output/{run_name}_YYYYMMDD-HHMMSS/
├── results.csv          # Complete time series data (all variables)
├── plots.pdf            # Multi-page charts organized by variable type
└── terminal_output.txt  # Console output from the run
```

The PDF contains four organized sections:
1. **Dimensionless Ratios** - Policy variables and summary outcomes
2. **Dollar Variables** - Economic flows and stocks
3. **Physical Variables** - Climate and emissions data
4. **Specified Functions** - Exogenous model inputs

#### Testing Different Scenarios

Create new test scenarios by copying and modifying configuration files:

```bash
# Copy baseline configuration
cp config_baseline.json config_my_test.json

# Edit parameters in config_my_test.json
# Then test with:
python test_integration.py config_my_test.json
```

This testing framework validates the complete model pipeline and provides immediate visual feedback on model behavior through the generated charts.

### Running Optimizations with Parameter Overrides

The optimization script supports command line parameter overrides, enabling automated parameter sweeps without creating multiple configuration files.

#### Command Line Override Syntax

Override any configuration parameter using dot notation:

```bash
python run_optimization.py config.json --key.subkey.value new_value
```

**Examples:**

```bash
# Override single parameter
python run_optimization.py config_baseline.json --scalar_parameters.alpha 0.35

# Override multiple parameters
python run_optimization.py config_baseline.json \
  --run_name "sensitivity_test" \
  --optimization_parameters.initial_guess 0.3 \
  --scalar_parameters.rho 0.015

# Override nested parameters
python run_optimization.py config_baseline.json \
  --time_functions.A.growth_rate 0.02 \
  --optimization_parameters.n_points_final 100
```

**Common overrides:**
- `--run_name <name>` - Set output directory name
- `--scalar_parameters.alpha <value>` - Capital share
- `--scalar_parameters.rho <value>` - Time preference rate
- `--scalar_parameters.eta <value>` - Risk aversion coefficient
- `--optimization_parameters.initial_guess <value>` - Starting point
- `--optimization_parameters.max_evaluations <value>` - Iteration budget
- `--optimization_parameters.n_points_final <value>` - Target control points
- `--time_functions.A.growth_rate <value>` - TFP growth rate

#### Automated Parameter Sweeps

The `run_initial_guess_sweep.py` script demonstrates automated testing across multiple parameter values:

```bash
python run_initial_guess_sweep.py config_baseline.json
```

This runs optimization 11 times with `initial_guess` values from 0.0 to 1.0 (step 0.1), automatically creating separate output directories for each run.

**Creating custom sweep scripts:**

```python
import subprocess

config_file = "config_baseline.json"

# Sweep over alpha values
for alpha in [0.25, 0.30, 0.35, 0.40]:
    cmd = [
        "python", "run_optimization.py", config_file,
        "--scalar_parameters.alpha", str(alpha),
        "--run_name", f"alpha_{alpha:.2f}"
    ]
    subprocess.run(cmd, check=True)
```

**Benefits of command line overrides:**
- No need to create dozens of nearly-identical JSON files
- Easy to script parameter sweeps in bash or Python
- Git-friendly: only baseline configs need version control
- Clear provenance: command documents what changed from baseline
- Composable: combine multiple overrides in one command

### Running Multiple Optimizations in Parallel

The `run_parallel.py` script enables launching multiple optimization jobs simultaneously, with each job running on its own CPU core. This is ideal for parameter sweeps or running multiple scenarios.

#### Parallel Execution

The script accepts file patterns (with wildcards) for JSON configuration files, plus optional parameter overrides:

```bash
python run_parallel.py <pattern1> [pattern2] [...] [--key value] [...]
```

**Examples:**

```bash
# Run all COIN equality configs in parallel
python run_parallel.py "config_COIN-equality_000*.json"

# Run specific configuration files
python run_parallel.py config_baseline.json config_sensitivity.json

# Run multiple patterns
python run_parallel.py "config_COIN*.json" "config_DICE*.json"

# Quick test with reduced evaluations (applied to all jobs)
python run_parallel.py "config_*.json" --optimization_params.max_evaluations 100

# Override multiple parameters
python run_parallel.py "config_*.json" --optimization_params.max_evaluations 100 --run_name quick_test
```

#### How It Works

- **Parallel execution**: All matching JSON files are launched simultaneously as separate Python processes
- **Independent cores**: Each optimization runs on its own CPU core
- **Parameter overrides**: Optional `--key value` pairs are applied to ALL jobs (useful for quick tests)
- **Terminal output**: Automatically saved to `terminal_output.txt` in each job's output directory
- **Non-blocking**: The script exits immediately after launching all jobs (does not wait for completion)

#### Monitoring and Controlling Jobs

The output directory and `terminal_output.txt` file are created at the start of each optimization run, allowing you to monitor progress in real-time:

```bash
# Find the most recent output directory for a run
ls -lt data/output/<run_name>_* | head -1

# Monitor progress in real-time (updates automatically)
tail -f data/output/<run_name>_YYYYMMDD-HHMMSS/terminal_output.txt

# View current progress
cat data/output/<run_name>_YYYYMMDD-HHMMSS/terminal_output.txt

# View running processes
ps aux | grep run_optimization
```

**Stopping jobs:**

```bash
# Kill a specific job by PID
kill <PID>

# Kill ALL run_optimization.py jobs at once
pkill -f run_optimization.py
```

Process IDs (PIDs) are displayed when jobs are launched. The terminal output file updates continuously as the optimization progresses, allowing you to track:
- Configuration loading and setup
- Optimization iterations and progress
- Function evaluations and objective values
- Final results and file generation

**Note:** The `pkill` command will terminate all running `run_optimization.py` processes, which is useful for stopping an entire parameter sweep but should be used with caution if you have multiple independent jobs running.

#### Typical Workflow

```bash
# 1. Quick test with reduced evaluations
python run_parallel.py "config_sensitivity_*.json" --optimization_params.max_evaluations 100

# 2. Monitor progress
watch -n 10 'ps aux | grep run_optimization | wc -l'

# 3. After test completes, run full optimization
python run_parallel.py "config_sensitivity_*.json"

# 4. After jobs complete, compare results
python compare_results.py "data/output/sensitivity_*/"
```

**Benefits:**
- Fully utilizes multi-core systems
- No need to wait for sequential completion
- Terminal output saved for each job
- Parameter overrides for quick testing
- Simple command-line interface

### Comparing Multiple Optimization Results

After running multiple optimizations (e.g., parameter sweeps or scenario comparisons), use the comparison tool to analyze and visualize differences across runs.

#### Running Comparisons

The `compare_results.py` script accepts unlimited directory paths with wildcard support:

```bash
python compare_results.py <path1> [path2] [path3] [...]
```

**Examples:**

```bash
# Compare all runs matching a pattern
python compare_results.py "data/output/test_*/"

# Compare specific directories
python compare_results.py data/output/baseline/ data/output/high_eta/

# Compare multiple patterns
python compare_results.py "data/output/alpha_*/" "data/output/rho_*/"
```

#### Comparison Outputs

The tool creates a timestamped directory `data/output/comparison_YYYYMMDD-HHMMSS/` containing three files:

1. **`optimization_comparison_summary.xlsx`** - Excel workbook with optimization metrics:
   - Sheet 1: "Directories" - list of all compared directories with case names
   - Sheet 2: "Objective" - objective values by iteration for each case
   - Sheet 3: "Evaluations" - function evaluation counts
   - Sheet 4: "Elapsed Time (s)" - computation time (if available)
   - Sheet 5: "Termination Status" - optimization termination reasons
   - Sheets 6+: "Iter N f(t)" - f control points for each iteration
     - Shows optimal f(t) trajectory (abatement allocation) at each iteration
     - Time in column A, f values for each case in subsequent columns
     - Allows comparing how the optimal control evolved across iterations
   - Additional sheets: "Iter N s(t)" - s control points (for dual optimization cases)
     - Shows optimal s(t) trajectory (savings rate) at each iteration
     - Only included if any case optimizes both f and s
   - Cases appear as columns, iterations as rows

2. **`results_comparison_summary.xlsx`** - Excel workbook with time series results:
   - Sheet 1: "Directories" - list of all compared directories
   - Sheets 2-28: One sheet per variable (27 model variables)
   - Each sheet has time in column A, one column per case for that variable
   - Variables match plots in PDF: economic, climate, abatement, inequality, and utility metrics
   - Includes Gini_climate: post-climate-damage inequality (before redistribution)
   - Includes marginal_abatement_cost: actual marginal cost at current mu, and theta1: marginal cost at mu=1

3. **`comparison_plots.pdf`** - PDF report with visualizations:
   - Page 1: Summary scatter plots (objective, time, evaluations)
   - Pages 2+: Time series overlays for all model variables (27 variables)
   - 16:9 landscape format optimized for screen viewing
   - Multi-line plots show different cases in different colors
   - For multi-case comparisons: unified legend in top-left position of each page (5 plots per page)
   - For single-case: 6 plots per page without legend

#### What Gets Compared

The tool compares data from two sources:

1. **`optimization_summary.csv`** - Optimization performance metrics:
   - Required in each result directory
   - Contains iteration-by-iteration optimization statistics

2. **`results.csv`** - Full model time series (optional):
   - If present, adds detailed time series comparisons to PDF
   - Includes all 27 model variables (economic, climate, inequality, etc.)
   - If missing, only optimization summary is compared

#### Example Workflow

**Sequential execution:**
```bash
# Run parameter sweep one at a time
python run_optimization.py config_baseline.json --scalar_parameters.eta 0.5 --run_name eta_0.5
python run_optimization.py config_baseline.json --scalar_parameters.eta 1.0 --run_name eta_1.0
python run_optimization.py config_baseline.json --scalar_parameters.eta 1.5 --run_name eta_1.5

# Compare results (creates data/output/comparison_YYYYMMDD-HHMMSS/)
python compare_results.py "data/output/eta_*/"
```

**Parallel execution (faster):**
```bash
# Create config files for parameter sweep
# (or use run_parallel.py with existing configs)

# Run all optimizations in parallel
python run_parallel.py "config_eta_*.json"

# After jobs complete, compare results
python compare_results.py "data/output/eta_*/"

# View outputs (use actual timestamp from comparison output)
cd data/output/comparison_YYYYMMDD-HHMMSS/
open optimization_comparison_summary.xlsx
open results_comparison_summary.xlsx
open comparison_plots.pdf
```

This workflow enables systematic comparison of how model results depend on parameter choices, facilitating sensitivity analysis and scenario comparison.

## Time Integration

The model uses Euler's method with fixed time steps for transparent integration that ensures all functional relationships are satisfied exactly at output points.

### Integration Function

```python
from economic_model import integrate_model
from parameters import load_configuration

config = load_configuration('config_baseline.json')
results = integrate_model(config)
```

The `integrate_model(config)` function:
- Uses simple Euler integration: `state(t+dt) = state(t) + dt * tendency(t)`
- Time step `dt` is specified in the JSON configuration
- Returns a dictionary containing time series for all model variables

### Implementation Notes

**Negative Emissions and Cumulative Emissions Floor:**

The model allows negative emissions E(t) (carbon removal through direct air capture, afforestation, etc.), but prevents cumulative emissions Ecum from going negative:

```python
# In integrate_model() Euler step:
state['Ecum'] = max(0.0, state['Ecum'] + dt * outputs['dEcum_dt'])
```

This ensures:
- Positive E: Normal emissions, Ecum increases
- Negative E: Carbon removal, Ecum decreases
- Floor at zero: Cannot remove more CO₂ than was ever emitted (Ecum ≥ 0)

The clamp is applied during integration rather than modifying E itself, allowing the emissions rate to reflect the model's physical calculations while preventing unphysical cumulative emissions.

### Performance Optimizations

The model includes several optimizations for computational efficiency while maintaining numerical accuracy:

**1. Lagged Climate Damage Calculation (economic_model.py)**

Climate damage depends on income distribution, which itself depends on climate damage (through tax/redistribution and damage effects). This circular dependency is resolved using a lagged damage approach that eliminates the need for iterative convergence.

**Lagged Damage Algorithm:**
1. Initialize Omega_base from temperature: Ω_base = min(psi1·ΔT + psi2·ΔT², 1 - EPSILON)
2. Use climate damage from the previous timestep to compute current income distribution
3. Compute critical income ranks (Fmin, Fmax) using analytical Lorenz integrals with stepwise damage functions
4. Calculate income at each rank using `y_of_F_lagged_damage()` with previous period's damage
5. Integrate climate damage and utility over three segments: [0, Fmin), [Fmin, Fmax), [Fmax, 1]
6. Update Omega from integrated aggregate_damage_fraction for use in next timestep
7. No within-timestep iteration required - all calculations are explicit

**Performance Characteristics:**
- **No convergence loops**: Each timestep calculates income distribution explicitly using previous damage
- **Analytical rank finding**: `find_Fmin_analytical()` and `find_Fmax_analytical()` use closed-form Lorenz integrals
- **Stepwise functions**: Climate damage represented as piecewise constant over quadrature intervals
- **Speedup**: ~1000× faster than numerical quadrature-based root finding
- **Tolerance**: LOOSE_EPSILON (1e-8) used for root finding in analytical rank calculations
- **Stability**: Eliminates convergence failures and provides deterministic, reproducible results

**2. Gauss-Legendre Quadrature Integration (distribution_utilities.py)**

Numerical integration over income distribution uses Gauss-Legendre quadrature for high accuracy with minimal function evaluations:

```python
from scipy.special import roots_legendre
# Precomputed once per timestep
xi, wi = roots_legendre(N_QUAD)  # N_QUAD = 32 quadrature points
```

**Integration Functions:**
- **stepwise_interpolate(F, yi, Fi_edges)**: Evaluates piecewise constant function at rank F
- **stepwise_integrate(F0, F1, yi, Fi_edges)**: Integrates piecewise constant function from F0 to F1
- **find_Fmin_analytical()**: Finds minimum rank for taxation using closed-form Lorenz integrals
- **find_Fmax_analytical()**: Finds maximum rank for subsidies using closed-form Lorenz integrals
- **y_of_F_lagged_damage()**: Computes income at rank F using previous timestep's damage distribution

**Performance:**
- **N_QUAD = 32**: Provides excellent accuracy (~1e-10 relative error) for smooth integrands
- **Quadrature nodes precomputed**: xi, wi computed once per timestep, reused across all integrations
- **Accuracy**: Machine precision for polynomial and exponential integrands

**3. Numerical Constants (constants.py)**

Multiple precision levels and iteration parameters:

- **EPSILON = 1e-12**: Strict tolerance for mathematical comparisons (Gini bounds, float comparisons)
- **LOOSE_EPSILON = 1e-8**: Practical tolerance for root finding and optimization convergence
  - Used in find_Fmin_analytical() and find_Fmax_analytical() root finding
  - Default value for xtol_abs in optimization (control parameter convergence)
- **N_QUAD = 32**: Number of Gauss-Legendre quadrature points for numerical integration
  - Used for computing income distributions and utility integrals
  - Provides ~1e-10 relative error for smooth integrands

**Cumulative Speedup:**

These optimizations provide significant performance improvements:
- Lagged damage approach eliminates within-timestep convergence loops
- Analytical Lorenz integrals provide ~1000× speedup over numerical quadrature
- 400-year simulation: typically completes in ~0.05 seconds
- Full optimization (10,000 evaluations): typically completes in ~10 minutes

### Output Variables

The results dictionary contains arrays for:
- **Time**: `t`
- **State variables**: `K`, `Ecum`, `delta_Gini`
- **Time-dependent inputs**: `A`, `L`, `sigma`, `theta1`, `f`, `s`, `gini`
- **Economic variables**: `Y_gross`, `Y_damaged`, `Y_net`, `y`, `y_net`, `y_damaged`
- **Climate variables**: `delta_T`, `Omega`, `Omega_base`, `E`, `Climate_Damage`, `climate_damage`
- **Abatement variables**: `mu`, `Lambda`, `AbateCost`, `marginal_abatement_cost`
- **Redistribution variables**: `redistribution`, `redistribution_amount`, `Redistribution_amount`, `uniform_redistribution_amount`, `uniform_tax_rate`
- **Income distribution segments**: `Fmin`, `Fmax`
- **Aggregate integrals**: `aggregate_utility`
- **Investment/Consumption**: `Savings`, `savings`, `Consumption`, `consumption`
- **Inequality/utility**: `Gini`, `G_eff`, `Gini_climate`, `U`, `discounted_utility`
- **Distribution diagnostics**: `gini_y_net`, `gini_consumption`, `gini_climate_damage`, `delta_gini_y_net`, `delta_gini_consumption`, `delta_gini_climate_damage`, `cv_utility`
- **Tendencies**: `dK_dt`, `dEcum_dt`, `d_delta_Gini_dt`, `delta_Gini_step_change`

**Variables from lagged damage calculation:**
- **Omega_base**: Base climate damage from temperature (used for next timestep)
- **y_damaged**: Per-capita gross production after climate damage
- **climate_damage**: Per-capita climate damage
- **Fmin**: Maximum income rank receiving targeted redistribution
- **Fmax**: Minimum income rank paying progressive taxation
- **aggregate_utility**: Total utility from segment-wise integration
- **uniform_redistribution_amount**: Per-capita uniform redistribution
- **uniform_tax_rate**: Uniform tax rate (if not progressive)

**Distribution diagnostics (inequality measures calculated from discretized distributions):**
- **gini_y_net**: Gini coefficient of net income distribution (after taxes/transfers)
- **gini_consumption**: Gini coefficient of consumption distribution
- **gini_climate_damage**: Gini coefficient of climate damage distribution
- **delta_gini_y_net**: Difference from input Gini (gini_y_net - gini)
- **delta_gini_consumption**: Difference from input Gini (gini_consumption - gini)
- **delta_gini_climate_damage**: Difference from input Gini (gini_climate_damage - gini)
- **cv_utility**: Coefficient of variation of utility distribution (weighted std / weighted mean)

These diagnostics help track how redistribution and climate damage policies affect inequality across different variables. The Gini coefficients are computed from the discretized distributions at Gauss-Legendre quadrature points using the Lorenz curve integral: Gini = 2 ∫₀¹ (F - L(F)) dF. The coefficient of variation for utility is used instead of a Gini coefficient because utility can have negative values when η > 1.

All arrays have the same length corresponding to time points from `t_start` to `t_end` in steps of `dt`.

## Output and Visualization

Model results are automatically saved to timestamped directories with CSV data and PDF plots.

### Saving Results

```python
from output import save_results

# After running integration
output_paths = save_results(results, config.run_name)
```

This creates a directory: `./data/output/{run_name}_YYYYMMDD-HHMMSS/`

### Output Files

**CSV File (`results.csv`):**
- Each column is a model variable
- Columns are ordered by category (time, inputs, economic flow, emissions, investment, utility, etc.)
- Each row is a time point
- First row contains variable names (header)
- Can be loaded into Excel, Python (pandas), R, etc.

**PDF File (`plots.pdf`):**
- Multi-page PDF with organized time series plots
- Each page header displays the run name for easy identification
- Variables grouped by type (dimensionless ratios, dollar variables, etc.)
- Individual plots for single variables, combined plots for related variables with legends
- Automatically uses scientific notation for large/small values

### Example Workflow

```python
from parameters import load_configuration
from economic_model import integrate_model
from output import save_results

# Load configuration
config = load_configuration('config_baseline.json')

# Run model
results = integrate_model(config)

# Save outputs
output_paths = save_results(results, config.run_name)
print(f"Results saved to: {output_paths['output_dir']}")
```

See the **Testing the Forward Model** section above for detailed instructions on using `test_integration.py`.

### Comparing Multiple Runs

The `plot_combined_results.py` utility creates combined PDF reports showing multiple optimization runs overlaid on the same plots for easy comparison.

**Usage:**

```bash
# Compare all runs matching a pattern
python plot_combined_results.py "data/output/COIN-equality_003_tt-*"

# Compare specific runs
python plot_combined_results.py data/output/run1 data/output/run2 data/output/run3

# Specify custom output file
python plot_combined_results.py --output my_comparison.pdf "data/output/COIN*"
```

**Output:**
- Creates `plots_full_combined.pdf` (or custom name with `--output`)
- Each panel shows all runs as different colored lines
- Includes a legend panel on each page identifying all cases
- Uses the same plot layout as individual run PDFs for consistency

**Examples:**

```bash
# Compare all runs with different tax policies (tt vs tf vs ff)
python plot_combined_results.py "data/output/*-t-t*" "data/output/*-t-f*" "data/output/*-f-f*"

# Compare all completed runs
python plot_combined_results.py "data/output/*/results.csv"
```

The utility automatically:
- Searches for `results.csv` files in specified paths
- Handles glob patterns (use quotes around patterns with `*`)
- Names cases using their parent directory names
- Handles duplicate names by appending numbers

## Optimization Configuration

The JSON configuration uses iterative refinement optimization through the `optimization_parameters` section.

### Iterative Refinement Optimization

Specify the number of refinement iterations to progressively add control points:

```json
"optimization_parameters": {
  "max_evaluations": 5000,
  "optimization_iterations": 4,
  "initial_guess_f": 0.5,
  "chebyshev_scaling_power": 1.5
}
```

**Configuration rules for iterative refinement:**
- `optimization_iterations`: Integer specifying number of refinement iterations
  - Must be ≥ 1
- `initial_guess_f`: Scalar value for initial f at all control points in first iteration
  - Must satisfy 0 ≤ f ≤ 1
- `max_evaluations`: Maximum objective function evaluations per iteration
- `chebyshev_scaling_power`: Power exponent for Chebyshev node transformation (optional, default 1.5)
  - Controls concentration of control points in time
  - Values > 1.0: concentrate points near t_start (early years)
  - Values < 1.0: concentrate points near t_end (late years)
  - Value = 1.0: standard transformed Chebyshev spacing
  - Default 1.5 concentrates points early where discounting makes decisions most impactful
  - Example: With t_end=400 and scaling_power=1.5, half the points occur before year 141
- `n_points_final_f`: Target number of control points in final iteration (optional)
  - If specified, the refinement base is calculated as: `base = (n_points_final_f - 1)^(1/(n_iterations - 1))`
  - If omitted, uses default `base = 2.0`
  - Non-integer bases prevent exact alignment with previous grids
  - Example: `n_points_final_f = 10` with 4 iterations gives base ≈ 2.08 → 2, 3, 5, 10 points
  - Example: default base = 2.0 with 5 iterations gives 2, 3, 5, 9, 17 points
- `xtol_abs`: Absolute tolerance on control parameters (optional, default from NLopt)
  - Recommended: `1e-10` (stops when all |Δf| < 1e-10)
  - Since f ∈ [0,1], absolute tolerance is more meaningful than relative tolerance

**Number of control points per iteration:**
- Iteration k produces `round(1 + base^(k-1))` control points
- Default base=2.0: Iteration 1: 2 points, Iteration 2: 3 points, Iteration 3: 5 points, etc.
- Custom base from n_points_final ensures the final iteration has exactly the target number of points

**Iterative refinement algorithm:**

The optimizer performs a sequence of optimizations with progressively finer control point grids. Each iteration uses the solution from the previous iteration to initialize the new optimization via PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation.

**Control point spacing - Chebyshev nodes:**

Control points are distributed using a power-transformed Chebyshev node distribution. This provides flexible concentration of points toward early or late periods through a single tunable parameter (`chebyshev_scaling_power`). A minimum spacing constraint ensures control points are never closer together than the integration time step.

For N control points (k = 0, 1, ..., N-1):
```
u[k] = (1 - cos(k * π / (N-1))) / 2    # Normalized to [0, 1]
u_scaled[k] = u[k]^scaling_power       # Power transformation
t[k] = t_start + (t_end - t_start) * u_scaled[k]

# Enforce minimum spacing constraint
t[k] = clip(t[k], t_start + k*dt, t_end - (N-1-k)*dt)
```

**Properties:**
- `t[0] = t_start` and `t[N-1] = t_end` exactly (endpoints are fixed)
- `scaling_power > 1.0`: concentrates points near t_start (early years)
- `scaling_power < 1.0`: concentrates points near t_end (late years)
- `scaling_power = 1.0`: standard transformed Chebyshev spacing
- Default `scaling_power = 1.5` concentrates points early where discounting makes decisions most impactful
- Minimum spacing: consecutive points are at least `dt` apart (integration time step)
- Prevents numerical issues from control points too close together

**Example:** With t_end=400 and scaling_power=1.5, half of the control points occur before year 141, providing more temporal resolution in the critical early period.

**Iteration schedule:**

- **Iteration 1**: 2 control points (k=0, 1) → `[t(0), t(1)]` = `[0, t_end]`
- **Iteration 2**: 3 control points (k=0, 1, 2)
- **Iteration 3**: 5 control points (k=0, 1, 2, 3, 4)
- **Iteration 4**: 9 control points
- **Iteration n**: 1 + 2^(n-1) control points

**Initial guess strategy:**
- **Iteration 1**: All points use `initial_guess` scalar value
- **Iteration n (n ≥ 2)**:
  - Existing points from iteration n-1 use their optimal values
  - New points use PCHIP interpolation from iteration n-1 solution
  - Interpolated values are clamped to [0, 1]

**Advantages of iterative refinement:**
- Better convergence by starting with coarse, well-initialized solutions
- Progressively captures finer temporal structure in optimal policy
- Each iteration "warm starts" from previous solution
- Avoids poor local minima that can occur with many control points from cold start
- Chebyshev-based spacing provides flexible control point concentration through `chebyshev_scaling_power`
- PCHIP interpolation preserves monotonicity and shape characteristics of previous solution

### Optimization Stopping Criteria

The optimization accepts optional NLopt stopping criteria parameters:
- `xtol_abs` - Absolute tolerance on control parameters (recommended)
- `xtol_rel` - Relative tolerance on control parameters
- `ftol_abs` - Absolute tolerance on objective function
- `ftol_rel` - Relative tolerance on objective function

**Recommended practice:** Use `xtol_abs = 1e-10` as the sole stopping criterion. Since the control variable f is bounded in [0,1], absolute tolerance is more meaningful than relative tolerance, and there's no reason to want different accuracy near 0 versus near 1. The objective function can have large absolute values, making `ftol_rel` trigger prematurely even when significant improvements remain possible.

### Gradient-Based Optimization

The optimizer supports both derivative-free and gradient-based algorithms. Gradient-based algorithms (LD_*) use numerical gradient computation via finite differences for improved convergence on smooth objectives.

#### Algorithm Selection

**Single algorithm for all iterations:**
```json
"algorithm": "LN_SBPLX"
```

**Per-iteration algorithm list (progressive refinement):**
```json
"optimization_iterations": 3,
"algorithm": ["GN_ISRES", "LN_SBPLX", "LD_SLSQP"]
```

The algorithm list length must exactly match `optimization_iterations`. This enables progressive refinement strategies where early iterations explore broadly and later iterations refine with gradient information.

#### Tested Algorithms

The following algorithms have been tested with the COIN_equality model:

**✅ Derivative-Free Algorithms (Recommended)**
- **LN_SBPLX** - Primary recommendation, fast and robust
- **LN_BOBYQA** - Good alternative
- **LN_COBYLA** - Handles nonlinear constraints
- **LN_NELDERMEAD** - Classic simplex method

**✅ Gradient-Based Algorithms (Working)**
- **LD_SLSQP** - Sequential Quadratic Programming, recommended for gradient-based optimization
- **LD_MMA** - Method of Moving Asymptotes, robust alternative

**❌ Known Issues**
- **LD_LBFGS** - Runtime errors due to numerical instability with this problem structure. Use LD_SLSQP instead.

**Untested**
- **LD_CCSAQ**, **LD_VAR1**, **LD_VAR2** - May work but not yet tested
- **GN_ISRES**, **GN_DIRECT_L** - Global optimizers, expect much longer runtime

#### Algorithm Categories

**LN_\* (Local, No derivatives):** LN_SBPLX, LN_BOBYQA, LN_COBYLA, LN_NELDERMEAD
- Fast, robust for noisy objectives
- No gradient computation overhead
- Recommended for early iterations and general use
- **Primary recommendation:** LN_SBPLX

**LD_\* (Local, Derivative-based):** LD_SLSQP, LD_MMA
- Uses numerical gradients via finite differences
- Requires N+1 objective evaluations per gradient (N = number of parameters)
- Better convergence for smooth objectives
- Recommended for final polishing after derivative-free convergence
- **Primary recommendation:** LD_SLSQP

**GN_\* (Global, No derivatives):** GN_ISRES, GN_DIRECT_L
- Explores parameter space broadly
- Good for avoiding local minima
- Slower convergence
- Use only for first iteration when starting from poor initial guess

#### Progressive Refinement Strategy

Start with global exploration, refine locally, finish with gradient-based polishing:

```json
"optimization_iterations": 4,
"algorithm": ["GN_ISRES", "LN_SBPLX", "LN_SBPLX", "LD_SLSQP"]
```

This strategy:
1. **Iteration 1 (GN_ISRES):** Explores parameter space to avoid local minima
2. **Iterations 2-3 (LN_SBPLX):** Refines solution with efficient derivative-free method
3. **Iteration 4 (LD_SLSQP):** Polishes with gradient-based method for high precision

#### Gradient Computation

Gradient-based algorithms compute gradients numerically using forward finite differences:

```
∂f/∂x[i] ≈ (f(x + ε·e[i]) - f(x)) / ε
```

where ε = 1e-6 (LOOSER_EPSILON) and e[i] is the i-th unit vector.

**Cost:** N+1 objective evaluations per gradient, where N is the total number of control parameters (n_f_points + n_s_points in dual mode).

**When to use gradient-based algorithms:**
- ✅ Final polishing after derivative-free convergence
- ✅ Smooth, well-behaved objective functions
- ✅ When high precision is needed
- ❌ Early iterations (use GN_ISRES or LN_SBPLX instead)
- ❌ Noisy or discontinuous objectives

### Dual Optimization (f and s)

The model supports simultaneous optimization of both the abatement allocation fraction f(t) and the savings rate s(t). This is enabled by adding an `s_control_function` alongside the standard `control_function`.

#### Configuration for Dual Optimization

**Basic dual optimization** (constant f and s):
```json
{
  "control_function": {
    "type": "constant",
    "value": 0.5
  },
  "s_control_function": {
    "type": "constant",
    "value": 0.24
  },
  "time_functions": {
    "s": {
      "type": "constant",
      "value": 0.23974
    }
  },
  "optimization_parameters": {
    "max_evaluations": 1000,
    "optimization_iterations": 2,
    "initial_guess_f": 0.5,
    "initial_guess_s": 0.24,
    "algorithm": "LN_SBPLX",
    "xtol_abs": 1e-10
  }
}
```

**Notes:**
- `s_control_function` enables dual optimization mode
- `time_functions['s']` is still required as a fallback but will be overridden by `s_control_function`
- When `s_control_function` is present, optimization will jointly optimize both f and s
- Both variables use the same NLopt algorithm and stopping criteria

#### Dual Optimization with Different Temporal Resolution

f(t) and s(t) can have **independent numbers of control points** through iterative refinement:

```json
"optimization_parameters": {
  "max_evaluations": 10000,
  "optimization_iterations": 4,
  "initial_guess_f": 0.5,
  "initial_guess_s": 0.24,
  "n_points_final_f": 16,        // f gets 16 points in final iteration
  "n_points_final_s": 8,         // s gets 8 points in final iteration
  "algorithm": "LN_SBPLX",
  "xtol_abs": 1e-10
}
```

**Key features:**
- **Independent temporal resolution:** f and s can have different numbers of control points
- **Independent refinement schedules:** Use `n_points_final_f` and `n_points_final_s` to control resolution
- **Interpolation:** Both variables use PCHIP interpolation between control points during refinement
- **Total dimension:** n_f + n_s (e.g., 16 + 8 = 24 dimensions in final iteration above)

#### Backward Compatibility

If `s_control_function` is **not** present in the configuration:
- Single-variable optimization mode (f only)
- s(t) comes from `time_functions['s']` and is fixed during optimization
- All existing configurations continue to work without modification

#### Example: Testing Different s(t) Trajectories

To test prescribed s(t) trajectories without optimization:

```json
{
  "time_functions": {
    "s": {
      "type": "piecewise_linear",
      "time_points": [0, 400],
      "values": [0.30, 0.20]
    }
  },
  "control_function": {
    "type": "constant",
    "value": 0.5
  }
}
```

This runs the model with f=0.5 and s declining linearly from 0.30 to 0.20, without invoking dual optimization.

## Dual Optimization of Savings Rate and Abatement Allocation

### Overview

The model now supports simultaneous optimization of both:
1. **f(t)** - allocation fraction between abatement and redistribution
2. **s(t)** - savings rate (fraction of net output invested)

This capability allows the model to optimize the tradeoff between present Consumption and future Consumption (via savings/investment) while simultaneously optimizing the allocation of resources between climate mitigation and inequality reduction.

**Implementation Status:** ✅ Infrastructure complete (Phases 1-5 done)
- Dual control functions operational
- Independent control points for f and s
- Full backward compatibility maintained
- Economic behavior validated

**Remaining Work:** Phase 6 (documentation and visualization), plus automated dual optimization method (future work)

### Implementation Plan

The implementation will parallel the existing optimization structure for f(t), creating independent control points and decision times for s(t):

#### Phase 1: Make Savings Rate Time-Dependent
**Status:** ✅ Completed

1. **Move s from scalar_params to time_functions** ✅
   - Removed `s` from `ScalarParameters` dataclass in `parameters.py`
   - Added `s` to time-dependent functions in configuration files
   - Updated `evaluate_params_at_time()` to evaluate `s(t)` from time functions
   - s(t) is evaluated at each timestep and can be bounded by configuration

2. **Update all code references to s** ✅
   - In `economic_model.py`: s is now treated as time-dependent parameter from `params` dict
   - In `integrate_model()`: s is obtained from `evaluate_params_at_time(t_start, config)`
   - Initial capital iteration correctly uses time-dependent s(0)
   - Verified s is correctly stored in results and output to CSV

3. **Update configuration files** ✅
   - Converted `s` from scalar parameter to time function in both config files:
     - `config_test_DICE.json`
     - `config_test_DICE_2x_0.02_10k_0.67.json`
   - Using `"type": "constant"` with same value (0.23974) to preserve behavior
   - Tested: time-dependent s(t) = constant reproduces previous results exactly

4. **Documentation updates** ✅
   - Updated `test_integration.py` to display s(0) from time_functions
   - Docstrings in `evaluate_params_at_time()` updated to list s as time-dependent
   - s is now part of the time-dependent evaluation pipeline alongside A, L, sigma, theta1

#### Phase 2: Extend Control Function Structure
**Status:** ✅ Completed

1. **Modify control function to return both f and s** ✅
   - Created `create_dual_control_from_single(f_control, s_time_function)` to wrap f control and s time function
   - Created `create_dual_control_from_specs(f_spec, s_spec)` for future dual control specifications
   - Control function now returns tuple `(f, s)` instead of scalar

2. **Update ModelConfiguration** ✅
   - `control_function` now returns `(f(t), s(t))` tuple
   - In `load_configuration()`, f control is wrapped with s time function to create dual control
   - Clean transition: f comes from control_function spec, s from time_functions

3. **Update evaluate_params_at_time()** ✅
   - Unpacks tuple from control_function: `f, s = config.control_function(t)`
   - Both f and s are added to params dict for use by economic model
   - Removed direct s evaluation from time_functions (now comes from control function)

4. **Verified integrate_model() compatibility** ✅
   - No changes needed in integrate_model() - all control function calls go through evaluate_params_at_time()
   - Both f and s correctly stored in results and CSV output
   - Backward compatibility maintained: constant s(t) produces identical results

#### Phase 3: Extend Optimization Framework
**Status:** ✅ Completed (Infrastructure)

1. **Created dual control function builder** ✅
   - Added `create_dual_control_function_from_points(f_control_points, s_control_points)` in optimization.py
   - Interpolates f and s independently using separate control point sets
   - Returns callable function: `(f, s) = control_function(t)`
   - Supports different numbers of control points and time spacing for each variable

2. **Extended calculate_objective for dual control** ✅
   - Added optional parameters: `s_control_values` and `s_control_times`
   - **Backward compatible**: If s parameters omitted, uses fixed s from time_functions
   - **Dual mode**: If s parameters provided, optimizes both f and s independently
   - Both control values clamped to [0, 1] automatically

3. **Verified backward compatibility** ✅
   - Tested: Single-variable optimization (f only) works identically to before
   - s remains fixed from configuration when not explicitly optimized
   - Existing optimization workflows unchanged
   - Objective function calculation: 3.28e13 (verified)

4. **Infrastructure ready for full dual optimization**
   - Foundation in place for extending `optimize_control_function()` to handle dual parameter vectors
   - Can manually optimize both f and s by calling `calculate_objective()` with full parameter sets
   - Future work: Integrate into `optimize_control_function()` and iterative refinement for automated dual optimization

**Current capabilities:**
- ✅ Optimize f while keeping s fixed (default, backward compatible)
- ✅ Infrastructure supports dual optimization via direct calls to `calculate_objective()`
- ⏳ Automated dual optimization via `optimize_control_function()` (Phase 4+)

#### Phase 4: Configuration and Initial Guesses
**Status:** ✅ Completed

1. **Updated configuration file structure** ✅
   - Added optional `s_control_function` field parallel to `control_function`
   - If `s_control_function` present: s is treated as control variable
   - If absent: s comes from `time_functions['s']` (backward compatible)
   - Clean precedence: s_control_function > time_functions['s']

2. **Extended OptimizationParameters dataclass** ✅
   - Dual optimization uses same `optimization_iterations` for both f and s
   - Added `initial_guess_s`: float, parallel to `initial_guess_f` for f
   - Added `s_n_points_final`: int, for iterative refinement of s
   - Added `is_dual_optimization()` method to check if optimizing both f and s
   - All fields optional with None default (backward compatible)

3. **Updated load_configuration()** ✅
   - Checks for optional `s_control_function` in JSON
   - If present: creates dual control from both f and s control functions
   - If absent: creates dual control from f control + s time function (default)
   - Uses `create_dual_control_from_specs()` for dual control mode

4. **Created test configurations** ✅
   - Created `config_test_dual_simple.json` with both f and s as controls
   - Tested with constant values: f=0.5, s=0.24
   - Verified integration works correctly
   - Confirmed CSV output shows s from s_control_function (0.24) not time_functions (0.23974)

**Current capabilities:**
- ✅ Configuration files can specify dual optimization
- ✅ OptimizationParameters track both f and s control specifications
- ✅ Full backward compatibility when s_control_function absent
- ✅ Infrastructure ready for automated dual optimization in future phases

#### Phase 5: Testing and Validation
**Status:** ✅ Completed

1. **Verify backward compatibility** ✅
   - ✅ With s(t) = constant, results match previous version exactly (objective 3.28e13)
   - ✅ All existing configurations run correctly

2. **Test time-dependent s** ✅
   - ✅ Linear ramp s: 0.20→0.30 trajectory verified
   - ✅ Economic variables respond correctly to changing s(t)
   - ✅ Capital accumulation dynamics work as expected

3. **Test dual optimization infrastructure** ✅
   - ✅ Single control point for both (2D optimization via grid search)
   - ✅ Multiple control points with same number (n_f = n_s = 3)
   - ✅ Multiple control points with different numbers (n_f = 2, n_s = 4 and vice versa)
   - ⚠️ Iterative refinement not yet implemented (deferred to future work)

4. **Verify economic sensibility** ✅
   - ✅ Different s(t) trajectories produce different outcomes (0.51% variation)
   - ✅ Best constant s = 0.30 in typical range [0.20, 0.35]
   - ✅ Higher savings improves utility by 0.17% over baseline
   - ✅ Declining s trajectory outperforms increasing s (matches economic intuition)

**Key accomplishments:**
- Dual control infrastructure fully functional
- Independent control points for f and s verified
- Economic behavior consistent with growth theory
- Full backward compatibility maintained

#### Phase 6: Documentation and Output
**Status:** ✅ Completed

1. **Update README.md** ✅
   - ✅ Documented dual control variable framework in "Control Variables" section
   - ✅ Added "Dual Optimization (f and s)" section in "Optimization Configuration"
   - ✅ Explained s(t) optimization and configuration options
   - ✅ Added examples of dual optimization configurations (basic and multi-point)
   - ✅ Documented backward compatibility behavior
   - ✅ Showed example of prescribed s(t) trajectory testing

2. **Update output visualization** ✅
   - ✅ s(t) plotted alongside f(t) in "Control Variables" combined chart
   - ✅ Both variables shown in dimensionless_ratios plot group
   - ✅ Visualization automatically handles both single and dual optimization modes
   - ✅ Uses existing combined chart infrastructure for clean f/s comparison

3. **Update optimization summary output** ✅
   - ✅ write_optimization_summary() reports optimal f(t) control points
   - ✅ write_optimization_summary() reports optimal s(t) control points (when present)
   - ✅ Separate sections for f and s control point tables
   - ✅ Infrastructure ready for full dual optimization method (future work)

**Key accomplishments:**
- Complete documentation of dual control framework
- User-facing examples for all dual optimization modes
- Output visualization supports both f and s
- Optimization summary ready for dual optimization results

### Key Design Decisions

1. **Independent control times**: Each variable (f, s) has its own set of control times, allowing different temporal resolution where needed

2. **Parallel iteration**: During iterative refinement, both variables go through the same number of iterations, but may have different numbers of control points

3. **Minimal structural change**: The implementation leverages the existing optimization framework, simply extending the parameter vector dimension

4. **Backward compatibility**: Configurations can specify constant s(t) to reproduce current behavior exactly

### Expected Benefits

- **More realistic optimization**: Savings rate is a key economic policy variable that should be optimized alongside other controls
- **Richer dynamics**: Time-varying s(t) allows model to balance present vs. future Consumption optimally
- **Methodological advancement**: Demonstrates framework extensibility to multi-dimensional control problems

## Project Structure

```
coin_equality/
├── README.md                          # This file
├── CLAUDE.md                          # AI coding style guide
├── requirements.txt                   # Python dependencies
├── distribution_utilities.py          # Income distribution, utility integration, and stepwise functions
├── economic_model.py                  # Economic production and tendency calculations
├── parameters.py                      # Parameter definitions and configuration loading
├── optimization.py                    # Optimization framework with iterative refinement
├── output.py                          # Output generation (CSV and PDF)
├── test_integration.py                # Test script for forward integration
├── run_optimization.py                # Main script for running optimizations
├── run_parallel.py                    # Launch multiple optimizations in parallel
├── compare_results.py                 # Compare multiple optimization runs
├── comparison_utils.py                # Utilities for multi-run comparison
├── visualization_utils.py             # Unified visualization functions
├── config_baseline.json               # Baseline scenario configuration
├── config_high_inequality.json        # High inequality scenario configuration
├── data/output/                       # Output directory (timestamped subdirectories)
├── coin_equality (methods) v0.1.pdf   # Detailed methods document
└── [source code directories]
```

## References

Barrage, L., & Nordhaus, W. (2024). "Policies, projections, and the social cost of carbon: Results from the DICE-2023 model." *Proceedings of the National Academy of Sciences*, 121(13), e2312030121. https://doi.org/10.1073/pnas.2312030121

Caldeira, K., Bala, G., & Cao, L. (2023). "Climate sensitivity uncertainty and the need for energy without CO₂ emission." *Environmental Research Letters*, 18(9), 094021. https://doi.org/10.1088/1748-9326/acf949

Nordhaus, W. D. (1992). "An optimal transition path for controlling greenhouse gases." *Science*, 258(5086), 1315-1319. https://doi.org/10.1126/science.258.5086.1315

Nordhaus, W. D. (2017). "Revisiting the social cost of carbon." *Proceedings of the National Academy of Sciences*, 114(7), 1518-1523. https://doi.org/10.1073/pnas.1609244114

## License

MIT License

Copyright (c) 2025 Lamprini Papargyri, ..., and Ken Caldeira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Authors

Lamprini Papargyri, ..., and Ken Caldeira
