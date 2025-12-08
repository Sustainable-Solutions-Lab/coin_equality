# Implementation Plan: Lagged Climate Damage

## Overview

Switch from iterative climate damage calculation to lagged (previous time step) damage calculation to eliminate the convergence loop in `calculate_tendencies()`.

## Current Approach (Implicit, Iterative)

**Problem:** At each time step, income distribution and climate damage depend on each other circularly:

```
y(F,t) = y_mean(t) * dL/dF(F; gini(t)) + uniform_redistribution(t)
         - omega_base(t) * (y(F,t) / y_net_reference)^alpha
```

where `y(F,t)` appears on both sides (in the damage term).

**Solution:** Iterate to find `Omega_base` that satisfies aggregate damage constraint (lines 207-396 in `economic_model.py`)

**Cost:**
- Convergence loop requires ~2-256 iterations per timestep
- Complex secant/regula-falsi convergence logic
- Can fail to converge in edge cases

## Proposed Approach (Explicit, No Iteration)

**Solution:** Use lagged income and damage from previous timestep:

```
y(F,t) = y_mean(t) * dL/dF(F; gini(t)) + uniform_redistribution(t)
         - omega_base(t-dt) * (y(F,t-dt) / y_net_reference)^alpha
```

**Benefit:**
- No iteration needed - explicit calculation
- Guaranteed "convergence" in 1 step
- Simpler, more robust code
- Physically reasonable: climate damage responds to past conditions

## Implementation Steps

### 1. Extend `previous_step_values` Dictionary

**Current state:** (lines 632-635 in `economic_model.py`)
```python
previous_step_values = {
    'y_mean': y_gross_initial,
    'gini': Gini_initial,
}
```

**New state:** Add all variables needed for lagged damage calculation:
```python
previous_step_values = {
    # Aggregate state
    'y_mean': y_gross_initial,           # Mean income before damage/tax/redistribution
    'gini': Gini_initial,                # Gini coefficient
    'Omega_base': 0.0,                   # Base damage parameter (initial: no damage)
    'y_net_reference': y_net_reference,  # Reference income for damage scaling

    # Income distribution boundaries
    'Fmin': 0.0,                         # Lower redistribution boundary
    'Fmax': 1.0,                         # Upper tax boundary

    # Redistribution/tax parameters
    'uniform_redistribution_amount': 0.0, # Per-capita uniform redistribution
    'uniform_tax_rate': 0.0,              # Uniform tax rate

    # For computing y(F,t-dt) when F boundaries change
    'y_gross': y_gross_initial,           # Gross income before tax
    'y_net': y_gross_initial,             # Net income after everything
}
```

### 2. Create New Function: `y_of_F_lagged_damage()`

**Location:** `income_distribution.py`

**Purpose:** Compute income distribution using lagged damage (explicit formula)

**Signature:**
```python
def y_of_F_lagged_damage(
    F,
    # Current timestep parameters
    Fmin_current,
    Fmax_current,
    y_mean_before_damage_current,
    uniform_redistribution_current,
    uniform_tax_rate_current,
    gini_current,
    # Lagged damage from previous timestep
    Omega_base_prev,
    y_damage_distribution_exponent,
    y_net_reference,
    # Previous income distribution for damage calculation
    Fmin_prev,
    Fmax_prev,
    y_gross_prev,
    uniform_redistribution_prev,
    uniform_tax_rate_prev,
    gini_prev
):
    """
    Compute y(F,t) using lagged climate damage from time t-dt.

    Uses explicit formula (no iteration needed):
        y(F,t) = y_mean(t) * dL/dF(F; gini(t)) + uniform_redistribution(t)
                 - damage(F,t-dt)

    where damage(F,t-dt) = Omega_base(t-dt) * (y(F,t-dt) / y_net_reference)^alpha

    Key challenge: Fmin and Fmax may differ between t and t-dt,
    so need to handle boundary changes when computing y(F,t-dt).
    """
```

**Algorithm:**
1. Compute base income before damage (current timestep):
   ```python
   A_current = y_mean_before_damage_current * dL/dF(F; gini_current)
               + uniform_redistribution_current
   ```

2. Compute income at same F rank on previous timestep:
   ```python
   y_prev_at_F = y_of_F_previous_timestep(
       F, Fmin_prev, Fmax_prev, y_gross_prev,
       uniform_redistribution_prev, uniform_tax_rate_prev, gini_prev
   )
   ```

3. Compute lagged damage:
   ```python
   damage_lagged = Omega_base_prev * (y_prev_at_F / y_net_reference)^alpha
   ```

4. Return explicit solution:
   ```python
   return A_current - damage_lagged
   ```

**Helper function needed:** `y_of_F_previous_timestep()`
- Reconstruct income distribution from previous timestep
- Handle case where F boundaries differ between timesteps
- Use saved parameters from `previous_step_values`

### 3. Modify `calculate_tendencies()` to Use Lagged Damage

**Remove:** Lines 200-396 (entire convergence loop)

**Replace with:**

```python
# Calculate income distribution using LAGGED damage (no iteration)
# Extract previous timestep values
Omega_base_prev = previous_step_values['Omega_base']
Fmin_prev = previous_step_values['Fmin']
Fmax_prev = previous_step_values['Fmax']
y_gross_prev = previous_step_values['y_gross']
uniform_redistribution_prev = previous_step_values['uniform_redistribution_amount']
uniform_tax_rate_prev = previous_step_values['uniform_tax_rate']
gini_prev = previous_step_values['gini']

# Compute current redistribution/tax amounts (same logic as before)
omega_for_budget = min(Omega_target, 1.0 - EPSILON)
available_for_redistribution_and_abatement = fract_gdp * y_gross * (1 - omega_for_budget)

if income_redistribution:
    redistribution_amount = (1 - f) * available_for_redistribution_and_abatement
else:
    redistribution_amount = 0.0

abateCost_amount = f * available_for_redistribution_and_abatement

# Find Fmin using LAGGED damage
if income_redistribution and income_dependent_redistribution_policy:
    uniform_redistribution_amount = 0.0
    Fmin = find_Fmin_lagged(
        y_gross, Omega_base_prev, y_damage_distribution_exponent,
        y_net_reference, uniform_redistribution_amount, gini, xi, wi,
        target_subsidy=redistribution_amount,
        # Previous timestep for damage calculation
        Fmin_prev=Fmin_prev, Fmax_prev=Fmax_prev,
        y_gross_prev=y_gross_prev, gini_prev=gini_prev,
        uniform_redistribution_prev=uniform_redistribution_prev,
        uniform_tax_rate_prev=uniform_tax_rate_prev
    )
else:
    uniform_redistribution_amount = redistribution_amount
    Fmin = 0.0

# Find Fmax using LAGGED damage
if income_dependent_tax_policy:
    tax_amount = abateCost_amount + redistribution_amount
    uniform_tax_rate = 0.0
    Fmax = find_Fmax_lagged(
        Fmin, y_gross, Omega_base_prev, y_damage_distribution_exponent,
        y_net_reference, uniform_redistribution_amount, gini, xi, wi,
        target_tax=tax_amount,
        # Previous timestep for damage calculation
        Fmin_prev=Fmin_prev, Fmax_prev=Fmax_prev,
        y_gross_prev=y_gross_prev, gini_prev=gini_prev,
        uniform_redistribution_prev=uniform_redistribution_prev,
        uniform_tax_rate_prev=uniform_tax_rate_prev
    )
else:
    uniform_tax_rate = (abateCost_amount + redistribution_amount) / (y_gross * (1 - omega_for_budget))
    Fmax = 1.0

# Calculate aggregate damage using LAGGED damage function
# (Three segments: bottom, middle, top)
aggregate_damage_fraction = 0.0
aggregate_utility = 0.0

if Fmin > EPSILON:
    # Low income segment - use lagged damage
    ...

if Fmax - Fmin > EPSILON:
    # Middle income segment - use lagged damage
    ...

if 1.0 - Fmax > EPSILON:
    # High income segment - use lagged damage
    ...

# Omega is now directly from aggregate damage (no iteration)
Omega = min(aggregate_damage_fraction, 1.0 - EPSILON)

# If income_dependent_aggregate_damage is False, need to scale Omega_base
# for NEXT timestep to hit target
if income_dependent_damage_distribution and not income_dependent_aggregate_damage:
    # Simple scaling: Omega_base_next = Omega_base_prev * (Omega_target / Omega)
    Omega_base = Omega_base_prev * (Omega_target / Omega) if Omega > EPSILON else Omega_base_prev
else:
    Omega_base = Omega_target  # Direct use

n_damage_iterations = 1  # Always exactly 1 (no iteration!)
```

### 4. Update Functions in `income_distribution.py`

**Create lagged versions:**

1. `find_Fmin_lagged()` - wrapper around `find_Fmin()` that uses `total_tax_bottom_lagged()`
2. `find_Fmax_lagged()` - wrapper around `find_Fmax()` that uses `total_tax_top_lagged()`
3. `total_tax_bottom_lagged()` - uses `segment_integral_with_cut_lagged()`
4. `total_tax_top_lagged()` - uses `segment_integral_with_cut_lagged()`
5. `segment_integral_with_cut_lagged()` - uses `y_of_F_lagged_damage()`
6. `climate_damage_integral_lagged()` - for aggregate damage calculation
7. `crra_utility_integral_with_damage_lagged()` - for utility calculation

**Alternative approach:** Modify existing functions to accept optional `use_lagged_damage` flag and `previous_step_values` parameter instead of creating duplicates.

### 5. Update Return Values from `calculate_tendencies()`

**Current:**
```python
results['current_income_dist'] = {
    'y_mean': y_net
}
```

**New:**
```python
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
}
```

### 6. Handle First Timestep (Initial Condition)

**Challenge:** No previous timestep exists at t=0

**Solution 1 (Simple):** Assume zero damage initially
```python
if t == t_start:
    previous_step_values['Omega_base'] = 0.0
    # All other fields use reasonable defaults
```

**Solution 2 (Better):** Compute initial damage from Ecum_initial
```python
if t == t_start:
    delta_T_initial = k_climate * Ecum_initial
    Omega_initial = psi1 * delta_T_initial + psi2 * delta_T_initial**2
    previous_step_values['Omega_base'] = Omega_initial
```

**Solution 3 (Best):** Run one iteration at t=0 to establish consistent initial state
- Use iterative approach only for t=0
- All subsequent steps use lagged approach
- Ensures first output is self-consistent

### 7. Testing Strategy

**Phase 1: Validation against existing code**
1. Add feature flag: `use_lagged_damage` (default: False)
2. Run both approaches on same configuration
3. Compare results - should be nearly identical (small lag effects acceptable)

**Phase 2: Performance testing**
1. Measure speedup from eliminating iteration
2. Verify no convergence failures
3. Test edge cases (catastrophic damage, near-zero redistribution, etc.)

**Phase 3: Acceptance criteria**
1. Results within 1% of iterative approach for standard cases
2. No convergence failures
3. Speedup > 2x (eliminating ~10-50 iterations per timestep)
4. All existing configurations run successfully

## Detailed Function Modifications

### `y_of_F_lagged_damage()` Implementation Details

```python
def y_of_F_lagged_damage(
    F,
    Fmin_current, Fmax_current,
    y_mean_before_damage_current,
    uniform_redistribution_current,
    uniform_tax_rate_current,
    gini_current,
    Omega_base_prev,
    y_damage_distribution_exponent,
    y_net_reference,
    Fmin_prev, Fmax_prev,
    y_gross_prev,
    uniform_redistribution_prev,
    uniform_tax_rate_prev,
    gini_prev
):
    # Current base income (before damage)
    a_current = (1.0 + 1.0 / gini_current) / 2.0
    dLdF_current = (1.0 - 1.0 / a_current) * (1.0 - F) ** (-1.0 / a_current)
    A_current = (y_mean_before_damage_current * (1 - uniform_tax_rate_current)
                 * dLdF_current + uniform_redistribution_current)

    # Previous income at same F rank (for damage calculation)
    # Need to reconstruct y(F,t-dt) from saved parameters
    a_prev = (1.0 + 1.0 / gini_prev) / 2.0
    dLdF_prev = (1.0 - 1.0 / a_prev) * (1.0 - F) ** (-1.0 / a_prev)
    y_prev = (y_gross_prev * (1 - uniform_tax_rate_prev) * dLdF_prev
              + uniform_redistribution_prev)

    # Handle boundary changes: if F was outside [Fmin_prev, Fmax_prev],
    # use boundary value for damage calculation
    F_clipped = np.clip(F, Fmin_prev, Fmax_prev)
    if F_clipped != F:
        dLdF_prev_clip = (1.0 - 1.0 / a_prev) * (1.0 - F_clipped) ** (-1.0 / a_prev)
        y_prev = (y_gross_prev * (1 - uniform_tax_rate_prev) * dLdF_prev_clip
                  + uniform_redistribution_prev)

    # Lagged damage
    if y_damage_distribution_exponent < EPSILON:
        damage_lagged = Omega_base_prev
    else:
        damage_lagged = Omega_base_prev * (y_prev / y_net_reference)**y_damage_distribution_exponent

    # Explicit solution
    return A_current - damage_lagged
```

### Handling Fmin/Fmax Changes Between Timesteps

**Problem:** Income distribution boundaries can shift:
- t-dt: redistribution for F ∈ [0, Fmin_prev], taxation for F ∈ [Fmax_prev, 1]
- t: redistribution for F ∈ [0, Fmin_current], taxation for F ∈ [Fmax_current, 1]

**Impact:** When computing damage at F, if F was outside [Fmin_prev, Fmax_prev], need to extrapolate

**Solution:** Use boundary value for damage calculation (conservative approach)
- If F < Fmin_prev: use y(Fmin_prev, t-dt) for damage
- If F > Fmax_prev: use y(Fmax_prev, t-dt) for damage
- If Fmin_prev ≤ F ≤ Fmax_prev: use y(F, t-dt) directly

## Implementation Sequence

1. ✅ **Branch created:** `lag-damage`

2. **Create helper functions** (new file: `income_distribution_lagged.py`?):
   - `y_of_F_lagged_damage()`
   - `y_of_F_previous_timestep_reconstruction()`

3. **Modify `calculate_tendencies()`**:
   - Add feature flag `use_lagged_damage` in params
   - Expand `previous_step_values` initialization
   - Replace convergence loop with explicit calculation (inside `if use_lagged_damage:`)
   - Keep old code path for validation

4. **Add tests**:
   - Unit test: `test_lagged_vs_iterative_damage.py`
   - Integration test: run optimization with both approaches

5. **Validate**:
   - Compare results on standard configurations
   - Document any differences

6. **Switch default**:
   - Set `use_lagged_damage = True` by default
   - Run full test suite

7. **Clean up**:
   - Remove old iterative code path
   - Remove feature flag
   - Update documentation

## Potential Issues and Mitigations

### Issue 1: Lag introduces temporal delay

**Problem:** Damage responds one timestep late to income changes

**Mitigation:**
- With dt = 1 year, lag is physically reasonable (climate responds to past emissions)
- Can reduce timestep if needed
- Validation should show negligible difference in optimized paths

### Issue 2: Initial condition sensitivity

**Problem:** First timestep has no previous values

**Mitigation:**
- Use temperature-based initial Omega_base from Ecum_initial
- OR: run one iteration at t=0 only
- Document assumption clearly

### Issue 3: Boundary discontinuities when Fmin/Fmax jump

**Problem:** Abrupt boundary changes could cause artifacts

**Mitigation:**
- Use boundary values for extrapolation (smooth)
- Monitor Fmin/Fmax changes in output
- Optimization should naturally avoid abrupt policy changes

### Issue 4: No feedback for income_dependent_aggregate_damage = False

**Problem:** With iterative approach, Omega_base adjusts to hit target. With lag, this is one step behind.

**Mitigation:**
- Simple scaling: `Omega_base(t+dt) = Omega_base(t) * (Omega_target / Omega_achieved(t))`
- This is essentially explicit Euler on the Omega_base adjustment
- Should converge to correct value over a few timesteps

## Success Metrics

1. **Correctness:** Results within 1% of iterative approach for test cases
2. **Performance:** ≥2x speedup in `calculate_tendencies()`
3. **Robustness:** Zero convergence failures in test suite
4. **Simplicity:** ~200 fewer lines of convergence logic
5. **Maintainability:** Easier to understand and debug

## Files to Modify

1. `economic_model.py` - main changes to `calculate_tendencies()`
2. `income_distribution.py` - add lagged damage functions
3. `parameters.py` - add `use_lagged_damage` flag (temporary)
4. `README.md` - document new approach
5. `test_integration.py` - add validation tests

## Timeline Estimate

- Helper functions: 2-3 hours
- Modify `calculate_tendencies()`: 2-3 hours
- Testing and validation: 3-4 hours
- Documentation: 1 hour
- **Total: ~8-11 hours**

## Open Questions

1. Should we keep iterative approach as fallback, or fully replace?
2. What initial condition works best for t=0?
3. Should lagged functions be in separate file or modify existing functions?
4. Need to update find_Fmin/find_Fmax or create new versions?
5. How to handle convergence diagnostic output (n_damage_iterations)?
