"""
Numerical constants and tolerances for COIN_equality model.

Defines small and large values used for numerical stability and bounds checking.
Centralizes epsilon/bignum definitions to ensure consistency across the codebase.
"""

# Large negative number for utility when constraints are violated
# Used as penalty value when Gini or other variables are out of valid range
NEG_BIGNUM = -1e30

# Small epsilon for numerical comparisons and bounds
# Used for:
# - Comparing floats to unity (e.g., eta ≈ 1)
# - Checking if values are effectively zero (e.g., fract_gdp ≈ 0)
# - Bounding variables away from exact 0 or 1 (e.g., Gini ∈ (ε, 1-ε))
# - Ensuring values stay strictly positive (e.g., A2 ≥ ε)
# - Root finding bracket offsets
EPSILON = 1e-12

# Looser epsilon for iterative convergence and optimization tolerances
# Used for:
# - Convergence criterion in y_net iterative solver
# - Default value for xtol_abs in optimization (control parameter convergence)
# Provides practical precision without requiring machine precision
LOOSE_EPSILON = 1e-6

# Even looser epsilon for numerical gradient computation via finite differences
# Used for:
# - Perturbation size in gradient-based optimization (LD_* algorithms)
# - Numerical differentiation via forward differences
# Larger than LOOSE_EPSILON for better numerical stability in finite difference approximations
LOOSER_EPSILON = 1e-6

# Objective function scaling factor for numerical stability in gradient-based optimization
# Used for:
# - Scaling objective values from ~1.5e13 to ~1.5 for better numerical conditioning
# - Applied consistently to both objective values and gradients in all optimization wrappers
# - Improves stability of gradient-based algorithms (LD_SLSQP, LD_LBFGS, etc.)
OBJECTIVE_SCALE = 1e-13

# Large value for detecting effectively infinite parameters
# Used for:
# - Checking if y_damage_distribution_exponent is so small that damage is effectively uniform
# - Detecting when parameters should trigger special case handling
INVERSE_EPSILON = 1.0 / EPSILON

# Maximum iterations for convergence loops
# Used for:
# - Initial capital stock convergence in integrate_model()
# - Climate damage convergence in calculate_tendencies()
# Set to 256 to allow slow but steady convergence during optimization
MAX_ITERATIONS = 256

# N_QUAD removed - now specified in config.integration_params.n_quad
# This enforces explicit configuration (no defaults per CLAUDE.md)
