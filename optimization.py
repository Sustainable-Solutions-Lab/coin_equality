"""
Optimization framework for finding optimal allocation between redistribution and abatement.

This module provides control point parameterization for the control function f(t)
and optimization using NLopt to maximize discounted aggregate utility.
"""

import numpy as np
import nlopt
import sys
import time
from scipy.interpolate import PchipInterpolator
from economic_model import integrate_model
from parameters import ModelConfiguration
from constants import EPSILON


def requires_gradient(algorithm_name):
    """
    Check if NLopt algorithm requires gradient computation.

    Parameters
    ----------
    algorithm_name : str
        NLopt algorithm name (e.g., 'LD_SLSQP', 'LN_SBPLX')

    Returns
    -------
    bool
        True if algorithm requires gradients (LD_* or GD_* algorithms)
        False for derivative-free algorithms (LN_*, GN_*)

    Notes
    -----
    NLopt algorithm naming convention:
    - LD_* : Local, Derivative-based (requires gradients)
    - LN_* : Local, No derivatives
    - GN_* : Global, No derivatives
    - GD_* : Global, Derivative-based (rare, not commonly used)
    """
    return algorithm_name.startswith('LD_') or algorithm_name.startswith('GD_')


def calculate_chebyshev_times(n_points, t_start, t_end, scaling_power, dt):
    """
    Calculate control point times using transformed Chebyshev nodes.

    Distributes control points using a power-transformed Chebyshev distribution,
    allowing flexible concentration of points toward early or late times. Enforces
    minimum spacing constraint to prevent points from being closer than dt.

    Parameters
    ----------
    n_points : int
        Number of control points to generate (must be >= 2)
    t_start : float
        Start time (years)
    t_end : float
        End time (years)
    scaling_power : float
        Exponent for power transformation (must be > 0)
        - scaling_power > 1: concentrates points near t_start
        - scaling_power < 1: concentrates points near t_end
        - scaling_power = 1: standard transformed Chebyshev distribution
    dt : float
        Minimum spacing between control points (years)
        Control points will be at least dt apart

    Returns
    -------
    ndarray
        Control times from t_start to t_end, with boundaries exactly at endpoints
        and minimum spacing dt between consecutive points

    Notes
    -----
    Algorithm:
    1. Generate normalized Chebyshev-like nodes: u[k] = (1 - cos(k*π/(N-1))) / 2
       for k = 0, 1, ..., N-1
    2. Calculate maximum scaling power that ensures x[1] ≥ t_start + dt:
       max_scaling_power = log(dt / (t_end - t_start)) / log(u[1])
    3. Use effective_scaling_power = min(scaling_power, max_scaling_power)
    4. Apply power transformation: u_scaled[k] = u[k]^effective_scaling_power
    5. Map to time interval: t[k] = t_start + (t_end - t_start) * u_scaled[k]

    The maximum scaling power constraint ensures:
    - The second point (k=1) is at least dt from t_start
    - All subsequent points maintain proper Chebyshev spacing
    - No artificial clipping that distorts the distribution

    This prevents numerical issues from having control points closer together
    than the integration time step while preserving the Chebyshev distribution shape.

    Examples
    --------
    Standard Chebyshev-like spacing (scaling_power=1.0):
    >>> times = calculate_chebyshev_times(5, 0, 100, 1.0, 1.0)

    Concentrate points in early period (scaling_power=1.5):
    >>> times = calculate_chebyshev_times(20, 0, 400, 1.5, 1.0)
    # Half of points will be in first ~141 years, with minimum 1-year spacing
    """
    N = n_points
    k_values = np.arange(N)

    # Transformed Chebyshev nodes mapped to [0, 1]
    u = (1 - np.cos(k_values * np.pi / (N - 1))) / 2

    # Calculate maximum scaling power that ensures x[1] >= t_start + dt
    # From: t_start + (t_end - t_start) * u[1]^scaling_power >= t_start + dt
    # We get: scaling_power <= log(dt / (t_end - t_start)) / log(u[1])
    u_1 = u[1]  # Second point's normalized position
    normalized_dt = dt / (t_end - t_start)
    max_scaling_power = np.log(normalized_dt) / np.log(u_1)

    # Use the minimum of requested scaling_power and the constraint
    effective_scaling_power = min(scaling_power, max_scaling_power)

    # Apply power transformation with effective scaling
    u_scaled = u ** effective_scaling_power

    # Map to [t_start, t_end]
    times = t_start + (t_end - t_start) * u_scaled

    # Ensure exact endpoints (handle floating point precision)
    times[0] = t_start
    times[-1] = t_end

    return times, effective_scaling_power


def interpolate_to_new_grid(old_times, old_values, new_times):
    """
    Interpolate control values to a new grid using PCHIP interpolation.

    Uses Piecewise Cubic Hermite Interpolating Polynomial for shape-preserving
    interpolation with continuous first derivatives. Clamps results to [0, 1]
    to ensure valid control values.

    Parameters
    ----------
    old_times : array_like
        Times at which old values are defined
    old_values : array_like
        Control values at old_times
    new_times : array_like
        Times at which to evaluate interpolated values

    Returns
    -------
    ndarray
        Interpolated values at new_times, clamped to [0, 1]

    Notes
    -----
    For new_times that match old_times, returns the exact old_values.
    For new_times beyond the range of old_times, uses constant extrapolation.
    Results are clamped to [0, 1] to ensure valid control function values.
    """
    old_times = np.asarray(old_times)
    old_values = np.asarray(old_values)
    new_times = np.asarray(new_times)

    # Ensure strictly increasing times
    sort_indices = np.argsort(old_times)
    old_times = old_times[sort_indices]
    old_values = old_values[sort_indices]

    # Remove duplicate times by keeping only unique times (within tolerance)
    eps = 1e-10
    unique_mask = np.concatenate([[True], np.diff(old_times) > eps])
    old_times = old_times[unique_mask]
    old_values = old_values[unique_mask]

    if len(old_times) == 1:
        return np.full_like(new_times, old_values[0], dtype=float)

    interpolator = PchipInterpolator(old_times, old_values, extrapolate=False)
    new_values = np.where(
        new_times <= old_times[-1],
        interpolator(new_times),
        old_values[-1]
    )

    new_values = np.clip(new_values, 0.0, 1.0)

    return new_values


def evaluate_control_function(control_points, t):
    """
    Evaluate f(t) from control points using Pchip interpolation.

    Uses Pchip (Piecewise Cubic Hermite Interpolating Polynomial) for
    shape-preserving interpolation with continuous first derivatives.
    For t beyond the last control point, uses constant extrapolation.
    For a single control point, returns constant for all t.

    Parameters
    ----------
    control_points : list of tuples
        List of (time, value) tuples defining control points.
        Must have at least one point. Values should satisfy 0 <= value <= 1.
    t : float or array_like
        Time(s) at which to evaluate the control function

    Returns
    -------
    float or ndarray
        Control function value(s) at time(s) t, clamped to [0, 1]

    Notes
    -----
    Interpolation properties:
    - C¹ continuity (continuous first derivatives)
    - Shape-preserving and monotonicity-preserving
    - No overshoot beyond the range of control point values
    - Results are clamped to [0,1] to handle numerical precision issues

    Special cases:
    - Single control point [(t₀, f₀)]: returns f₀ for all t (constant)
    - For t > t_max: returns f(t_max) (constant extrapolation)

    Examples
    --------
    Single control point (constant trajectory):
    >>> f_func = lambda t: evaluate_control_function([(0, 0.5)], t)
    >>> f_func(100)  # Returns 0.5 for all t

    Multiple control points:
    >>> points = [(0, 0.2), (50, 0.8), (100, 0.6)]
    >>> f_func = lambda t: evaluate_control_function(points, t)
    >>> f_func(25)  # Interpolated value between 0.2 and 0.8
    """
    control_points = sorted(control_points)
    times = np.array([pt[0] for pt in control_points])
    values = np.array([pt[1] for pt in control_points])

    # Ensure strictly increasing times (handles numerical precision issues)
    sort_indices = np.argsort(times)
    times = times[sort_indices]
    values = values[sort_indices]

    # Remove duplicate times by keeping only unique times (within tolerance)
    # When times are duplicated, keep the first value
    eps = 1e-10
    unique_mask = np.concatenate([[True], np.diff(times) > eps])
    times = times[unique_mask]
    values = values[unique_mask]

    t_array = np.atleast_1d(t)

    if len(times) == 1:
        result = np.full_like(t_array, values[0], dtype=float)
    else:
        interpolator = PchipInterpolator(times, values, extrapolate=False)
        result = np.where(
            t_array <= times[-1],
            interpolator(t_array),
            values[-1]
        )

    result = np.clip(result, 0.0, 1.0)

    return result if np.ndim(t) > 0 else float(result[0])


def create_control_function_from_points(control_points):
    """
    Create a callable control function from control points.

    Parameters
    ----------
    control_points : list of tuples
        List of (time, value) tuples defining control points

    Returns
    -------
    callable
        Function f(t) that evaluates the control function at time t
    """
    return lambda t: evaluate_control_function(control_points, t)


def create_f_and_s_control_function_from_points(f_control_points, s_control_points):
    """
    Create a callable control function for both f and s from separate control points.

    Parameters
    ----------
    f_control_points : list of tuples
        List of (time, f_value) tuples defining f control points
    s_control_points : list of tuples
        List of (time, s_value) tuples defining s control points

    Returns
    -------
    callable
        Function returning (f(t), s(t)) tuple that evaluates both controls at time t

    Notes
    -----
    f and s are interpolated independently using their own control points.
    This allows different numbers of control points and different time spacing
    for each variable.
    """
    return lambda t: (
        evaluate_control_function(f_control_points, t),
        evaluate_control_function(s_control_points, t)
    )


class UtilityOptimizer:
    """
    Optimizer for finding optimal allocation between redistribution and abatement.

    Maximizes the discounted aggregate utility integral:
        max ∫₀^T e^(-ρt) · U(t) · L(t) dt

    where U(t) is mean utility and L(t) is population.

    The control function f(t) is parameterized by discrete control points,
    with interpolation and extrapolation handled by evaluate_control_function().
    """

    def __init__(self, base_config):
        """
        Initialize optimizer with base configuration.

        Parameters
        ----------
        base_config : ModelConfiguration
            Base model configuration. The control function will be replaced
            during optimization.
        """
        self.base_config = base_config
        self.n_evaluations = 0
        self.best_objective = -np.inf
        self.best_control_values = None
        self.degenerate_case = False
        self.degenerate_reason = None

    def calculate_objective(self, control_values, control_times, s_control_values=None, s_control_times=None):
        """
        Calculate the discounted aggregate utility for given control point values.

        Parameters
        ----------
        control_values : array_like
            Control function values (f) at control_times (one per control point)
        control_times : array_like
            Times at which f control points are placed
        s_control_values : array_like, optional
            Control function values (s) at s_control_times. If None, uses fixed s from time_functions.
        s_control_times : array_like, optional
            Times at which s control points are placed. If None, uses fixed s from time_functions.

        Returns
        -------
        float
            Discounted utility integral

        Notes
        -----
        Uses trapezoidal integration for the discounted utility integral.
        Control values are clamped to [0, 1] to handle numerical precision issues.

        If s_control_values and s_control_times are provided, creates dual control function
        with independent interpolation for f and s. Otherwise, uses fixed s(t) from configuration.
        """
        self.n_evaluations += 1

        control_values = np.clip(control_values, 0.0, 1.0)
        f_control_points = list(zip(control_times, control_values))

        if s_control_values is not None and s_control_times is not None:
            # f and s optimization mode: both f and s are control variables
            s_control_values = np.clip(s_control_values, 0.0, 1.0)
            s_control_points = list(zip(s_control_times, s_control_values))
            control_function = create_f_and_s_control_function_from_points(f_control_points, s_control_points)
        else:
            # Single optimization mode: only f is optimized, s is fixed from time_functions or initial_guess
            f_control = create_control_function_from_points(f_control_points)
            if 's' in self.base_config.time_functions:
                s_time_function = self.base_config.time_functions['s']
            else:
                # Use the s control function from the base config
                base_s_func = lambda t: self.base_config.control_function(t)[1]
                s_time_function = base_s_func
            from parameters import create_f_and_s_control_from_single
            control_function = create_f_and_s_control_from_single(f_control, s_time_function)

        config = ModelConfiguration(
            run_name=self.base_config.run_name,
            scalar_params=self.base_config.scalar_params,
            time_functions=self.base_config.time_functions,
            integration_params=self.base_config.integration_params,
            optimization_params=self.base_config.optimization_params,
            initial_state=self.base_config.initial_state,
            control_function=control_function
        )

        # Use store_detailed_output=False during optimization for better performance
        results = integrate_model(config, store_detailed_output=False)

        rho = self.base_config.scalar_params.rho
        t = results['t']
        U = results['U']
        L = results['L']

        discount_factors = np.exp(-rho * t)
        integrand = discount_factors * U * L

        # np.trapezoid for numerically integrates by drawing a straight line between points
        objective_value = np.trapezoid(integrand, t)

        if objective_value > self.best_objective:
            self.best_objective = objective_value
            self.best_control_values = control_values.copy()

        return objective_value

    def sensitivity_analysis(self, f_values):
        """
        Evaluate objective function at multiple fixed f values.

        Useful for understanding the objective function landscape and
        validating optimization results.

        Parameters
        ----------
        f_values : array_like
            Array of f values to evaluate (each should be in [0, 1])

        Returns
        -------
        dict
            Results containing:
            - 'f_values': input f values
            - 'objectives': corresponding objective values
            - 'n_evaluations': total evaluations performed
        """
        self.n_evaluations = 0
        control_times = [self.base_config.integration_params.t_start]

        objectives = []
        for f_val in f_values:
            obj = self.calculate_objective([f_val], control_times)
            objectives.append(obj)

        return {
            'f_values': np.array(f_values),
            'objectives': np.array(objectives),
            'n_evaluations': self.n_evaluations
        }

    def optimize_control_points(self, control_times, initial_guess, max_evaluations,
                                         algorithm=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, xtol_abs=None):
        """
        Optimize allocation with multiple control points (time-varying trajectory).

        Finds optimal control point values that maximize discounted utility.

        Parameters
        ----------
        control_times : array_like
            Times at which control points are placed
        initial_guess : array_like
            Initial guess for control values at each control time
        max_evaluations : int
            Maximum number of objective function evaluations
        algorithm : str, optional
            NLopt algorithm name (e.g., 'LN_SBPLX', 'LN_BOBYQA', 'GN_ISRES').
            If None, defaults to 'LN_SBPLX'.
        ftol_rel : float, optional
            Relative tolerance on objective function changes (None = use NLopt default)
        ftol_abs : float, optional
            Absolute tolerance on objective function changes (None = use NLopt default)
        xtol_rel : float, optional
            Relative tolerance on parameter changes (None = use NLopt default)
        xtol_abs : float, optional
            Absolute tolerance on parameter changes (None = use NLopt default)

        Returns
        -------
        dict
            Optimization results containing:
            - 'optimal_values': optimal control values at each control time
            - 'optimal_objective': maximum utility achieved
            - 'n_evaluations': number of objective evaluations used
            - 'control_points': list of (time, value) tuples
            - 'status': optimization status string
            - 'termination_code': NLopt termination code
            - 'termination_name': human-readable termination reason
            - 'algorithm': algorithm name used
        """
        self.n_evaluations = 0
        self.best_objective = -np.inf
        self.best_control_values = None
        self.degenerate_case = False
        self.degenerate_reason = None

        if algorithm is None:
            algorithm = 'LN_SBPLX'

        fract_gdp = self.base_config.scalar_params.fract_gdp
        if abs(fract_gdp) < EPSILON:
            self.degenerate_case = True
            self.degenerate_reason = "fract_gdp = 0: No income available for redistribution or abatement. Control values have no effect on outcome."
            control_times_array = np.array(control_times)
            initial_guess_array = np.array(initial_guess)
            obj = self.calculate_objective(initial_guess_array, control_times_array)
            control_points = list(zip(control_times_array, initial_guess_array))
            return {
                'optimal_values': initial_guess_array,
                'optimal_objective': obj,
                'n_evaluations': self.n_evaluations,
                'control_points': control_points,
                'status': 'degenerate',
                'termination_code': None,
                'termination_name': 'DEGENERATE_CASE',
                'algorithm': algorithm
            }

        n_points = len(control_times)
        control_times = np.array(control_times)

        def objective_wrapper(x, grad):
            """Objective function wrapper with numerical gradient computation."""
            obj_value = self.calculate_objective(x, control_times)

            # Check if objective value is valid
            if not np.isfinite(obj_value):
                print(f"WARNING: Invalid objective value: {obj_value}")
                print(f"  x: {x}")
                return -1e30  # Return large negative value for invalid objective

            # Compute gradient if requested (grad.size > 0 for gradient-based algorithms)
            if grad.size > 0:
                from constants import LOOSER_EPSILON, OBJECTIVE_SCALE
                eps = LOOSER_EPSILON
                for i in range(len(x)):
                    x_pert = x.copy()
                    x_pert[i] += eps
                    obj_pert = self.calculate_objective(x_pert, control_times)
                    grad[i] = (obj_pert - obj_value) / eps

                    # Check for NaN or Inf in gradient
                    if not np.isfinite(grad[i]):
                        grad[i] = 0.0

                # Scale gradient
                grad[:] = grad * OBJECTIVE_SCALE

            # Return scaled objective
            from constants import OBJECTIVE_SCALE
            return obj_value * OBJECTIVE_SCALE

        # Get bounds from config, default to [0.0, 1.0]
        bounds_f = self.base_config.optimization_params.bounds_f if self.base_config.optimization_params.bounds_f is not None else [0.0, 1.0]

        nlopt_algorithm = getattr(nlopt, algorithm)
        opt = nlopt.opt(nlopt_algorithm, n_points)
        opt.set_lower_bounds(np.full(n_points, bounds_f[0]))
        opt.set_upper_bounds(np.full(n_points, bounds_f[1]))
        opt.set_max_objective(objective_wrapper)
        opt.set_maxeval(max_evaluations)

        if ftol_rel is not None:
            opt.set_ftol_rel(ftol_rel)
        if ftol_abs is not None:
            opt.set_ftol_abs(ftol_abs)
        if xtol_rel is not None:
            opt.set_xtol_rel(xtol_rel)
        if xtol_abs is not None:
            opt.set_xtol_abs(xtol_abs)

        x0 = np.array(initial_guess)
        optimal_x = opt.optimize(x0)
        optimal_f = opt.last_optimum_value()
        termination_code = opt.last_optimize_result()

        termination_names = {
            1: 'SUCCESS',
            2: 'STOPVAL_REACHED',
            3: 'FTOL_REACHED',
            4: 'XTOL_REACHED',
            5: 'MAXEVAL_REACHED',
            6: 'MAXTIME_REACHED',
            -1: 'FAILURE',
            -2: 'INVALID_ARGS',
            -3: 'OUT_OF_MEMORY',
            -4: 'ROUNDOFF_LIMITED',
            -5: 'FORCED_STOP'
        }
        termination_name = termination_names.get(termination_code, f'UNKNOWN_{termination_code}')

        control_points = list(zip(control_times, optimal_x))

        return {
            'optimal_values': optimal_x,
            'optimal_objective': optimal_f,
            'n_evaluations': self.n_evaluations,
            'control_points': control_points,
            'status': 'success',
            'termination_code': termination_code,
            'termination_name': termination_name,
            'algorithm': algorithm
        }

    def optimize_control_points_f_and_s(self, f_control_times, f_initial_guess,
                                     s_control_times, s_initial_guess,
                                     max_evaluations,
                                     algorithm=None, ftol_rel=None, ftol_abs=None,
                                     xtol_rel=None, xtol_abs=None):
        """
        Optimize both f and s control points simultaneously.

        Parameters
        ----------
        f_control_times : array_like
            Times at which f control points are placed
        f_initial_guess : array_like
            Initial guess for f values at each f control time
        s_control_times : array_like
            Times at which s control points are placed
        s_initial_guess : array_like
            Initial guess for s values at each s control time
        max_evaluations : int
            Maximum number of objective function evaluations
        algorithm : str, optional
            NLopt algorithm name
        ftol_rel, ftol_abs, xtol_rel, xtol_abs : float, optional
            Tolerance parameters

        Returns
        -------
        dict
            Optimization results with separate f and s control points
        """
        self.n_evaluations = 0
        self.best_objective = -np.inf
        self.best_control_values = None
        self.degenerate_case = False
        self.degenerate_reason = None

        if algorithm is None:
            algorithm = 'LN_SBPLX'

        fract_gdp = self.base_config.scalar_params.fract_gdp
        if abs(fract_gdp) < EPSILON:
            self.degenerate_case = True
            self.degenerate_reason = "fract_gdp = 0: No income available for redistribution or abatement."
            f_control_times_array = np.array(f_control_times)
            f_initial_guess_array = np.array(f_initial_guess)
            s_control_times_array = np.array(s_control_times)
            s_initial_guess_array = np.array(s_initial_guess)
            obj = self.calculate_objective(
                f_initial_guess_array, f_control_times_array,
                s_initial_guess_array, s_control_times_array
            )
            f_control_points = list(zip(f_control_times_array, f_initial_guess_array))
            s_control_points = list(zip(s_control_times_array, s_initial_guess_array))
            return {
                'optimal_values': f_initial_guess_array,
                's_optimal_values': s_initial_guess_array,
                'optimal_objective': obj,
                'n_evaluations': self.n_evaluations,
                'control_points': f_control_points,
                's_control_points': s_control_points,
                'status': 'degenerate',
                'termination_code': None,
                'termination_name': 'DEGENERATE_CASE',
                'algorithm': algorithm
            }

        n_f_points = len(f_control_times)
        n_s_points = len(s_control_times)
        n_total_points = n_f_points + n_s_points

        f_control_times = np.array(f_control_times)
        s_control_times = np.array(s_control_times)

        def objective_wrapper(x, grad):
            """Dual optimization objective wrapper with gradient computation."""
            # Split combined vector into f and s components
            f_values = x[:n_f_points]
            s_values = x[n_f_points:]

            obj_value = self.calculate_objective(f_values, f_control_times,
                                                 s_values, s_control_times)

            # Check if objective value is valid
            if not np.isfinite(obj_value):
                print(f"WARNING: Invalid objective value: {obj_value}")
                print(f"  f_values: {f_values}")
                print(f"  s_values: {s_values}")
                return -1e30  # Return large negative value for invalid objective

            # Compute gradient if requested (grad.size > 0 for gradient-based algorithms)
            if grad.size > 0:
                from constants import LOOSER_EPSILON, OBJECTIVE_SCALE
                eps = LOOSER_EPSILON
                for i in range(len(x)):
                    x_pert = x.copy()
                    # Use smaller perturbation if we're near the upper bound
                    if i < n_f_points:  # f component
                        if x_pert[i] + eps > bounds_f[1]:
                            x_pert[i] -= eps  # Use backward difference instead
                            obj_pert = self.calculate_objective(x_pert[:n_f_points], f_control_times,
                                                                x_pert[n_f_points:], s_control_times)
                            grad[i] = (obj_value - obj_pert) / eps
                        else:
                            x_pert[i] += eps
                            obj_pert = self.calculate_objective(x_pert[:n_f_points], f_control_times,
                                                                x_pert[n_f_points:], s_control_times)
                            grad[i] = (obj_pert - obj_value) / eps
                    else:  # s component
                        if x_pert[i] + eps > bounds_s[1]:
                            x_pert[i] -= eps  # Use backward difference instead
                            obj_pert = self.calculate_objective(x_pert[:n_f_points], f_control_times,
                                                                x_pert[n_f_points:], s_control_times)
                            grad[i] = (obj_value - obj_pert) / eps
                        else:
                            x_pert[i] += eps
                            obj_pert = self.calculate_objective(x_pert[:n_f_points], f_control_times,
                                                                x_pert[n_f_points:], s_control_times)
                            grad[i] = (obj_pert - obj_value) / eps

                    # Check for NaN or Inf in gradient
                    if not np.isfinite(grad[i]):
                        grad[i] = 0.0

                # Scale gradient
                grad[:] = grad * OBJECTIVE_SCALE

            # Return scaled objective
            from constants import OBJECTIVE_SCALE
            return obj_value * OBJECTIVE_SCALE

        # Get bounds from config, default to [0.0, 1.0]
        bounds_f = self.base_config.optimization_params.bounds_f if self.base_config.optimization_params.bounds_f is not None else [0.0, 1.0]
        bounds_s = self.base_config.optimization_params.bounds_s if self.base_config.optimization_params.bounds_s is not None else [0.0, 1.0]

        # Combine bounds: [f_min, f_min, ..., s_min, s_min, ...]
        lower_bounds = np.concatenate([np.full(n_f_points, bounds_f[0]), np.full(n_s_points, bounds_s[0])])
        upper_bounds = np.concatenate([np.full(n_f_points, bounds_f[1]), np.full(n_s_points, bounds_s[1])])

        nlopt_algorithm = getattr(nlopt, algorithm)
        opt = nlopt.opt(nlopt_algorithm, n_total_points)
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
        opt.set_max_objective(objective_wrapper)
        opt.set_maxeval(max_evaluations)

        # For LD_LBFGS, set vector storage (number of corrections to store)
        # Default is often too small; use larger value for better approximation
        if algorithm == 'LD_LBFGS':
            opt.set_vector_storage(20)

        if ftol_rel is not None:
            opt.set_ftol_rel(ftol_rel)
        if ftol_abs is not None:
            opt.set_ftol_abs(ftol_abs)
        if xtol_rel is not None:
            opt.set_xtol_rel(xtol_rel)
        if xtol_abs is not None:
            opt.set_xtol_abs(xtol_abs)

        # Combine f and s initial guesses
        x0 = np.concatenate([np.array(f_initial_guess), np.array(s_initial_guess)])

        # Ensure x0 is within bounds (clip to bounds to handle floating point precision issues)
        x0 = np.clip(x0, lower_bounds, upper_bounds)

        # For gradient-based algorithms, test the gradient computation at initial point
        if algorithm.startswith('LD_') or algorithm.startswith('GD_'):
            print(f"\n  Testing gradient computation at initial point...")
            test_grad = np.zeros(len(x0))
            test_obj = objective_wrapper(x0, test_grad)
            print(f"  Initial objective value (scaled): {test_obj:.6e}")
            print(f"  Initial gradient norm: {np.linalg.norm(test_grad):.6e}")
            print(f"  Gradient min: {np.min(test_grad):.6e}, max: {np.max(test_grad):.6e}")
            if not np.all(np.isfinite(test_grad)):
                print(f"  WARNING: {np.sum(~np.isfinite(test_grad))} non-finite gradient components!")

        try:
            optimal_x = opt.optimize(x0)
            optimal_f_val = opt.last_optimum_value()
            termination_code = opt.last_optimize_result()
        except Exception as e:
            print(f"\n  ERROR during optimization:")
            print(f"    Exception type: {type(e).__name__}")
            print(f"    Exception message: {e}")
            print(f"    Algorithm: {algorithm}")
            print(f"    Initial guess: {x0}")
            print(f"    Bounds: f=[{bounds_f[0]}, {bounds_f[1]}], s=[{bounds_s[0]}, {bounds_s[1]}]")
            raise

        termination_names = {
            1: 'SUCCESS',
            2: 'STOPVAL_REACHED',
            3: 'FTOL_REACHED',
            4: 'XTOL_REACHED',
            5: 'MAXEVAL_REACHED',
            6: 'MAXTIME_REACHED',
            -1: 'FAILURE',
            -2: 'INVALID_ARGS',
            -3: 'OUT_OF_MEMORY',
            -4: 'ROUNDOFF_LIMITED',
            -5: 'FORCED_STOP'
        }
        termination_name = termination_names.get(termination_code, f'UNKNOWN_{termination_code}')

        # Split optimal values back into f and s
        optimal_f_values = optimal_x[:n_f_points]
        optimal_s_values = optimal_x[n_f_points:]

        f_control_points = list(zip(f_control_times, optimal_f_values))
        s_control_points = list(zip(s_control_times, optimal_s_values))

        return {
            'optimal_values': optimal_f_values,
            's_optimal_values': optimal_s_values,
            'optimal_objective': optimal_f_val,
            'n_evaluations': self.n_evaluations,
            'control_points': f_control_points,
            's_control_points': s_control_points,
            'status': 'success',
            'termination_code': termination_code,
            'termination_name': termination_name,
            'algorithm': algorithm
        }

    def optimize_time_adjustment(self, initial_f_control_points, initial_s_control_points,
                                max_evaluations, algorithm, ftol_rel, ftol_abs,
                                xtol_rel, xtol_abs):
        """
        Optimize the timing of control points while keeping control values fixed.

        After standard iterative refinement completes, this method adjusts the temporal
        spacing of control points to maximize the objective function. The control values
        (f and optionally s) remain fixed at their optimized values, but their timing
        is adjusted to improve the objective.

        Parameters
        ----------
        initial_f_control_points : list of tuples
            List of (time, f_value) tuples from previous optimization.
            First and last points remain fixed at t_start and t_end.
        initial_s_control_points : list of tuples or None
            List of (time, s_value) tuples from previous optimization.
            If None, only f times are optimized (single-variable mode).
            If provided, both f and s times are optimized independently.
        max_evaluations : int
            Maximum number of objective function evaluations
        algorithm : str
            NLopt algorithm name (e.g., 'LN_SBPLX')
        ftol_rel : float or None
            Relative tolerance on objective function
        ftol_abs : float or None
            Absolute tolerance on objective function
        xtol_rel : float or None
            Relative tolerance on parameters
        xtol_abs : float or None
            Absolute tolerance on parameters

        Returns
        -------
        dict
            Optimization results containing:
            - 'optimal_values': optimal f values (unchanged from input)
            - 'optimal_objective': maximum utility achieved with adjusted times
            - 'n_evaluations': number of objective evaluations used
            - 'control_points': list of (adjusted_time, f_value) tuples
            - 's_control_points': list of (adjusted_time, s_value) tuples (if s provided)
            - 'status': optimization status string
            - 'termination_code': NLopt termination code
            - 'termination_name': human-readable termination reason
            - 'algorithm': algorithm name used

        Notes
        -----
        Parameterization for each interior point n (where n = 1 to N-2):
            t_new[n] = ctrl[n-1] * (t[n+1] - t[n-1]) + t[n-1]

        where ctrl[n-1] bounds ensure minimum dt spacing:
            - Lower bound: ctrl >= dt / (t[n+1] - t[n-1])  ensures t_new[n] >= t[n-1] + dt
            - Upper bound: ctrl <= (t[n+1] - dt - t[n-1]) / (t[n+1] - t[n-1])  ensures t_new[n] <= t[n+1] - dt
            - ctrl = (t[n] - t[n-1]) / (t[n+1] - t[n-1]): stays at current position

        This ensures all control points maintain minimum spacing of dt (the integration timestep),
        preventing numerical issues while allowing the optimizer to adjust temporal placement.

        For dual optimization (f and s):
            - Total parameters: (N_f - 2) + (N_s - 2)
            - f and s times adjusted independently
            - Different numbers of control points supported
            - Both f and s times enforce dt spacing constraints
        """
        self.n_evaluations = 0
        self.best_objective = -np.inf

        f_times = np.array([pt[0] for pt in initial_f_control_points])
        f_values = np.array([pt[1] for pt in initial_f_control_points])
        n_f = len(f_times)
        n_f_interior = n_f - 2

        optimize_both = initial_s_control_points is not None

        if optimize_both:
            s_times = np.array([pt[0] for pt in initial_s_control_points])
            s_values = np.array([pt[1] for pt in initial_s_control_points])
            n_s = len(s_times)
            n_s_interior = n_s - 2
        else:
            s_times = None
            s_values = None
            n_s_interior = 0

        n_params = n_f_interior + n_s_interior

        if n_params == 0:
            result = {
                'optimal_values': f_values,
                'optimal_objective': self.calculate_objective(
                    f_values, f_times,
                    s_values if optimize_both else None,
                    s_times if optimize_both else None
                ),
                'n_evaluations': self.n_evaluations,
                'control_points': initial_f_control_points,
                'status': 'skipped',
                'termination_code': None,
                'termination_name': 'NO_INTERIOR_POINTS',
                'algorithm': algorithm
            }
            if optimize_both:
                result['s_optimal_values'] = s_values
                result['s_control_points'] = initial_s_control_points
            return result

        # Get dt constraint from config
        dt = self.base_config.integration_params.dt

        ctrl_initial = np.zeros(n_params)
        ctrl_lower = np.zeros(n_params)
        ctrl_upper = np.ones(n_params)

        # Set bounds for f control points to enforce minimum dt spacing
        for i in range(n_f_interior):
            n = i + 1
            t_left = f_times[n - 1]
            t_curr = f_times[n]
            t_right = f_times[n + 1]

            # Current fractional position
            ctrl_initial[i] = (t_curr - t_left) / (t_right - t_left)

            # Lower bound: ensure t_new[n] >= t_left + dt
            # t_new[n] = ctrl * (t_right - t_left) + t_left >= t_left + dt
            # ctrl >= dt / (t_right - t_left)
            ctrl_lower[i] = dt / (t_right - t_left)

            # Upper bound: ensure t_new[n] <= t_right - dt
            # t_new[n] = ctrl * (t_right - t_left) + t_left <= t_right - dt
            # ctrl <= (t_right - dt - t_left) / (t_right - t_left)
            ctrl_upper[i] = (t_right - dt - t_left) / (t_right - t_left)

        if optimize_both:
            # Set bounds for s control points to enforce minimum dt spacing
            for i in range(n_s_interior):
                n = i + 1
                t_left = s_times[n - 1]
                t_curr = s_times[n]
                t_right = s_times[n + 1]

                # Current fractional position
                ctrl_initial[n_f_interior + i] = (t_curr - t_left) / (t_right - t_left)

                # Lower bound: ensure minimum dt spacing from left neighbor
                ctrl_lower[n_f_interior + i] = dt / (t_right - t_left)

                # Upper bound: ensure minimum dt spacing from right neighbor
                ctrl_upper[n_f_interior + i] = (t_right - dt - t_left) / (t_right - t_left)

        # Ensure initial guess respects bounds (clip if necessary)
        ctrl_initial = np.clip(ctrl_initial, ctrl_lower, ctrl_upper)

        def reconstruct_times_and_evaluate(ctrl):
            f_times_new = np.zeros(n_f)
            f_times_new[0] = f_times[0]
            f_times_new[-1] = f_times[-1]

            for i in range(n_f_interior):
                n = i + 1
                t_left = f_times[n - 1]
                t_right = f_times[n + 1]
                f_times_new[n] = ctrl[i] * (t_right - t_left) + t_left

            if optimize_both:
                s_times_new = np.zeros(n_s)
                s_times_new[0] = s_times[0]
                s_times_new[-1] = s_times[-1]

                for i in range(n_s_interior):
                    n = i + 1
                    t_left = s_times[n - 1]
                    t_right = s_times[n + 1]
                    s_times_new[n] = ctrl[n_f_interior + i] * (t_right - t_left) + t_left
            else:
                s_times_new = None

            return self.calculate_objective(
                f_values, f_times_new,
                s_values if optimize_both else None,
                s_times_new if optimize_both else None
            )

        def objective_wrapper(x, grad):
            """Time adjustment objective wrapper with gradient computation."""
            obj_value = reconstruct_times_and_evaluate(x)

            # Check if objective value is valid
            if not np.isfinite(obj_value):
                print(f"WARNING: Invalid objective value: {obj_value}")
                print(f"  x: {x}")
                return -1e30  # Return large negative value for invalid objective

            # Compute gradient if requested (grad.size > 0 for gradient-based algorithms)
            if grad.size > 0:
                from constants import LOOSER_EPSILON, OBJECTIVE_SCALE
                eps = LOOSER_EPSILON
                for i in range(len(x)):
                    x_pert = x.copy()
                    x_pert[i] += eps
                    obj_pert = reconstruct_times_and_evaluate(x_pert)
                    grad[i] = (obj_pert - obj_value) / eps

                    # Check for NaN or Inf in gradient
                    if not np.isfinite(grad[i]):
                        grad[i] = 0.0

                # Scale gradient
                grad[:] = grad * OBJECTIVE_SCALE

            # Return scaled objective
            from constants import OBJECTIVE_SCALE
            return obj_value * OBJECTIVE_SCALE

        nlopt_algorithm = getattr(nlopt, algorithm)
        opt = nlopt.opt(nlopt_algorithm, n_params)
        opt.set_lower_bounds(ctrl_lower)
        opt.set_upper_bounds(ctrl_upper)
        opt.set_max_objective(objective_wrapper)
        opt.set_maxeval(max_evaluations)

        if ftol_rel is not None:
            opt.set_ftol_rel(ftol_rel)
        if ftol_abs is not None:
            opt.set_ftol_abs(ftol_abs)
        if xtol_rel is not None:
            opt.set_xtol_rel(xtol_rel)
        if xtol_abs is not None:
            opt.set_xtol_abs(xtol_abs)

        optimal_ctrl = opt.optimize(ctrl_initial)
        optimal_objective = opt.last_optimum_value()
        termination_code = opt.last_optimize_result()

        termination_names = {
            1: 'SUCCESS',
            2: 'STOPVAL_REACHED',
            3: 'FTOL_REACHED',
            4: 'XTOL_REACHED',
            5: 'MAXEVAL_REACHED',
            6: 'MAXTIME_REACHED',
            -1: 'FAILURE',
            -2: 'INVALID_ARGS',
            -3: 'OUT_OF_MEMORY',
            -4: 'ROUNDOFF_LIMITED',
            -5: 'FORCED_STOP'
        }
        termination_name = termination_names.get(termination_code, f'UNKNOWN_{termination_code}')

        f_times_final = np.zeros(n_f)
        f_times_final[0] = f_times[0]
        f_times_final[-1] = f_times[-1]

        for i in range(n_f_interior):
            n = i + 1
            t_left = f_times[n - 1]
            t_right = f_times[n + 1]
            f_times_final[n] = optimal_ctrl[i] * (t_right - t_left) + t_left

        f_control_points_final = list(zip(f_times_final, f_values))

        result = {
            'optimal_values': f_values,
            'optimal_objective': optimal_objective,
            'n_evaluations': self.n_evaluations,
            'control_points': f_control_points_final,
            'status': 'success',
            'termination_code': termination_code,
            'termination_name': termination_name,
            'algorithm': algorithm
        }

        if optimize_both:
            s_times_final = np.zeros(n_s)
            s_times_final[0] = s_times[0]
            s_times_final[-1] = s_times[-1]

            for i in range(n_s_interior):
                n = i + 1
                t_left = s_times[n - 1]
                t_right = s_times[n + 1]
                s_times_final[n] = optimal_ctrl[n_f_interior + i] * (t_right - t_left) + t_left

            s_control_points_final = list(zip(s_times_final, s_values))
            result['s_optimal_values'] = s_values
            result['s_control_points'] = s_control_points_final

        return result

    def optimize_with_iterative_refinement(self, n_iterations, initial_guess_scalar,
                                          max_evaluations_initial, max_evaluations_final,
                                          max_evaluations_time_points, algorithm=None,
                                          ftol_rel=None, ftol_abs=None,
                                          xtol_rel=None, xtol_abs=None,
                                          n_points_final=None,
                                          n_points_initial=2,
                                          initial_guess_s_scalar=None,
                                          n_points_final_s=None,
                                          n_points_initial_s=2,
                                          optimize_time_points=False):
        """
        Optimize using iterative refinement with progressively finer control grids.

        Performs a sequence of optimizations with increasing numbers of control points.
        Each iteration uses PCHIP interpolation of the previous solution to initialize
        the optimization, providing better convergence than cold-starting with many
        control points.

        Parameters
        ----------
        n_iterations : int
            Number of refinement iterations to perform.
            Iteration k produces round(1 + base^(k-1)) control points.
        initial_guess_scalar : float
            Initial f value for all control points in first iteration.
            Must satisfy 0 ≤ f ≤ 1.
        max_evaluations_initial : int
            Maximum objective function evaluations for first iteration
        max_evaluations_final : int
            Maximum objective function evaluations for final iteration
        max_evaluations_time_points : int
            Maximum objective function evaluations for time adjustment optimization (when optimize_time_points=True)
        algorithm : str, optional
            NLopt algorithm name. If None, defaults to 'LN_SBPLX'.
        ftol_rel : float, optional
            Relative tolerance on objective function changes
        ftol_abs : float, optional
            Absolute tolerance on objective function changes
        xtol_rel : float, optional
            Relative tolerance on parameter changes
        xtol_abs : float, optional
            Absolute tolerance on parameter changes
        n_points_final : int, optional
            Target number of f control points in final iteration.
            If specified, base = ((n_points_final - 1) / (n_points_initial - 1))^(1/(n_iterations - 1))
            If None, uses base = 2.0 (default: 2, 3, 5, 9, 17, ...)
        n_points_initial : int, optional
            Number of f control points in first iteration. Default: 2
            Used with n_points_final to determine refinement base.
        initial_guess_s_scalar : float, optional
            Initial s value for all control points in first iteration (enables dual optimization)
        n_points_final_s : int, optional
            Target number of s control points in final iteration.
            If None, uses same refinement base as f.
        n_points_initial_s : int, optional
            Number of s control points in first iteration. Default: 2
            Used with n_points_final_s to determine refinement base for s.
        optimize_time_points : bool, optional
            If True, runs time adjustment optimization after standard iterations complete.
            Adjusts temporal spacing of control points while keeping values fixed.
            Default: False (disabled for backward compatibility)

        Returns
        -------
        dict
            Optimization results containing:
            - 'optimal_values': optimal control values from final iteration
            - 'optimal_objective': maximum utility achieved
            - 'n_evaluations': total evaluations across all iterations
            - 'control_points': list of (time, value) tuples from final iteration
            - 'status': optimization status string
            - 'algorithm': algorithm name used
            - 'n_iterations': number of iterations performed
            - 'iteration_history': list of results from each iteration
            - 'iteration_control_grids': control times used at each iteration
            - 'refinement_base': base used for point growth

        Notes
        -----
        Iteration schedule (default base=2.0, n_points_initial=2):
        - Iteration 1: n_points_initial control points
        - Iteration k: round(1 + (n_points_initial - 1) * base^(k-1)) control points

        With n_points_initial=2 (default):
        - Iteration 1: 2 control points at [t_start, t_end]
        - Iteration 2: 3 control points
        - Iteration k: round(1 + base^(k-1)) control points

        Initial guess strategy:
        - First iteration: uses initial_guess_scalar for all points
        - Subsequent iterations: uses previous optimal values at existing points,
          PCHIP interpolation for new midpoints
        """
        # Determine if f and s optimization is enabled
        optimize_f_and_s = initial_guess_s_scalar is not None

        # Calculate refinement base for max_evaluations
        if n_iterations <= 1:
            refinement_base_evaluations = 2.0  # Doesn't matter, only one iteration uses max_evaluations_final
        else:
            refinement_base_evaluations = ((max_evaluations_final - 1) / (max_evaluations_initial - 1)) ** (1.0 / (n_iterations - 1))

        # Calculate refinement base for f
        if n_points_final is not None:
            if n_iterations <= 1:
                refinement_base_f = 2.0
            else:
                refinement_base_f = ((n_points_final - 1) / (n_points_initial - 1)) ** (1.0 / (n_iterations - 1))
        else:
            refinement_base_f = 2.0

        # Calculate refinement base for s (if optimizing both f and s)
        if optimize_f_and_s:
            if n_points_final_s is not None:
                if n_iterations <= 1:
                    refinement_base_s = 2.0
                else:
                    refinement_base_s = ((n_points_final_s - 1) / (n_points_initial_s - 1)) ** (1.0 / (n_iterations - 1))
            else:
                refinement_base_s = refinement_base_f  # Use same base as f if not specified

        chebyshev_scaling = self.base_config.optimization_params.chebyshev_scaling_power
        print(f"\nIterative refinement: {n_iterations} iterations, base_f = {refinement_base_f:.4f}, base_evaluations = {refinement_base_evaluations:.4f}")
        print(f"Chebyshev scaling power: {chebyshev_scaling:.2f}")
        if n_points_final is not None:
            print(f"Target final f points: {n_points_final}")
        print(f"Max evaluations: initial={max_evaluations_initial}, final={max_evaluations_final}")
        if optimize_f_and_s:
            print(f"Optimizing both f and s: base_s = {refinement_base_s:.4f}")
            if n_points_final_s is not None:
                print(f"Target final s points: {n_points_final_s}")

        iteration_history = []
        iteration_f_control_grids = []
        iteration_s_control_grids = [] if optimize_f_and_s else None
        total_evaluations = 0

        for iteration in range(1, n_iterations + 1):
            # Get algorithm for this iteration
            iteration_algorithm = self.base_config.optimization_params.get_algorithm_for_iteration(iteration)

            # Calculate max_evaluations for this iteration
            if n_iterations == 1:
                iteration_max_evaluations = max_evaluations_final
            else:
                iteration_max_evaluations = round(1 + (max_evaluations_initial - 1) * refinement_base_evaluations**(iteration - 1))

            # Calculate f control points
            if n_iterations == 1:
                n_points_f = n_points_final if n_points_final is not None else n_points_initial
            else:
                n_points_f = round(1 + (n_points_initial - 1) * refinement_base_f**(iteration - 1))
            f_control_times, f_effective_scaling = calculate_chebyshev_times(
                n_points_f,
                self.base_config.integration_params.t_start,
                self.base_config.integration_params.t_end,
                self.base_config.optimization_params.chebyshev_scaling_power,
                self.base_config.integration_params.dt
            )

            if iteration == 1:
                f_initial_guess = np.full(n_points_f, initial_guess_scalar)
            else:
                old_f_times = iteration_f_control_grids[-1]
                old_f_values = iteration_history[-1]['optimal_values']
                f_initial_guess = interpolate_to_new_grid(old_f_times, old_f_values, f_control_times)

            iteration_f_control_grids.append(f_control_times.copy())

            # Calculate s control points (if optimizing both f and s)
            s_effective_scaling = None
            if optimize_f_and_s:
                if n_iterations == 1:
                    n_points_s = n_points_final_s if n_points_final_s is not None else n_points_initial_s
                else:
                    n_points_s = round(1 + (n_points_initial_s - 1) * refinement_base_s**(iteration - 1))
                s_control_times, s_effective_scaling = calculate_chebyshev_times(
                    n_points_s,
                    self.base_config.integration_params.t_start,
                    self.base_config.integration_params.t_end,
                    self.base_config.optimization_params.chebyshev_scaling_power,
                    self.base_config.integration_params.dt
                )

                if iteration == 1:
                    s_initial_guess = np.full(n_points_s, initial_guess_s_scalar)
                else:
                    old_s_times = iteration_s_control_grids[-1]
                    old_s_values = iteration_history[-1]['s_optimal_values']
                    s_initial_guess = interpolate_to_new_grid(old_s_times, old_s_values, s_control_times)

                iteration_s_control_grids.append(s_control_times.copy())

            # Print iteration info
            print(f"\n{'=' * 80}")
            print(f"  ITERATION {iteration}/{n_iterations}")
            print(f"  Algorithm: {iteration_algorithm}")
            if requires_gradient(iteration_algorithm):
                print(f"    (gradient-based, using numerical derivatives)")
            print(f"  Max evaluations: {iteration_max_evaluations}")

            # Print effective scaling power (may be constrained by minimum spacing)
            requested_scaling = self.base_config.optimization_params.chebyshev_scaling_power
            if f_effective_scaling < requested_scaling - 1e-10:
                print(f"  Chebyshev scaling: {requested_scaling:.3f} (requested) → {f_effective_scaling:.3f} (effective, constrained by dt)")
            else:
                print(f"  Chebyshev scaling: {f_effective_scaling:.3f}")

            print(f"\n  f (abatement fraction) - OPTIMIZED:")
            print(f"    Control points: {n_points_f}")
            print(f"    Time points: {f_control_times}")

            if optimize_f_and_s:
                print(f"\n  s (savings rate) - OPTIMIZED:")
                print(f"    Control points: {n_points_s}")
                print(f"    Time points: {s_control_times}")
            print(f"{'=' * 80}\n")

            # Run optimization with timing
            iteration_start_time = time.time()

            if optimize_f_and_s:
                opt_result = self.optimize_control_points_f_and_s(
                    f_control_times,
                    f_initial_guess,
                    s_control_times,
                    s_initial_guess,
                    iteration_max_evaluations,
                    algorithm=iteration_algorithm,
                    ftol_rel=ftol_rel,
                    ftol_abs=ftol_abs,
                    xtol_rel=xtol_rel,
                    xtol_abs=xtol_abs
                )
            else:
                opt_result = self.optimize_control_points(
                    f_control_times,
                    f_initial_guess,
                    iteration_max_evaluations,
                    algorithm=iteration_algorithm,
                    ftol_rel=ftol_rel,
                    ftol_abs=ftol_abs,
                    xtol_rel=xtol_rel,
                    xtol_abs=xtol_abs
                )

            iteration_elapsed_time = time.time() - iteration_start_time

            opt_result['iteration'] = iteration
            opt_result['n_control_points'] = n_points_f
            opt_result['elapsed_time'] = iteration_elapsed_time
            if optimize_f_and_s:
                opt_result['n_s_control_points'] = n_points_s
            iteration_history.append(opt_result)
            total_evaluations += opt_result['n_evaluations']

            # Unscale objective for display (it was scaled by OBJECTIVE_SCALE in the wrapper)
            from constants import OBJECTIVE_SCALE
            unscaled_objective = opt_result['optimal_objective'] / OBJECTIVE_SCALE

            print(f"\nIteration {iteration} complete:")
            print(f"  Objective value: {unscaled_objective:.12e}")
            print(f"  Evaluations: {opt_result['n_evaluations']}")
            print(f"  Elapsed time: {iteration_elapsed_time:.2f} seconds")
            print(f"  Status: {opt_result['termination_name']}")
            print(f"\n  Optimized f values: {opt_result['optimal_values']}")
            if optimize_f_and_s:
                print(f"  Optimized s values: {opt_result['s_optimal_values']}")
            sys.stdout.flush()

        final_result = iteration_history[-1]

        if optimize_time_points:
            n_points_f = len(final_result['control_points'])
            print(f"\n{'=' * 80}")
            print(f"  TIME ADJUSTMENT OPTIMIZATION")
            print(f"  Optimizing temporal placement of {n_points_f} control points")
            print(f"  Keeping control values fixed")
            print(f"{'=' * 80}\n")

            time_opt_start_time = time.time()

            time_opt_result = self.optimize_time_adjustment(
                final_result['control_points'],
                final_result.get('s_control_points', None),
                max_evaluations_time_points,
                algorithm if algorithm is not None else 'LN_SBPLX',
                ftol_rel,
                ftol_abs,
                xtol_rel,
                xtol_abs
            )

            time_opt_elapsed_time = time.time() - time_opt_start_time

            time_opt_result['iteration'] = n_iterations + 1
            time_opt_result['n_control_points'] = len(time_opt_result['control_points'])
            time_opt_result['elapsed_time'] = time_opt_elapsed_time
            if 's_control_points' in time_opt_result and time_opt_result['s_control_points'] is not None:
                time_opt_result['n_s_control_points'] = len(time_opt_result['s_control_points'])

            # Add time-adjusted grids to the control grids lists
            f_times_adjusted = np.array([pt[0] for pt in time_opt_result['control_points']])
            iteration_f_control_grids.append(f_times_adjusted)
            if optimize_f_and_s and iteration_s_control_grids is not None:
                s_times_adjusted = np.array([pt[0] for pt in time_opt_result['s_control_points']])
                iteration_s_control_grids.append(s_times_adjusted)

            final_result = time_opt_result
            iteration_history.append(time_opt_result)
            total_evaluations += time_opt_result['n_evaluations']

            # Unscale objective for display
            from constants import OBJECTIVE_SCALE
            unscaled_objective = time_opt_result['optimal_objective'] / OBJECTIVE_SCALE

            print(f"\nTime adjustment complete:")
            print(f"  Objective value: {unscaled_objective:.12e}")
            print(f"  Evaluations: {time_opt_result['n_evaluations']}")
            print(f"  Elapsed time: {time_opt_elapsed_time:.2f} seconds")
            print(f"  Status: {time_opt_result['termination_name']}")
            print(f"\n  Optimized control points (time, f_value):")
            for pt in time_opt_result['control_points']:
                print(f"    ({float(pt[0])}, {float(pt[1])})")
            if 's_control_points' in time_opt_result and time_opt_result['s_control_points'] is not None:
                print(f"\n  Optimized s control points (time, s_value):")
                for pt in time_opt_result['s_control_points']:
                    print(f"    ({float(pt[0])}, {float(pt[1])})")
            sys.stdout.flush()

        result = {
            'optimal_values': final_result['optimal_values'],
            'optimal_objective': final_result['optimal_objective'],
            'n_evaluations': total_evaluations,
            'control_points': final_result['control_points'],
            'status': 'success',
            'algorithm': algorithm if algorithm is not None else 'LN_SBPLX',
            'n_iterations': n_iterations,
            'iteration_history': iteration_history,
            'iteration_control_grids': iteration_f_control_grids,
            'refinement_base': refinement_base_f
        }

        if optimize_f_and_s:
            result['s_optimal_values'] = final_result['s_optimal_values']
            result['s_control_points'] = final_result['s_control_points']
            result['iteration_s_control_grids'] = iteration_s_control_grids
            result['refinement_base_s'] = refinement_base_s

        return result
