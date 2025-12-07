import math
import numpy as np
from scipy.optimize import root_scalar, fsolve
from constants import EPSILON, LOOSE_EPSILON, MAX_ITERATIONS

# --- basic maps ---

def a_from_G(G):  # Pareto index a from Gini
    if not (0 < G < 1):
        raise ValueError("G must be in (0,1).")
    return (1.0 + 1.0/G) / 2.0

def G_from_a(a):  # Gini from Pareto index a (inverse of a_from_G)
    if a <= 1:
        raise ValueError("a must be > 1 for finite Gini.")
    return 1.0 / (2.0 * a - 1.0)

def L_pareto(F, G):  # Lorenz curve at F for Pareto-Lorenz with G
    a = a_from_G(G)
    return 1.0 - (1.0 - F)**(1.0 - 1.0/a)

def L_pareto_derivative(F, G):  # Derivative of Lorenz curve dL/dF at F for Pareto-Lorenz with G
    a = a_from_G(G)
    return (1.0 - 1.0/a) * (1.0 - F)**(-1.0/a)

def crossing_rank_from_G(Gini_initial, G2):
    if Gini_initial == G2:
        return 0.5
    r = ((1.0 - G2) * (1.0 + Gini_initial)) / ((1.0 + G2) * (1.0 - Gini_initial))
    s = ((1.0 + Gini_initial) * (1.0 + G2)) / (2.0 * (G2 - Gini_initial))
    return 1.0 - (r ** s)

def _phi(r):  # helper for bracketing cap; φ(r) = (r-1) r^{1/(r-1)-1}
    if r <= 0:
        return float("-inf")
    if abs(r - 1.0) < EPSILON:
        return 0.0
    sgn = 1.0 if r > 1.0 else -1.0
    log_abs = math.log(abs(r - 1.0)) + (1.0/(r - 1.0) - 1.0) * math.log(r)
    return sgn * math.exp(log_abs)


# Global flag to print diagnostics only on first call
_first_call_diagnostics_printed = False
_call_counter = 0

def y_of_F_after_damage(F, Fmin, Fmax, y_mean_before_damage, omega_base, y_damage_distribution_exponent, y_net_reference, uniform_redistribution, gini, branch=0):
    """
    Compute y(F) from the implicit equation

        y(F) = y_mean_before_damage * dL/dF(F; gini) + uniform_redistribution - omega_base * (y(F) / y_net_reference)**y_damage_distribution_exponent,

    where the Lorenz curve is Pareto with Gini index gini:

        L(F) = 1 - (1-F)^(1 - 1/a),
        a    = (1 + 1/gini)/2,
        dL/dF(F) = (1 - 1/a) * (1 - F)^(-1/a).

    The implicit equation is solved numerically using a root finder.

    Parameters
    ----------
    F : float or array-like
        Population rank(s) in [0,1].
    Fmin : float
        Minimum population rank for income in [0,1].
    Fmax : float
        Maximum population rank for income in [0,1].
    y_mean_before_damage : float
        Mean income.
    omega_base : float
        Maximum damage scale.
    y_damage_distribution_exponent : float
        Damage distribution exponent.
    y_net_reference : float
        Reference income for power-law damage scaling ($/person).
    uniform_redistribution : float
        Additive constant in A(F).
    gini : float
        Gini index (0 < gini < 1).
    branch : int, optional
        Unused parameter (kept for backward compatibility).

    Returns
    -------
    y_of_F : float or ndarray
        y(F) evaluated at the given F values.
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

    # A(F)
    A = y_mean_before_damage * dLdF + uniform_redistribution

    # Handle y_damage_distribution_exponent ≈ 0 case
    if np.abs(y_damage_distribution_exponent) < EPSILON:
        result = A - omega_base
        return result[0] if is_scalar else result

    # Handle y_damage_distribution_exponent ≈ 0.5 case (analytic solution via quadratic formula)
    # Implicit equation: y = A - omega_base * (y / y_net_reference)^0.5
    # Substituting t = sqrt(y): t^2 + B*t - A = 0, where B = omega_base / sqrt(y_net_reference)
    # Solution: t = (-B + sqrt(B^2 + 4A)) / 2, then y = t^2
    if np.abs(y_damage_distribution_exponent - 0.5) < EPSILON:
        B = omega_base / np.sqrt(y_net_reference)
        discriminant = B**2 + 4.0 * A
        t = (-B + np.sqrt(discriminant)) / 2.0
        result = t**2
        return result[0] if is_scalar else result

    # Solve implicit equation: y = A - omega_base * (y / y_net_reference)**y_damage_distribution_exponent
    def equation(y, A_val):
        return y - A_val + omega_base * (y / y_net_reference)**y_damage_distribution_exponent

    # Solve for each element with relaxed tolerances to avoid false convergence warnings
    y_solution = np.zeros_like(A)
    convergence_issues = []
    total_fev = 0
    first_point_history = None

    for i in range(len(A)):
        y_guess = np.maximum(A[i] - omega_base, EPSILON)

        # Always track iteration history for first point (for diagnostics if needed)
        if i == 0:
            # Custom solver with iteration tracking
            iteration_history = []
            def tracked_equation(y):
                result = equation(y, A[i])
                iteration_history.append({'y': float(y), 'residual': float(result)})
                return result

            result, info, ier, mesg = fsolve(
                tracked_equation, y_guess,
                full_output=True,
                xtol=LOOSE_EPSILON,
                maxfev=MAX_ITERATIONS
            )
            first_point_history = iteration_history
        else:
            result, info, ier, mesg = fsolve(
                equation, y_guess, args=(A[i],),
                full_output=True,
                xtol=LOOSE_EPSILON,
                maxfev=MAX_ITERATIONS
            )

        y_solution[i] = result[0]
        total_fev += info['nfev']

        # Only report convergence issues if residual is actually large
        residual = abs(info['fvec'][0])
        if ier != 1 and residual > LOOSE_EPSILON:
            convergence_issues.append({
                'index': i,
                'A': A[i],
                'y_guess': y_guess,
                'y_solution': result[0],
                'final_residual': residual,
                'n_calls': info['nfev'],
                'message': mesg
            })

    # Print diagnostic info on first call or if convergence is slow
    avg_fev = total_fev / len(A) if len(A) > 0 else 0
    is_slow = avg_fev > 15

    if not _first_call_diagnostics_printed or is_slow:
        if not _first_call_diagnostics_printed:
            _first_call_diagnostics_printed = True
            header = "FIRST CALL"
        else:
            header = f"SLOW CONVERGENCE (call #{_call_counter})"

        print(f"\n=== y_of_F_after_damage {header} diagnostics ===")
        print(f"Parameters:")
        print(f"  omega_base={omega_base:.6e}")
        print(f"  y_damage_distribution_exponent={y_damage_distribution_exponent:.4f}")
        print(f"  y_net_reference={y_net_reference:.2f}")
        print(f"  gini={gini:.4f}")
        print(f"  y_mean_before_damage={y_mean_before_damage:.2f}")
        print(f"Convergence stats:")
        print(f"  Number of points: {len(A)}")
        print(f"  Total function evaluations: {total_fev}")
        print(f"  Average function evals per point: {avg_fev:.1f}")
        print(f"  Convergence issues (residual > {LOOSE_EPSILON:.1e}): {len(convergence_issues)}")

        if first_point_history:
            print(f"\nIteration history for first point (A[0]={A[0]:.4e}):")
            print(f"  Initial guess: y={first_point_history[0]['y']:.6e}")
            print(f"  Iterations:")
            for idx, step in enumerate(first_point_history[:20]):  # Limit to first 20 iterations
                print(f"    {idx:3d}: y={step['y']:12.6e}, residual={step['residual']:12.6e}")
            if len(first_point_history) > 20:
                print(f"    ... ({len(first_point_history) - 20} more iterations)")
            print(f"  Final solution: y={y_solution[0]:.6e}")

        if convergence_issues:
            print(f"\nProblem cases:")
            for issue in convergence_issues[:10]:
                print(f"  Index {issue['index']}: A={issue['A']:.4e}, "
                      f"residual={issue['final_residual']:.4e}, calls={issue['n_calls']}")
        print("="*50 + "\n")

    return y_solution[0] if is_scalar else y_solution


def segment_integral_with_cut(
    Flo,
    Fhi,
    Fcut,
    Fmin,
    Fmax_for_clip,
    y_mean_before_damage,
    omega_base,
    y_damage_distribution_exponent,
    y_net_reference,
    uniform_redistribution,
    gini,
    xi,
    wi,
    cut_at,
    branch=0,
):
    """
    Compute ∫_{Flo}^{Fhi} [ y(F; Fmin, Fmax_for_clip, ...) - y(Fcut; Fmin, Fmax_for_clip, ...) ] dF

    Generic function for computing integrals over income distribution segments with a reference cut.
    Used for both taxation (upper tail) and redistribution (lower tail) calculations.

    Parameters
    ----------
    Flo : float
        Lower integration bound.
    Fhi : float
        Upper integration bound.
    Fcut : float
        Rank where reference income y(Fcut) is evaluated.
    Fmin : float
        Minimum rank for clipping in y_of_F_after_damage.
    Fmax_for_clip : float
        Maximum rank for clipping in y_of_F_after_damage.
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_exponent : float
        Damage distribution coefficient parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    branch : int, optional
        Lambert W branch (default 0).
    cut_at : str
        Semantic label: "upper" for taxation, "lower" for redistribution.

    Returns
    -------
    float
        Integral value.
    """
    # Map Gauss-Legendre nodes from [-1, 1] to [Flo, Fhi]
    F_nodes = 0.5 * (Fhi - Flo) * xi + 0.5 * (Fhi + Flo)
    w_nodes = 0.5 * (Fhi - Flo) * wi

    # y(F) over the segment, using same Fmin and Fmax_for_clip
    y_vals = y_of_F_after_damage(
        F_nodes,
        Fmin,
        Fmax_for_clip,
        y_mean_before_damage,
        omega_base,
        y_damage_distribution_exponent,
        y_net_reference,
        uniform_redistribution,
        gini,
        branch=branch,
    )

    # reference value y(Fcut)
    y_cut = y_of_F_after_damage(
        Fcut,
        Fmin,
        Fmax_for_clip,
        y_mean_before_damage,
        omega_base,
        y_damage_distribution_exponent,
        y_net_reference,
        uniform_redistribution,
        gini,
        branch=branch,
    )

    # For redistribution (lower tail): y(Fcut) - y(F) gives positive values (poor have less income)
    # For taxation (upper tail): y(F) - y(Fcut) gives positive values (rich have more income)
    if cut_at == "lower":
        integrand = y_cut - y_vals
    else:
        integrand = y_vals - y_cut
    integral_val = np.dot(w_nodes, integrand)

    return integral_val


def total_tax_top(
    Fmax,
    Fmin,
    y_mean_before_damage,
    omega_base,
    y_damage_distribution_exponent,
    y_net_reference,
    uniform_redistribution,
    gini,
    xi,
    wi,
    target_tax=0.0,
    branch=0,
):
    """
    Compute ∫_{Fmax}^{1} [ y(F; Fmin, 1, ...) - y(Fmax; Fmin, Fmax, ...) ] dF - target_tax

    This is the function we will set to zero in root finding for taxation.

    Parameters
    ----------
    Fmax : float
        Upper boundary for taxation (income ranks above Fmax are taxed).
    Fmin : float
        Lower boundary for income distribution.
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_exponent : float
        Damage distribution coefficient parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    target_tax : float, optional
        Target tax amount to subtract (default 0.0).
    branch : int, optional
        Lambert W branch (default 0).

    Returns
    -------
    float
        Integral value minus target_tax (for root finding).
    """
    integral_val = segment_integral_with_cut(
        Flo=Fmax,
        Fhi=1.0,
        Fcut=Fmax,
        Fmin=Fmin,
        Fmax_for_clip=1.0,  # we want F clipped to [Fmin, 1]
        y_mean_before_damage=y_mean_before_damage,
        omega_base=omega_base,
        y_damage_distribution_exponent=y_damage_distribution_exponent,
        y_net_reference=y_net_reference,
        uniform_redistribution=uniform_redistribution,
        gini=gini,
        xi=xi,
        wi=wi,
        cut_at="upper",
        branch=branch,
    )

    return integral_val - target_tax


def total_tax_bottom(
    Fmin,
    y_mean_before_damage,
    omega_base,
    y_damage_distribution_exponent,
    y_net_reference,
    uniform_redistribution,
    gini,
    xi,
    wi,
    target_subsidy=0.0,
    branch=0,
):
    """
    Compute ∫_{0}^{Fmin} [ y(Fmin; 0, Fmin, ...) - y(F; 0, Fmin, ...) ] dF - target_subsidy

    This is the function we will set to zero in root finding for redistribution.

    Parameters
    ----------
    Fmin : float
        Lower boundary for redistribution (income ranks below Fmin receive redistribution).
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_exponent : float
        Damage distribution coefficient parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    target_subsidy : float, optional
        Target subsidy amount to subtract (default 0.0).
    branch : int, optional
        Lambert W branch (default 0).

    Returns
    -------
    float
        Integral value minus target_subsidy (for root finding).
    """
    integral_val = segment_integral_with_cut(
        Flo=0.0,
        Fhi=Fmin,
        Fcut=Fmin,
        Fmin=0.0,              # model Fmin as bottom of support
        Fmax_for_clip=Fmin,    # clip inside [0, Fmin]
        y_mean_before_damage=y_mean_before_damage,
        omega_base=omega_base,
        y_damage_distribution_exponent=y_damage_distribution_exponent,
        y_net_reference=y_net_reference,
        uniform_redistribution=uniform_redistribution,
        gini=gini,
        xi=xi,
        wi=wi,
        cut_at="lower",
        branch=branch,
    )

    return integral_val - target_subsidy


def find_Fmax(Fmin,
              y_mean_before_damage,
              omega_base,
              y_damage_distribution_exponent,
              y_net_reference,
              uniform_redistribution,
              gini,
              xi,
              wi,
              target_tax=0.0,
              branch=0,
              tol=LOOSE_EPSILON):
    """
    Find Fmax in [Fmin, 1) such that total_tax_top(Fmax) = target_tax.

    Uses a bracketing root-finder.

    Parameters
    ----------
    Fmin : float
        Lower boundary for income distribution.
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_exponent : float
        Damage distribution coefficient parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    target_tax : float, optional
        Target tax amount (default 0.0).
    branch : int, optional
        Lambert W branch (default 0).
    tol : float, optional
        Tolerance for root finding (default LOOSE_EPSILON = 1e-8).

    Returns
    -------
    float
        Fmax value such that total_tax_top(Fmax) = target_tax.
    """
    # Define a wrapper with all parameters bound
    def f(Fmax):
        return total_tax_top(
            Fmax,
            Fmin,
            y_mean_before_damage,
            omega_base,
            y_damage_distribution_exponent,
            y_net_reference,
            uniform_redistribution,
            gini,
            xi,
            wi,
            target_tax=target_tax,
            branch=branch,
        )

    # For extremely small target_tax (relative to mean income), Fmax must be very close to 1
    # where the Pareto distribution is singular. In this case, just return value close to 1.
    if target_tax < EPSILON * y_mean_before_damage:
        return 1.0 - EPSILON

    # Bracket Fmax between Fmin and just below 1 (to avoid Pareto singularity at F=1)
    left = Fmin
    right = 1.0 - EPSILON

    f_left = f(left)
    f_right = f(right)

    if f_left * f_right > 0:
        raise RuntimeError(
            f"Root not bracketed: total_tax_top(Fmin)={f_left}, total_tax_top({right})={f_right}\n"
            f"  Fmin={Fmin}\n"
            f"  target_tax={target_tax}\n"
            f"  y_mean_before_damage={y_mean_before_damage}\n"
            f"  omega_base={omega_base}\n"
            f"  uniform_redistribution={uniform_redistribution}"
        )

    sol = root_scalar(f, bracket=[left, right], method="brentq", xtol=tol)
    if not sol.converged:
        raise RuntimeError("root_scalar did not converge for find_Fmax")

    return sol.root


def find_Fmin(y_mean_before_damage,
              omega_base,
              y_damage_distribution_exponent,
              y_net_reference,
              uniform_redistribution,
              gini,
              xi,
              wi,
              target_subsidy=0.0,
              branch=0,
              tol=LOOSE_EPSILON):
    """
    Find Fmin in (0, 1) such that total_tax_bottom(Fmin) = target_subsidy.

    Uses a bracketing root-finder.

    Parameters
    ----------
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_exponent : float
        Damage distribution coefficient parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    target_subsidy : float, optional
        Target subsidy amount (default 0.0).
    branch : int, optional
        Lambert W branch (default 0).
    tol : float, optional
        Tolerance for root finding (default LOOSE_EPSILON = 1e-8).

    Returns
    -------
    float
        Fmin value such that total_tax_bottom(Fmin) = target_subsidy.
    """
    # Define a wrapper with all parameters bound
    def f(Fmin):
        return total_tax_bottom(
            Fmin,
            y_mean_before_damage,
            omega_base,
            y_damage_distribution_exponent,
            y_net_reference,
            uniform_redistribution,
            gini,
            xi,
            wi,
            target_subsidy=target_subsidy,
            branch=branch,
        )

    # For extremely small target_subsidy (relative to mean income), Fmin must be very close to 0.
    # In this case, just return a value close to 0.
    if target_subsidy < EPSILON * y_mean_before_damage:
        return EPSILON

    # Bracket Fmin between 0 and something less than 1
    left = 0.0
    right = 0.999999

    f_left = f(left)
    f_right = f(right)

    if f_left * f_right > 0:
        raise RuntimeError(
            f"Root not bracketed: total_tax_bottom(0.0)={f_left}, total_tax_bottom(0.999999)={f_right}\n"
            f"  target_subsidy={target_subsidy}\n"
            f"  y_mean_before_damage={y_mean_before_damage}\n"
            f"  omega_base={omega_base}\n"
            f"  uniform_redistribution={uniform_redistribution}"
        )

    sol = root_scalar(f, bracket=[left, right], method="brentq", xtol=tol)
    if not sol.converged:
        raise RuntimeError("root_scalar did not converge for find_Fmin")

    return sol.root
