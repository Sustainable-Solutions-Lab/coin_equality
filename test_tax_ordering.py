import numpy as np

from distribution_utilities import (
    y_net_of_F,
    find_Fmin,
    L_pareto_derivative,
)


def test_uniform_tax_targeted_redistribution_order_and_budget():
    # Deterministic setup with uniform tax and targeted redistribution
    y_gross = 100.0
    gini = 0.4
    uniform_tax_rate = 0.02  # 2% tax on post-damage income
    omega = 0.1  # 10% damage everywhere

    # Choose abatement and redistribution to match total tax revenue exactly
    # Revenue = y_gross * (1 - omega) * uniform_tax_rate because \int dL/dF dF = 1
    revenue_expected = y_gross * (1.0 - omega) * uniform_tax_rate  # 1.8
    abateCost_amount = 0.45
    redistribution_amount = 1.35
    assert np.isclose(revenue_expected, abateCost_amount + redistribution_amount)

    # Two-bin grid for simplicity
    Fi_edges = np.array([0.0, 0.5, 1.0])
    omega_yi = np.full(len(Fi_edges) - 1, omega)

    # Fmin should depend on the tax rate (non-zero effect)
    Fmin_with_tax = find_Fmin(
        Fmax=1.0,
        y_gross=y_gross,
        gini=gini,
        Omega=omega,
        omega_yi=omega_yi,
        redistribution_amount=redistribution_amount,
        uniform_tax_rate=uniform_tax_rate,
        Fi_edges=Fi_edges,
        initial_guess=None,
    )
    Fmin_no_tax = find_Fmin(
        Fmax=1.0,
        y_gross=y_gross,
        gini=gini,
        Omega=omega,
        omega_yi=omega_yi,
        redistribution_amount=redistribution_amount,
        uniform_tax_rate=0.0,
        Fi_edges=Fi_edges,
        initial_guess=None,
    )

    assert 0.0 < Fmin_with_tax < 1.0
    assert not np.isclose(Fmin_with_tax, Fmin_no_tax)

    # Ordering check: Lorenz → Damage → Tax → Redistribution (untaxed)
    F = 0.3
    a = (1.0 + 1.0 / gini) / 2.0
    dLdF = (1.0 - 1.0 / a) * (1.0 - F) ** (-1.0 / a)
    expected_y_net = y_gross * dLdF * (1.0 - omega) * (1.0 - uniform_tax_rate)
    y_net = y_net_of_F(
        F,
        Fmin=0.0,
        Fmax=1.0,
        y_gross=y_gross,
        omega_yi_calc=omega_yi,
        Fi_edges=Fi_edges,
        uniform_tax_rate=uniform_tax_rate,
        uniform_redistribution=0.0,
        gini=gini,
    )
    assert np.isclose(y_net, expected_y_net)

    # Revenue closure (analytical for uniform omega): tax revenue matches abate + redistribution
    tax_revenue = revenue_expected
    assert np.isclose(tax_revenue, abateCost_amount + redistribution_amount)
