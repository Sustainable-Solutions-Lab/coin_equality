"""
Test script for COIN_equality model integration.

Demonstrates loading configuration, running the forward model, and saving outputs.

Usage:
    python test_integration.py <config_file>

Arguments:
    config_file: Path to JSON configuration file (required)
"""

import sys
import os
from parameters import load_configuration
from economic_model import integrate_model
from output import save_results

def main():
    # Get config file from command line (required)
    if len(sys.argv) != 2:
        print("Usage: python test_integration.py <config_file>")
        print("Example: python test_integration.py config_baseline.json")
        sys.exit(1)

    config_file = sys.argv[1]

    # Load configuration
    print(f'Loading configuration from: {config_file}')
    config = load_configuration(config_file)

    print(f'=' * 60)
    print(f'COIN_equality Model Integration Test')
    print(f'=' * 60)
    print(f'Run name: {config.run_name}')
    print(f'Time span: {config.integration_params.t_start} to {config.integration_params.t_end} yr')
    print(f'Time step: {config.integration_params.dt} yr')
    print(f'\nScalar Parameters:')
    print(f'  alpha = {config.scalar_params.alpha}')
    print(f'  delta = {config.scalar_params.delta}')
    print(f'  eta = {config.scalar_params.eta}')
    print(f'  rho = {config.scalar_params.rho}')
    print(f'  fract_gdp = {config.scalar_params.fract_gdp}')
    print(f'\nTime-Dependent Parameters (at t=0):')
    print(f'  gini(0) = {config.time_functions["gini"](0.0)}')
    print(f'  s(0) = {config.time_functions["s"](0.0)}')

    # Run integration
    print(f'\nRunning integration...')
    results = integrate_model(config)

    # Display results
    print(f'Integration complete!')
    print(f'Number of time points: {len(results["t"])}')

    print(f'\n' + '=' * 60)
    print(f'Results Summary')
    print(f'=' * 60)

    print(f'\nInitial State (t={results["t"][0]:.1f} yr):')
    print(f'  Capital stock (K):            {results["K"][0]:.3e} $')
    print(f'  Cumulative emissions (Ecum):  {results["Ecum"][0]:.3e} tCO2')
    print(f'  Gross production (Y_gross):   {results["Y_gross"][0]:.3e} $/yr')
    print(f'  Temperature change (ΔT):      {results["delta_T"][0]:.3f} °C')
    print(f'  Emissions rate (E):           {results["E"][0]:.3e} tCO2/yr')
    print(f'  Abatement fraction (μ):       {results["mu"][0]:.4f}')
    print(f'  Effective Gini (G_eff):       {results["G_eff"][0]:.4f}')
    print(f'  Mean utility (U):             {results["U"][0]:.6f}')

    print(f'\nFinal State (t={results["t"][-1]:.1f} yr):')
    print(f'  Capital stock (K):            {results["K"][-1]:.3e} $')
    print(f'  Cumulative emissions (Ecum):  {results["Ecum"][-1]:.3e} tCO2')
    print(f'  Gross production (Y_gross):   {results["Y_gross"][-1]:.3e} $/yr')
    print(f'  Temperature change (ΔT):      {results["delta_T"][-1]:.3f} °C')
    print(f'  Emissions rate (E):           {results["E"][-1]:.3e} tCO2/yr')
    print(f'  Abatement fraction (μ):       {results["mu"][-1]:.4f}')
    print(f'  Effective Gini (G_eff):       {results["G_eff"][-1]:.4f}')
    print(f'  Mean utility (U):             {results["U"][-1]:.6f}')

    print(f'\nChanges over simulation:')
    print(f'  ΔK = {results["K"][-1] - results["K"][0]:.3e} $ ({100*(results["K"][-1]/results["K"][0] - 1):.1f}%)')
    print(f'  ΔEcum = {results["Ecum"][-1] - results["Ecum"][0]:.3e} tCO2 ({100*(results["Ecum"][-1]/results["Ecum"][0] - 1):.1f}%)')
    print(f'  ΔT_change = {results["delta_T"][-1] - results["delta_T"][0]:.3f} °C')
    print(f'  ΔU = {results["U"][-1] - results["U"][0]:.6f}')

    # Save results
    print(f'\n' + '=' * 60)
    print(f'Saving Results')
    print(f'=' * 60)
    plot_short_horizon = config.integration_params.plot_short_horizon
    config_filename_base = os.path.basename(config_file)
    output_paths = save_results(results, config.run_name, plot_short_horizon, config_filename=config_filename_base)
    print(f'Output directory: {output_paths["output_dir"]}')
    print(f'CSV file:         {output_paths["csv_file"]}')
    print(f'PDF file:         {output_paths["pdf_file"]}')
    if 'pdf_file_short' in output_paths:
        print(f'Short-term PDF:   {output_paths["pdf_file_short"]}')

    print(f'\n' + '=' * 60)

if __name__ == '__main__':
    main()
