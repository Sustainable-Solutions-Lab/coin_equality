"""
Output functions for COIN_equality model.

Creates CSV files and PDF plots of model results in timestamped directories.
"""

import os
import csv
import shutil
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter


# Variable metadata for better plot labels and descriptions
VARIABLE_METADATA = {
    't': {'description': 'Time', 'units': 'yr', 'group': 'time'},
    'A': {'description': 'Total Factor Productivity', 'units': '', 'group': 'economic'},
    'E': {'description': 'Actual Emissions', 'units': 'tCO₂/yr', 'group': 'climate'},
    'Ecum': {'description': 'Cumulative Emissions', 'units': 'tCO₂', 'group': 'climate'},
    'K': {'description': 'Capital Stock', 'units': '$', 'group': 'economic'},
    'L': {'description': 'Population', 'units': 'people', 'group': 'economic'},
    'Lambda': {'description': 'Abatement Cost Fraction', 'units': '', 'group': 'abatement'},
    'Omega': {'description': 'Climate Damage Fraction', 'units': '', 'group': 'climate'},
    'U': {'description': 'Mean Population Utility', 'units': '', 'group': 'inequality'},
    'Y_damaged': {'description': 'GDP After Climate Damage', 'units': '$', 'group': 'economic'},
    'Y_gross': {'description': 'Gross GDP', 'units': '$', 'group': 'economic'},
    'Y_net': {'description': 'Net GDP', 'units': '$', 'group': 'economic'},
    'AbateCost': {'description': 'Abatement Expenditure', 'units': '$', 'group': 'abatement'},
    'dEcum_dt': {'description': 'Emissions Rate', 'units': 'tCO₂/yr', 'group': 'climate'},
    'dK_dt': {'description': 'Capital Growth Rate', 'units': '$/yr', 'group': 'economic'},
    'delta_T': {'description': 'Temperature Change', 'units': '°C', 'group': 'climate'},
    'redistribution': {'description': 'Per-Capita Redistribution Amount', 'units': '$/person', 'group': 'inequality'},
    'Redistribution_amount': {'description': 'Total Redistribution Amount', 'units': '$/yr', 'group': 'inequality'},
    'f': {'description': 'Abatement Allocation Fraction', 'units': '', 'group': 'policy'},
    'mu': {'description': 'Emissions Abatement Fraction', 'units': '', 'group': 'abatement'},
    'sigma': {'description': 'Carbon Intensity of GDP', 'units': 'tCO₂/$', 'group': 'climate'},
    'theta1': {'description': 'Marginal Abatement Cost at mu=1', 'units': '$/tCO₂', 'group': 'abatement'},
    'marginal_abatement_cost': {'description': 'Marginal Abatement Cost', 'units': '$/tCO₂', 'group': 'abatement'},
    'y_damaged': {'description': 'Per-Capita Income After Climate Damage', 'units': '$/person', 'group': 'economic'},
    'y_net': {'description': 'Per-Capita Net Income', 'units': '$/person', 'group': 'economic'},
    'Consumption': {'description': 'Total Consumption', 'units': '$/yr', 'group': 'economic'},
    'Savings': {'description': 'Gross Investment', 'units': '$/yr', 'group': 'economic'},
    's': {'description': 'Savings Rate', 'units': '', 'group': 'policy'},
    'gini': {'description': 'Background Gini Index', 'units': '', 'group': 'inequality'},
    'Gini': {'description': 'Gini Index', 'units': '', 'group': 'inequality'}
}

# Variable grouping for organized layout with combined charts
# Ordered by: dimensionless ratios, dollar variables, everything else, specified functions
VARIABLE_GROUPS = {
    'dimensionless_ratios': [
        {'type': 'combined', 'title': 'Control Variables', 'variables': ['f', 's'], 'units': 'fraction'},
        {'type': 'single', 'variables': ['mu']},
        {'type': 'combined', 'title': 'Economic Impact Fractions', 'variables': ['Omega', 'Lambda'], 'units': 'fraction'},
        {'type': 'single', 'variables': ['Gini']},
        {'type': 'single', 'variables': ['U']}
    ],
    'dollar_variables': [
        {'type': 'combined', 'title': 'GDP Components', 'variables': ['Y_gross', 'Y_damaged', 'Y_net'], 'units': '$'},
        {'type': 'single', 'variables': ['K']},
        {'type': 'single', 'variables': ['Consumption']},
        {'type': 'single', 'variables': ['Savings']},
        {'type': 'single', 'variables': ['AbateCost']},
        {'type': 'single', 'variables': ['dK_dt']},
        {'type': 'single', 'variables': ['y_net']},
        {'type': 'combined', 'title': 'Redistribution', 'variables': ['redistribution', 'Redistribution_amount'], 'units': '$'}
    ],
    'physical_variables': [
        {'type': 'combined', 'title': 'Emissions', 'variables': ['E', 'dEcum_dt'], 'units': 'tCO₂/yr'},
        {'type': 'single', 'variables': ['Ecum']},
        {'type': 'single', 'variables': ['delta_T']},
        {'type': 'single', 'variables': ['marginal_abatement_cost']}
    ],
    'specified_functions': [
        {'type': 'single', 'variables': ['L']},
        {'type': 'single', 'variables': ['A']},
        {'type': 'single', 'variables': ['sigma']},
        {'type': 'single', 'variables': ['theta1']}
    ]
}


def format_scientific_notation(x, pos=None):
    """Custom formatter for scientific notation with proper spacing."""
    if x == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10 ** exp)
    if abs(exp) <= 2:
        return f'{x:.3g}'
    else:
        return f'{mantissa:.1f}×10^{exp}'


def create_output_directory(run_name):
    """
    Create timestamped output directory.

    Parameters
    ----------
    run_name : str
        Name of the model run

    Returns
    -------
    str
        Path to created output directory

    Notes
    -----
    Directory format: ./data/output/{run_name}_YYYYMMDD-HHMMSS
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join('data', 'output', f'{run_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def copy_config_file(config_path, output_dir):
    """
    Copy configuration JSON file to output directory.

    Parameters
    ----------
    config_path : str
        Path to input configuration file
    output_dir : str
        Directory to copy file to

    Returns
    -------
    str
        Path to copied configuration file

    Notes
    -----
    Preserves the original filename and formatting.
    Ensures reproducibility by keeping exact copy of configuration used.
    """
    filename = os.path.basename(config_path)
    output_path = os.path.join(output_dir, filename)
    shutil.copy2(config_path, output_path)
    return output_path


def write_optimization_summary(opt_results, sensitivity_results, output_dir, filename='optimization_summary.csv'):
    """
    Write optimization summary statistics to CSV file.

    Parameters
    ----------
    opt_results : dict
        Optimization results from UtilityOptimizer
    sensitivity_results : dict or None
        Sensitivity analysis results (optional)
    output_dir : str
        Directory to write CSV file
    filename : str
        Name of CSV file

    Returns
    -------
    str
        Path to created CSV file

    Notes
    -----
    Creates a CSV file with optimization statistics including:
    - Optimal control point values
    - Objective function value
    - Number of function evaluations
    - Convergence status
    - Iteration-by-iteration results (for iterative refinement mode)
    - Sensitivity analysis statistics (if provided)
    """
    csv_path = os.path.join(output_dir, filename)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Optimization Summary'])
        writer.writerow([])

        writer.writerow(['Parameter', 'Value'])

        # Check if this is dual optimization (has s values)
        is_dual = 's_optimal_values' in opt_results and opt_results['s_optimal_values'] is not None

        # Write optimal f values
        for i, val in enumerate(opt_results['optimal_values']):
            writer.writerow([f'Optimal f value at control point {i}', f"{val:.6f}"])

        # Write optimal s values if dual optimization
        if is_dual:
            for i, val in enumerate(opt_results['s_optimal_values']):
                writer.writerow([f'Optimal s value at control point {i}', f"{val:.6f}"])

        writer.writerow(['Optimal objective', f"{opt_results['optimal_objective']:.12e}"])
        writer.writerow(['Function evaluations', opt_results['n_evaluations']])
        writer.writerow(['Status', opt_results['status']])

        if 'algorithm' in opt_results:
            writer.writerow(['Algorithm', opt_results['algorithm']])
        if 'n_iterations' in opt_results:
            writer.writerow(['Number of iterations', opt_results['n_iterations']])
        if 'termination_name' in opt_results:
            writer.writerow(['Termination reason', opt_results['termination_name']])
        if 'termination_code' in opt_results and opt_results['termination_code'] is not None:
            writer.writerow(['Termination code', opt_results['termination_code']])

        if opt_results['status'] == 'degenerate':
            writer.writerow([])
            writer.writerow(['WARNING: DEGENERATE CASE'])
            writer.writerow(['Reason', 'No income available for redistribution or abatement (fract_gdp = 0)'])
            writer.writerow(['Note', 'Control values have no effect on outcome. Returning initial guess.'])
        writer.writerow([])

        # Write final control points for f
        writer.writerow(['Final f(t) Control Points'])
        writer.writerow(['Time', 'f Value'])
        for time, value in opt_results['control_points']:
            writer.writerow([f"{time:.2f}", f"{value:.6f}"])

        # Write final control points for s if dual optimization
        if is_dual and 's_control_points' in opt_results:
            writer.writerow([])
            writer.writerow(['Final s(t) Control Points'])
            writer.writerow(['Time', 's Value'])
            for time, value in opt_results['s_control_points']:
                writer.writerow([f"{time:.2f}", f"{value:.6f}"])

        if 'iteration_history' in opt_results:
            writer.writerow([])
            writer.writerow(['Iterative Refinement - Iteration History'])
            writer.writerow(['Iteration', 'Control_Points', 'Objective', 'Evaluations', 'Status'])
            for iter_result in opt_results['iteration_history']:
                writer.writerow([
                    iter_result['iteration'],
                    iter_result['n_control_points'],
                    f"{iter_result['optimal_objective']:.12e}",
                    iter_result['n_evaluations'],
                    iter_result['termination_name']
                ])

            writer.writerow([])
            writer.writerow(['Iterative Refinement - Control Values by Iteration'])

            max_points = max(iter_result['n_control_points'] for iter_result in opt_results['iteration_history'])
            header = ['Iteration', 'Control_Points'] + [f'f_{i}' for i in range(max_points)]
            writer.writerow(header)

            for iter_result in opt_results['iteration_history']:
                row = [
                    iter_result['iteration'],
                    iter_result['n_control_points']
                ]
                for val in iter_result['optimal_values']:
                    row.append(f"{val:.6f}")
                for _ in range(max_points - len(iter_result['optimal_values'])):
                    row.append('')
                writer.writerow(row)

            writer.writerow([])
            writer.writerow(['Iterative Refinement - Control Times by Iteration'])
            header = ['Iteration', 'Control_Points'] + [f't_{i}' for i in range(max_points)]
            writer.writerow(header)

            for i, iter_result in enumerate(opt_results['iteration_history']):
                control_times = opt_results['iteration_control_grids'][i]
                row = [
                    iter_result['iteration'],
                    iter_result['n_control_points']
                ]
                for time in control_times:
                    row.append(f"{time:.2f}")
                for _ in range(max_points - len(control_times)):
                    row.append('')
                writer.writerow(row)

        if sensitivity_results:
            writer.writerow([])
            writer.writerow(['Sensitivity Analysis'])
            writer.writerow(['f_Value', 'Objective'])
            for f_val, obj in zip(sensitivity_results['f_values'], sensitivity_results['objectives']):
                writer.writerow([f"{f_val:.6f}", f"{obj:.6e}"])

    return csv_path


def write_results_csv(results, output_dir, filename='results.csv'):
    """
    Write results dictionary to CSV file.

    Parameters
    ----------
    results : dict
        Results dictionary from integrate_model()
    output_dir : str
        Directory to write CSV file
    filename : str
        Name of CSV file

    Returns
    -------
    str
        Path to created CSV file

    Notes
    -----
    Each column is a variable, each row is a time point.
    First row contains variable names (header).
    """
    csv_path = os.path.join(output_dir, filename)

    # Define column order as specified
    ordered_columns = [
        't',  # Time
        # Time-dependent driving parameters
        'A',  # Total factor productivity
        'L',  # Population
        'theta1',  # Marginal abatement cost at mu=1
        'sigma',  # Carbon intensity of GDP
        # Decision variables
        'f',  # Control: fraction allocated to abatement
        's',  # Savings rate
        # State variables
        'K',  # Capital stock
        'gini',  # Background Gini index
        'Gini',  # Gini index
        'Ecum',  # Cumulative emissions
        # Related to Ecum
        'dEcum_dt',  # Rate of cumulative emissions change
        'delta_T',  # Global mean temperature change
        # Production chain
        'Y_gross',  # Gross production before climate damage
        'Climate_damage',  # Total climate damage (Y_gross * Omega)
        'Y_damaged',  # Output after climate damage before abatement
        # Costs and net
        'AbateCost',  # Total abatement expenditure
        'Y_net',  # Output net of climate damage and abatement cost
        # Investment and Consumption
        'Savings',  # Gross investment
        'Consumption',  # Total Consumption
        # Per capita indices
        'y_net',  # Effective per-capita income after abatement
        'redistribution',  # Per-capita redistribution amount
        'Redistribution_amount',  # Total redistribution amount
        'U',  # Mean utility per capita
        # Dimensionless variables
        'Omega',  # Climate damage as fraction of gross output
        'Lambda',  # Abatement cost as fraction of damaged output
        # Remaining variables
        'mu',  # Abatement fraction
        'E',  # CO2 emissions
        'dK_dt',  # Net capital accumulation
        'marginal_abatement_cost',  # Marginal abatement cost (social cost of carbon)
    ]

    # Add any remaining variables not in the ordered list
    remaining_vars = sorted([k for k in results.keys() if k not in ordered_columns])
    var_names = ordered_columns + remaining_vars

    # Define variable descriptions and units
    var_info = {
        't': ('Time', 'yr'),
        'A': ('Total factor productivity', '$'),
        'mu': ('Abatement fraction', 'dimensionless'),
        'theta1': ('Marginal abatement cost at mu=1', '$/tCO2'),
        'sigma': ('Carbon intensity of GDP', 'tCO2/$'),
        'L': ('Population', 'persons'),
        'Ecum': ('Cumulative CO2 emissions', 'tCO2'),
        'delta_T': ('Global mean temperature change', '°C'),
        'Omega': ('Climate damage as fraction of gross output', 'dimensionless'),
        'Climate_damage': ('Total climate damage', '$/yr'),
        'Y_damaged': ('Output after climate damage before abatement', '$/yr'),
        'AbateCost': ('Total abatement expenditure', '$/yr'),
        'Y_net': ('Output net of climate damage and abatement cost', '$/yr'),
        'E': ('CO2 emissions', 'tCO2/yr'),
        'Savings': ('Gross investment', '$/yr'),
        'K': ('Capital stock', '$'),
        'Consumption': ('Total Consumption', '$/yr'),
        'dK_dt': ('Net capital accumulation', '$/yr'),
        'y_net': ('Per-capita net income', '$/person/yr'),
        'Gini': ('Gini index', 'dimensionless'),
        'gini': ('Background Gini index', 'dimensionless'),
        'redistribution': ('Per-capita redistribution amount', '$/person/yr'),
        'Redistribution_amount': ('Total redistribution amount', '$/yr'),
        'U': ('Mean utility per capita', 'dimensionless'),
        's': ('Savings rate', 'dimensionless'),
        'f': ('Control: fraction allocated to abatement vs redistribution', 'dimensionless'),
        'marginal_abatement_cost': ('Marginal abatement cost (social cost of carbon)', '$/tCO2'),
        'Y_gross': ('Gross production before climate damage', '$/yr'),
        'y_net': ('Effective per-capita income after abatement', '$/person/yr'),
        'Lambda': ('Abatement cost as fraction of damaged output', 'dimensionless'),
        'redistribution': ('Per-capita redistributable income', '$/person/yr'),
        'dEcum_dt': ('Rate of cumulative emissions change', 'tCO2/yr'),
    }

    # Create headers with format: "variable, description, (units)"
    headers = []
    for var in var_names:
        if var in var_info:
            desc, units = var_info[var]
            headers.append(f"{var}, {desc}, ({units})")
        else:
            # Fallback for any variables not in the dictionary
            headers.append(var)

    # Open CSV file and write
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(headers)

        # Get number of time points
        n_points = len(results['t'])

        # Write data rows
        for i in range(n_points):
            row = [results[var][i] for var in var_names]
            writer.writerow(row)

    return csv_path


def plot_results_pdf(results, output_dir, run_name, filename='plots.pdf', config_filename=None):
    """
    Create PDF with time series plots of all variables, organized by topic with combined charts.

    Parameters
    ----------
    results : dict
        Results dictionary from integrate_model()
    output_dir : str
        Directory to write PDF file
    run_name : str
        Name of the model run to display in header
    filename : str
        Name of PDF file
    config_filename : str, optional
        Name of configuration file to display on each page

    Returns
    -------
    str
        Path to created PDF file

    Notes
    -----
    Creates multi-page PDF organized by variable groups (economic, climate, etc.).
    Supports both single-variable and multi-variable combined charts.
    """
    pdf_path = os.path.join(output_dir, filename)

    # Get time array
    t = results['t']

    # Create PDF with organized plots by groups
    with PdfPages(pdf_path) as pdf:

        # Plot each group on separate pages
        for group_name, chart_specs in VARIABLE_GROUPS.items():
            # Filter chart specs to only include those with available variables
            available_charts = []
            for spec in chart_specs:
                available_vars = [var for var in spec['variables'] if var in results]
                if available_vars:
                    spec_copy = spec.copy()
                    spec_copy['variables'] = available_vars
                    available_charts.append(spec_copy)

            if not available_charts:
                continue

            # Determine optimal subplot layout
            n_charts = len(available_charts)
            if n_charts <= 4:
                rows, cols = 2, 2
                figsize = (12, 8)
            elif n_charts <= 6:
                rows, cols = 2, 3
                figsize = (15, 8)
            elif n_charts <= 9:
                rows, cols = 3, 3
                figsize = (15, 12)
            else:
                # Split large groups across multiple pages
                charts_per_page = 6
                for page_start in range(0, n_charts, charts_per_page):
                    page_charts = available_charts[page_start:page_start + charts_per_page]
                    _create_plot_page_new(t, results, page_charts, group_name, run_name, pdf, page_start//charts_per_page + 1, config_filename=config_filename)
                continue

            # Create single page for this group
            _create_plot_page_new(t, results, available_charts, group_name, run_name, pdf, layout=(rows, cols), figsize=figsize, config_filename=config_filename)

    return pdf_path


def _create_plot_page_new(t, results, chart_specs, group_name, run_name, pdf, page_num=None, layout=None, figsize=None, config_filename=None):
    """
    Create a single page of plots for a variable group with support for combined charts.

    Parameters
    ----------
    t : array
        Time array
    results : dict
        Results dictionary
    chart_specs : list
        List of chart specifications (single or combined)
    group_name : str
        Name of the variable group
    run_name : str
        Name of the model run to display in header
    pdf : PdfPages
        PDF object to add page to
    page_num : int, optional
        Page number for multi-page groups
    layout : tuple, optional
        (rows, cols) layout. If None, defaults to (2, 3)
    figsize : tuple, optional
        Figure size. If None, defaults to (15, 10)
    config_filename : str, optional
        Name of configuration file to display on page
    """
    if layout is None:
        layout = (2, 3)
    if figsize is None:
        figsize = (15, 10)

    rows, cols = layout
    n_charts = len(chart_specs)

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Create title with run_name
    title = f'{run_name} - COIN_equality Model Results - {group_name.title()} Variables'
    if page_num is not None:
        title += f' (Page {page_num})'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

    # Handle single subplot case
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Define colors for multi-line plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Variables that should use logarithmic y-axis
    log_scale_vars = {'y_net', 'y_damaged', 'K', 'Consumption', 'Savings', 'Y_gross', 'Y_net', 'A', 'redistribution', 'Redistribution_amount'}

    # Plot each chart
    for i, chart_spec in enumerate(chart_specs):
        if i >= len(axes):
            break

        ax = axes[i]
        chart_type = chart_spec['type']
        var_list = chart_spec['variables']

        if chart_type == 'single':
            # Single variable plot
            var_name = var_list[0]
            meta = VARIABLE_METADATA.get(var_name, {})
            description = meta.get('description', var_name)
            units = meta.get('units', '')

            # Plot the time series
            ax.plot(t, results[var_name], linewidth=2, color=colors[0], alpha=0.8)

            # Set labels
            ax.set_xlabel('Time (yr)', fontsize=11)
            if units:
                ylabel = f'{description}\n({units})'
            else:
                ylabel = description
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'{var_name}: {description}', fontsize=12, fontweight='bold', pad=10)

        elif chart_type == 'combined':
            # Combined variables plot
            chart_title = chart_spec.get('title', 'Combined Variables')
            chart_units = chart_spec.get('units', '')

            # Plot each variable with different colors
            for j, var_name in enumerate(var_list):
                meta = VARIABLE_METADATA.get(var_name, {})
                description = meta.get('description', var_name)

                color = colors[j % len(colors)]
                ax.plot(t, results[var_name], linewidth=2, color=color, alpha=0.8,
                       label=description)

            # Set labels
            ax.set_xlabel('Time (yr)', fontsize=11)
            if chart_units:
                ylabel = f'{chart_title}\n({chart_units})'
            else:
                ylabel = chart_title
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(chart_title, fontsize=12, fontweight='bold', pad=10)

            # Add legend
            ax.legend(fontsize=9, loc='best', framealpha=0.8)

        # Apply logarithmic scale if any variable in this chart uses it
        if any(var in log_scale_vars for var in var_list):
            ax.set_yscale('log')

        # Improve grid and formatting
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Use custom scientific notation formatting (skip for log scale - already handles it)
        if not any(var in log_scale_vars for var in var_list):
            all_data = np.concatenate([results[var] for var in var_list])
            max_val = np.max(np.abs(all_data))

            if max_val == 0:
                pass  # No formatting needed for zero data
            elif max_val > 1e4 or max_val < 1e-2:
                ax.yaxis.set_major_formatter(FuncFormatter(format_scientific_notation))

        # Set background color
        ax.set_facecolor('#fafafa')

    # Hide unused subplots
    for i in range(n_charts, len(axes)):
        axes[i].set_visible(False)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Apply zero-bound expansion AFTER layout adjustment (to prevent being overridden)
    for i, chart_spec in enumerate(chart_specs):
        if i >= len(axes):
            break
        ax = axes[i]
        var_list = chart_spec['variables']

        # Skip log scale plots
        if not any(var in log_scale_vars for var in var_list):
            # Check actual data range (not axis limits which may have matplotlib padding)
            all_data = np.concatenate([results[var] for var in var_list])
            data_min = np.min(all_data)
            data_max = np.max(all_data)

            # Only apply if data doesn't cross zero (all positive or all negative)
            if data_min * data_max > 0:  # Same sign
                abs_data_min = abs(data_min)
                abs_data_max = abs(data_max)
                # If the smaller DATA bound is less than half the larger, replace it with zero
                if min(abs_data_min, abs_data_max) < 0.5 * max(abs_data_min, abs_data_max):
                    ymin, ymax = ax.get_ylim()
                    if abs_data_min < abs_data_max:
                        # Data starts closer to zero - set lower bound to zero
                        ax.set_ylim(0, ymax)
                    else:
                        # Data ends closer to zero - set upper bound to zero
                        ax.set_ylim(ymin, 0)

    # Add config filename at bottom of page if provided
    if config_filename:
        fig.text(0.99, 0.01, f'Config: {config_filename}',
                ha='right', va='bottom', fontsize=6, color='gray', style='italic')

    # Save to PDF
    pdf.savefig(fig, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_results(results, run_name, plot_short_horizon=None, output_dir=None, config_filename=None):
    """
    Save model results to CSV and PDF in timestamped directory.

    Parameters
    ----------
    results : dict
        Results dictionary from integrate_model()
    run_name : str
        Name of the model run
    plot_short_horizon : float or None
        If provided, creates a second PDF with results up to this time (years).
        Creates both plots_full.pdf and plots_short.pdf.
        If None, creates single plots.pdf with all results.
    output_dir : str or None
        If provided, uses this directory for output. If None, creates new timestamped directory.
    config_filename : str, optional
        Name of configuration file to display on each PDF page

    Returns
    -------
    dict
        Dictionary with paths:
        - 'output_dir': path to output directory
        - 'csv_file': path to CSV file
        - 'pdf_file': path to full PDF file
        - 'pdf_file_short': path to short-horizon PDF file (if plot_short_horizon provided)

    Notes
    -----
    Creates directory: ./data/output/{run_name}_YYYYMMDD-HHMMSS (if output_dir not provided)
    Writes files:
    - results.csv: all variables in tabular format
    - plots_full.pdf: time series plots for entire integration period
    - plots_short.pdf: time series plots for short horizon (if plot_short_horizon provided)
    """
    if output_dir is None:
        output_dir = create_output_directory(run_name)
    csv_file = write_results_csv(results, output_dir)

    output_dict = {
        'output_dir': output_dir,
        'csv_file': csv_file,
    }

    if plot_short_horizon is not None:
        t = results['t']
        mask = t <= plot_short_horizon

        results_short = {key: val[mask] if isinstance(val, np.ndarray) else val
                        for key, val in results.items()}

        pdf_file_full = plot_results_pdf(results, output_dir, run_name, filename='plots_full.pdf', config_filename=config_filename)
        pdf_file_short = plot_results_pdf(results_short, output_dir, run_name, filename='plots_short.pdf', config_filename=config_filename)

        output_dict['pdf_file'] = pdf_file_full
        output_dict['pdf_file_short'] = pdf_file_short
    else:
        pdf_file = plot_results_pdf(results, output_dir, run_name, filename='plots.pdf', config_filename=config_filename)
        output_dict['pdf_file'] = pdf_file

    return output_dict
