"""
Visualization utilities for COIN_equality optimization results.

Provides reusable plotting functions for:
- Single-run result visualization (time series plots)
- Multi-run comparison plots
- Summary statistics plots

All plotting functions use a unified interface that works for both single
and multiple cases, ensuring consistent formatting across all reports.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from pathlib import Path


def clean_column_names(df):
    """
    Extract simple variable names from CSV headers with format 'var_name, Description, (units)'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with full column names from results.csv

    Returns
    -------
    pd.DataFrame
        DataFrame with simplified column names (just the variable name)
    """
    new_columns = []
    for col in df.columns:
        # Extract variable name (everything before first comma)
        var_name = col.split(',')[0].strip()
        new_columns.append(var_name)
    df.columns = new_columns
    return df


def plot_timeseries(ax, case_data, variable, ylabel, title, show_legend=False):
    """
    Create time series plot on given axes (unified function for single/multi-run).

    Works for both single-run and multi-run cases with consistent formatting.
    For single case: plots one line without legend.
    For multiple cases: plots multiple lines (legend controlled by show_legend).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    case_data : dict
        Dictionary mapping case names to DataFrames containing results
        Single case: {'Case': df}
        Multiple cases: {'Case1': df1, 'Case2': df2, ...}
    variable : str
        Column name in DataFrame to plot (e.g., 'T_atm', 'y', 'K')
    ylabel : str
        Label for y-axis
    title : str
        Plot title
    show_legend : bool, optional
        If True, show legend on this plot (default: False)

    Notes
    -----
    - Uses tab10 colormap for consistent colors across plots
    - Legend display controlled by show_legend parameter
    - Gracefully handles missing variables (skips if not in DataFrame)
    """
    colors = plt.cm.tab10(np.arange(10))

    for idx, (case_name, df) in enumerate(case_data.items()):
        if variable in df.columns:
            ax.plot(
                df['t'],
                df[variable],
                label=case_name,
                linewidth=2,
                color=colors[idx % 10]
            )

    ax.set_xlabel('Time (years)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f'{variable}: {title}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Apply logarithmic scale for specific variables
    log_scale_vars = {'y_net', 'K', 'Consumption', 'Savings', 'Y_gross', 'Y_net', 'A', 'redistribution', 'Redistribution_amount'}
    if variable in log_scale_vars:
        ax.set_yscale('log')

    # Apply zero-bound expansion for linear scales (must be done AFTER layout/sizing)
    # This will be called later in create_results_report_pdf_to_existing

    if show_legend and len(case_data) > 1:
        ax.legend(loc='best', fontsize=8)


def create_legend_panel(ax, case_data):
    """
    Create a dedicated legend panel showing all case names and colors.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to use for legend
    case_data : dict
        Dictionary mapping case names to DataFrames
    """
    colors = plt.cm.tab10(np.arange(10))

    # Turn off axis decorations
    ax.axis('off')

    # Create dummy lines for legend
    handles = []
    labels = []
    for idx, case_name in enumerate(case_data.keys()):
        line = plt.Line2D([0], [0], color=colors[idx % 10], linewidth=3)
        handles.append(line)
        labels.append(case_name)

    # Create legend in center of panel
    ax.legend(handles, labels, loc='center', fontsize=12, frameon=True,
              title='Cases', title_fontsize=14)
    ax.set_title('Legend', fontsize=11, fontweight='bold', pad=10)


def create_results_report_pdf(case_data, output_pdf):
    """
    Create PDF report with all time series plots (6 per page, 16:9 landscape).

    Generates comprehensive visualization of all key variables from results.csv.
    Uses unified plotting function for consistent formatting across single and
    multi-run cases.

    Parameters
    ----------
    case_data : dict
        Dictionary mapping case names to results DataFrames
        {case_name: results_df, ...}
    output_pdf : Path or str
        Output PDF file path

    Notes
    -----
    Layout:
    - 16:9 aspect ratio optimized for screen viewing
    - 6 plots per page in 2×3 grid
    - ~4-5 pages total for all 25 variables
    - Handles missing variables gracefully (shows "not available" message)
    """
    # Clean column names from 'var, Description, (units)' format to just 'var'
    case_data = {name: clean_column_names(df.copy()) for name, df in case_data.items()}

    # Define all variables to plot (in display order, grouped by category)
    variable_specs = [
        # Economic Variables
        ('y_net', 'Effective income ($/person/yr)', 'Effective Per-Capita Income'),
        ('K', 'Capital stock ($)', 'Capital Stock Over Time'),
        ('Consumption', 'Total consumption ($/yr)', 'Total Consumption'),
        ('Savings', 'Gross investment ($/yr)', 'Gross Investment'),
        ('s', 'Savings rate', 'Savings Rate Over Time'),
        ('Y_gross', 'Gross production ($/yr)', 'Gross Production'),
        ('Y_net', 'Net output ($/yr)', 'Net Output After Damages & Abatement'),

        # Climate Variables
        ('delta_T', 'Temperature change (°C)', 'Temperature Change from Pre-Industrial'),
        ('E', 'CO₂ emissions (tCO2/yr)', 'CO₂ Emissions Rate'),
        ('Ecum', 'Cumulative emissions (tCO2)', 'Cumulative CO₂ Emissions'),

        # Abatement and Damage
        ('f', 'Abatement allocation fraction', 'Abatement Allocation Fraction'),
        ('mu', 'Abatement fraction', 'Emissions Abatement Fraction'),
        ('Lambda', 'Abatement cost fraction', 'Abatement Cost (% of Output)'),
        ('AbateCost', 'Abatement cost ($/yr)', 'Total Abatement Cost'),
        ('Omega', 'Climate damage fraction', 'Climate Damage (% of Output)'),
        ('Climate_damage', 'Climate damage ($/yr)', 'Total Climate Damage'),

        # Inequality and Utility
        ('Gini', 'Gini before redistribution', 'Background Gini Index'),
        ('U', 'Mean utility', 'Mean Utility Per Capita'),
        ('redistribution', 'Per-capita redistribution ($/person/yr)', 'Per-Capita Redistribution'),
        ('Redistribution_amount', 'Total redistribution ($/yr)', 'Total Redistribution'),

        # Abatement Economics
        ('marginal_abatement_cost', 'Marginal abatement cost ($/tCO2)', 'Marginal Abatement Cost'),

        # Exogenous Drivers
        ('A', 'Total factor productivity ($)', 'Total Factor Productivity'),
        ('L', 'Population (persons)', 'Population'),
        ('sigma', 'Carbon intensity (tCO2/$)', 'Carbon Intensity of GDP'),
        ('theta1', 'Marginal cost at mu=1 ($/tCO2)', 'Abatement Cost Parameter (theta1)'),
    ]

    # Determine plots per page and starting index
    is_multi_case = len(case_data) > 1
    plots_per_page = 5 if is_multi_case else 6
    plot_start_idx = 1 if is_multi_case else 0

    with PdfPages(output_pdf) as pdf:
        # Process plots in groups
        for page_start in range(0, len(variable_specs), plots_per_page):
            page_vars = variable_specs[page_start:page_start + plots_per_page]

            # Create 2×3 subplot grid in 16:9 landscape orientation
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            fig.suptitle('Results Comparison', fontsize=14, fontweight='bold')
            axes = axes.flatten()

            # For multi-case, use first position for legend
            if is_multi_case:
                create_legend_panel(axes[0], case_data)
                axes_offset = 1
            else:
                axes_offset = 0

            for idx, (var_name, ylabel, title) in enumerate(page_vars):
                ax_idx = idx + axes_offset
                # Check if variable exists in at least one dataset
                if any(var_name in df.columns for df in case_data.values()):
                    plot_timeseries(axes[ax_idx], case_data, var_name, ylabel, title)
                else:
                    # Variable not found - show message
                    axes[ax_idx].text(0.5, 0.5, f'{var_name}\nnot available',
                                     ha='center', va='center',
                                     transform=axes[ax_idx].transAxes,
                                     fontsize=11, color='gray')
                    axes[ax_idx].set_title(title, fontsize=11)

            # Hide any unused subplots on last page
            for idx in range(len(page_vars) + axes_offset, 6):
                axes[idx].axis('off')

            plt.tight_layout()
            pdf.savefig(fig, orientation='landscape')
            plt.close(fig)

    print(f"Results report saved to: {output_pdf}")


def create_objective_scatter_on_axes(ax, optimization_data):
    """
    Create scatter plot comparing objective improvement across cases and iterations.

    Shows improvement relative to first iteration (iteration N - iteration 1).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    optimization_data : dict
        {case_name: optimization_summary_df, ...}

    Notes
    -----
    - Different markers for different iterations (circle, square, triangle, etc.)
    - Cases positioned along x-axis
    - Objective improvement (relative to iteration 1) on y-axis
    - Skips iteration 1 (baseline)
    """
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    colors = plt.cm.tab10(np.arange(10))

    case_names = list(optimization_data.keys())
    max_iterations = max(len(df) for df in optimization_data.values())

    # Plot each iteration (starting from iteration 2) relative to iteration 1
    for iter_idx in range(1, max_iterations):  # Skip iteration 0 (index 0), start from iteration 1 (index 1)
        marker = markers[min(iter_idx - 1, len(markers) - 1)]

        x_positions = []
        y_values = []

        for case_idx, (case_name, df) in enumerate(optimization_data.items()):
            if iter_idx < len(df) and len(df) > 0:
                # Calculate improvement: current iteration - first iteration
                baseline_obj = df.iloc[0]['objective']
                current_obj = df.iloc[iter_idx]['objective']
                improvement = current_obj - baseline_obj

                x_positions.append(case_idx)
                y_values.append(improvement)

        if x_positions:
            ax.scatter(
                x_positions,
                y_values,
                marker=marker,
                s=100,
                label=f'Iter {iter_idx + 1} - Iter 1',
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )

    ax.set_xlabel('Case', fontsize=10)
    ax.set_ylabel('Objective Improvement (vs. Iter 1)', fontsize=10)
    ax.set_title('Objective Improvement by Iteration', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(case_names)))
    ax.set_xticklabels(case_names, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)


def create_elapsed_time_scatter_on_axes(ax, optimization_data):
    """
    Create scatter plot comparing elapsed time across cases and iterations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    optimization_data : dict
        {case_name: optimization_summary_df, ...}
    """
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    case_names = list(optimization_data.keys())
    max_iterations = max(len(df) for df in optimization_data.values())

    for iter_idx in range(max_iterations):
        marker = markers[min(iter_idx, len(markers) - 1)]

        x_positions = []
        y_values = []

        for case_idx, (case_name, df) in enumerate(optimization_data.items()):
            if iter_idx < len(df) and 'elapsed_time' in df.columns:
                x_positions.append(case_idx)
                y_values.append(df.iloc[iter_idx]['elapsed_time'])

        if x_positions:
            ax.scatter(
                x_positions,
                y_values,
                marker=marker,
                s=100,
                label=f'Iteration {iter_idx + 1}',
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )

    ax.set_xlabel('Case', fontsize=10)
    ax.set_ylabel('Elapsed Time (seconds)', fontsize=10)
    ax.set_title('Computation Time Comparison', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(case_names)))
    ax.set_xticklabels(case_names, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')


def create_evaluations_scatter_on_axes(ax, optimization_data):
    """
    Create scatter plot comparing function evaluations across cases and iterations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    optimization_data : dict
        {case_name: optimization_summary_df, ...}
    """
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    case_names = list(optimization_data.keys())
    max_iterations = max(len(df) for df in optimization_data.values())

    for iter_idx in range(max_iterations):
        marker = markers[min(iter_idx, len(markers) - 1)]

        x_positions = []
        y_values = []

        for case_idx, (case_name, df) in enumerate(optimization_data.items()):
            if iter_idx < len(df) and 'n_evaluations' in df.columns:
                x_positions.append(case_idx)
                y_values.append(df.iloc[iter_idx]['n_evaluations'])

        if x_positions:
            ax.scatter(
                x_positions,
                y_values,
                marker=marker,
                s=100,
                label=f'Iteration {iter_idx + 1}',
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )

    ax.set_xlabel('Case', fontsize=10)
    ax.set_ylabel('Function Evaluations', fontsize=10)
    ax.set_title('Evaluations Comparison', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(case_names)))
    ax.set_xticklabels(case_names, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')


def create_comparison_report_pdf(optimization_data, results_data, output_pdf):
    """
    Create comprehensive PDF report comparing multiple optimization runs.

    Combines optimization summary plots and detailed results time series.
    Uses 6-plots-per-page 16:9 landscape layout for screen viewing.

    Parameters
    ----------
    optimization_data : dict
        {case_name: optimization_summary_df, ...}
    results_data : dict
        {case_name: results_df, ...}
    output_pdf : Path or str
        Output PDF file path
    """
    with PdfPages(output_pdf) as pdf:
        # Page 1: Summary plots (2×3 grid, 16:9 landscape)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle('Optimization Summary Comparison', fontsize=14, fontweight='bold')
        axes = axes.flatten()

        # Plot 1: Objective comparison scatter
        create_objective_scatter_on_axes(axes[0], optimization_data)

        # Plot 2: Elapsed time comparison (if available)
        has_elapsed_time = any('elapsed_time' in df.columns for df in optimization_data.values())
        if has_elapsed_time:
            create_elapsed_time_scatter_on_axes(axes[1], optimization_data)
        else:
            axes[1].axis('off')

        # Plot 3: Evaluations comparison
        create_evaluations_scatter_on_axes(axes[2], optimization_data)

        # Plots 4-6: Reserved for future use or hide
        for idx in range(3, 6):
            axes[idx].axis('off')

        plt.tight_layout()
        pdf.savefig(fig, orientation='landscape')
        plt.close(fig)

        # Pages 2+: Time series comparisons using unified plotting function
        if results_data:
            # Temporarily save to pdf object by passing it
            create_results_report_pdf_to_existing(results_data, pdf)

    print(f"Comparison PDF report saved to: {output_pdf}")


def create_results_report_pdf_to_existing(case_data, pdf):
    """
    Add results plots to existing PDF object.

    Parameters
    ----------
    case_data : dict
        {case_name: results_df, ...}
    pdf : PdfPages
        Existing PDF pages object to add to
    """
    # Clean column names from 'var, Description, (units)' format to just 'var'
    case_data = {name: clean_column_names(df.copy()) for name, df in case_data.items()}

    variable_specs = [
        # Economic Variables
        ('y_net', 'Effective income ($/person/yr)', 'Effective Per-Capita Income'),
        ('K', 'Capital stock ($)', 'Capital Stock Over Time'),
        ('Consumption', 'Total consumption ($/yr)', 'Total Consumption'),
        ('Savings', 'Gross investment ($/yr)', 'Gross Investment'),
        ('s', 'Savings rate', 'Savings Rate Over Time'),
        ('Y_gross', 'Gross production ($/yr)', 'Gross Production'),
        ('Y_net', 'Net output ($/yr)', 'Net Output After Damages & Abatement'),

        # Climate Variables
        ('delta_T', 'Temperature change (°C)', 'Temperature Change from Pre-Industrial'),
        ('E', 'CO₂ emissions (tCO2/yr)', 'CO₂ Emissions Rate'),
        ('Ecum', 'Cumulative emissions (tCO2)', 'Cumulative CO₂ Emissions'),

        # Abatement and Damage
        ('f', 'Abatement allocation fraction', 'Abatement Allocation Fraction'),
        ('mu', 'Abatement fraction', 'Emissions Abatement Fraction'),
        ('Lambda', 'Abatement cost fraction', 'Abatement Cost (% of Output)'),
        ('AbateCost', 'Abatement cost ($/yr)', 'Total Abatement Cost'),
        ('Omega', 'Climate damage fraction', 'Climate Damage (% of Output)'),
        ('Climate_damage', 'Climate damage ($/yr)', 'Total Climate Damage'),

        # Inequality and Utility
        ('Gini', 'Gini before redistribution', 'Background Gini Index'),
        ('U', 'Mean utility', 'Mean Utility Per Capita'),
        ('redistribution', 'Per-capita redistribution ($/person/yr)', 'Per-Capita Redistribution'),
        ('Redistribution_amount', 'Total redistribution ($/yr)', 'Total Redistribution'),

        # Abatement Economics
        ('marginal_abatement_cost', 'Marginal abatement cost ($/tCO2)', 'Marginal Abatement Cost'),

        # Exogenous Drivers
        ('A', 'Total factor productivity ($)', 'Total Factor Productivity'),
        ('L', 'Population (persons)', 'Population'),
        ('sigma', 'Carbon intensity (tCO2/$)', 'Carbon Intensity of GDP'),
        ('theta1', 'Marginal cost at mu=1 ($/tCO2)', 'Abatement Cost Parameter (theta1)'),
    ]

    # Determine plots per page and starting index
    is_multi_case = len(case_data) > 1
    plots_per_page = 5 if is_multi_case else 6

    # Process plots in groups
    for page_start in range(0, len(variable_specs), plots_per_page):
        page_vars = variable_specs[page_start:page_start + plots_per_page]

        # Create 2×3 subplot grid in 16:9 landscape orientation
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle('Results Comparison', fontsize=14, fontweight='bold')
        axes = axes.flatten()

        # For multi-case, use first position for legend
        if is_multi_case:
            create_legend_panel(axes[0], case_data)
            axes_offset = 1
        else:
            axes_offset = 0

        for idx, (var_name, ylabel, title) in enumerate(page_vars):
            ax_idx = idx + axes_offset
            # Check if variable exists in at least one dataset
            if any(var_name in df.columns for df in case_data.values()):
                plot_timeseries(axes[ax_idx], case_data, var_name, ylabel, title)
            else:
                # Variable not found - show message
                axes[ax_idx].text(0.5, 0.5, f'{var_name}\nnot available',
                                 ha='center', va='center',
                                 transform=axes[ax_idx].transAxes,
                                 fontsize=11, color='gray')
                axes[ax_idx].set_title(title, fontsize=11)

        # Hide any unused subplots on last page
        for idx in range(len(page_vars) + axes_offset, 6):
            axes[idx].axis('off')

        plt.tight_layout()

        # Apply zero-bound expansion AFTER layout (to prevent being overridden)
        log_scale_vars = {'y_net', 'K', 'Consumption', 'Savings', 'Y_gross', 'Y_net', 'A', 'redistribution', 'Redistribution_amount'}
        for idx, (var_name, ylabel, title) in enumerate(page_vars):
            ax_idx = idx + axes_offset
            ax = axes[ax_idx]

            # Skip if variable not available or uses log scale
            if not any(var_name in df.columns for df in case_data.values()):
                continue
            if var_name in log_scale_vars:
                continue

            # Check actual data range (not axis limits which may have matplotlib padding)
            all_data = []
            for df in case_data.values():
                if var_name in df.columns:
                    all_data.extend(df[var_name].values)

            if not all_data:
                continue

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

        pdf.savefig(fig, orientation='landscape')
        plt.close(fig)
