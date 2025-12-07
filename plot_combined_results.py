#!/usr/bin/env python
"""
Combine multiple optimization runs into a single PDF with overlaid plots.

This utility creates a plots_full_combined.pdf file where each panel shows
data from multiple runs overlaid as different colored lines. It searches for
results.csv files in the specified directories and combines them into a
single comparison report.

Usage:
    python plot_combined_results.py "data/output/COIN*tt-f-tt*"
    python plot_combined_results.py "data/output/*/results.csv"
    python plot_combined_results.py data/output/run1 data/output/run2 data/output/run3
    python plot_combined_results.py --output combined.pdf "data/output/COIN*"

Examples:
    # Combine all runs matching a pattern
    python plot_combined_results.py "data/output/COIN-equality_003_tt-*"

    # Combine specific directories
    python plot_combined_results.py data/output/run1 data/output/run2

    # Specify output file
    python plot_combined_results.py --output my_comparison.pdf "data/output/*"
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from glob import glob
from visualization_utils import create_results_report_pdf


def find_results_files(path_patterns):
    """
    Find all results.csv files matching the given path patterns.

    Parameters
    ----------
    path_patterns : list of str
        List of file/directory paths or glob patterns

    Returns
    -------
    list of Path
        List of Path objects pointing to results.csv files
    """
    results_files = []

    for pattern in path_patterns:
        # Expand glob pattern
        expanded_paths = glob(pattern, recursive=True)

        if not expanded_paths:
            # Try as literal path
            expanded_paths = [pattern]

        for path_str in expanded_paths:
            path = Path(path_str)

            if path.is_file() and path.name == 'results.csv':
                results_files.append(path)
            elif path.is_dir():
                # Look for results.csv in this directory
                results_csv = path / 'results.csv'
                if results_csv.exists():
                    results_files.append(results_csv)

    return results_files


def extract_case_name(results_path):
    """
    Extract a meaningful case name from the results.csv file path.

    Uses the parent directory name as the case identifier.

    Parameters
    ----------
    results_path : Path
        Path to results.csv file

    Returns
    -------
    str
        Case name (parent directory name)
    """
    return results_path.parent.name


def load_case_data(results_files):
    """
    Load all results.csv files into a case_data dictionary.

    Parameters
    ----------
    results_files : list of Path
        List of Path objects pointing to results.csv files

    Returns
    -------
    dict
        Dictionary mapping case names to DataFrames
    """
    case_data = {}

    for results_file in results_files:
        case_name = extract_case_name(results_file)

        # Handle duplicate case names by appending a number
        original_name = case_name
        counter = 1
        while case_name in case_data:
            case_name = f"{original_name}_{counter}"
            counter += 1

        # Load CSV
        df = pd.read_csv(results_file)
        case_data[case_name] = df

        print(f"Loaded: {case_name} ({len(df)} timesteps)")

    return case_data


def main():
    parser = argparse.ArgumentParser(
        description='Combine multiple optimization runs into a single comparison PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'paths',
        nargs='+',
        help='Paths to results.csv files, directories containing results.csv, or glob patterns (e.g., "data/output/COIN*")'
    )

    parser.add_argument(
        '--output', '-o',
        default='plots_full_combined.pdf',
        help='Output PDF file name (default: plots_full_combined.pdf)'
    )

    args = parser.parse_args()

    # Find all results.csv files
    print(f"Searching for results.csv files...")
    results_files = find_results_files(args.paths)

    if not results_files:
        print(f"ERROR: No results.csv files found matching patterns: {args.paths}")
        print("\nSearched patterns:")
        for pattern in args.paths:
            print(f"  - {pattern}")
        sys.exit(1)

    print(f"Found {len(results_files)} results.csv file(s)")

    # Load data
    print(f"\nLoading data...")
    case_data = load_case_data(results_files)

    if not case_data:
        print("ERROR: No data loaded successfully")
        sys.exit(1)

    # Create combined PDF
    print(f"\nCreating combined PDF with {len(case_data)} case(s)...")
    output_path = Path(args.output)
    create_results_report_pdf(case_data, output_path)

    print(f"\nSUCCESS: Combined plots saved to {output_path}")
    print(f"Cases included:")
    for case_name in case_data.keys():
        print(f"  - {case_name}")


if __name__ == '__main__':
    main()
