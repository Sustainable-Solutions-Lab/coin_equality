#!/usr/bin/env python3
"""
Generate all 32 variants of a config file by varying 5 boolean switches.

This script creates 32 configuration files covering all combinations of:
1. income_dependent_damage_distribution
2. income_dependent_aggregate_damage
3. income_dependent_tax_policy
4. income_redistribution
5. income_dependent_redistribution_policy

For each variant, the run_name is updated to reflect the switch pattern
(e.g., "ttttt" for all true, "fffff" for all false, etc.)

Usage:
    python generate_switch_variants.py <input_config.json>

Example:
    python generate_switch_variants.py config_COIN-equality_004_tt-f-tt_0.02_fast10+t_1000.json
"""

import sys
import json
import itertools
from pathlib import Path


def generate_pattern_string(switches):
    """
    Convert boolean switch values to pattern string.

    Parameters
    ----------
    switches : tuple of bool
        5-element tuple of boolean values

    Returns
    -------
    str
        Pattern string like "ttttt", "ttftf", etc.
    """
    return ''.join('t' if s else 'f' for s in switches)


def update_config_with_switches(config, switches):
    """
    Update config with new switch values.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    switches : tuple of bool
        5-element tuple: (damage_dist, aggregate_damage, tax_policy, redistribution, redistribution_policy)

    Returns
    -------
    dict
        Updated configuration dictionary
    """
    damage_dist, aggregate_damage, tax_policy, redistribution, redistribution_policy = switches

    config['scalar_parameters']['income_dependent_damage_distribution'] = damage_dist
    config['scalar_parameters']['income_dependent_aggregate_damage'] = aggregate_damage
    config['scalar_parameters']['income_dependent_tax_policy'] = tax_policy
    config['scalar_parameters']['income_redistribution'] = redistribution
    config['scalar_parameters']['income_dependent_redistribution_policy'] = redistribution_policy

    return config


def update_run_name(config, pattern_string):
    """
    Update run_name to reflect switch pattern.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    pattern_string : str
        Pattern string like "ttttt", "ttftf", etc.

    Returns
    -------
    dict
        Updated configuration with new run_name
    """
    old_run_name = config['run_name']

    # Replace the switch pattern in the run_name
    # Pattern in original: "COIN-equality_004_tt-f-tt_0.02_fast10+t_1000"
    # We need to find and replace the "tt-f-tt" part
    parts = old_run_name.split('_')

    # Find the part that looks like a switch pattern (contains 't' or 'f' and hyphens)
    # Start from the end to avoid matching parts that might incidentally have these chars
    pattern_replaced = False
    for i in range(len(parts) - 1, -1, -1):
        part = parts[i]
        # Check if this part looks like a switch pattern
        if any(c in part for c in ['t', 'f']) and '-' in part and len(part.replace('-', '')) >= 3:
            # Replace with new pattern (insert hyphens every 1 character for readability)
            parts[i] = '-'.join(pattern_string)
            pattern_replaced = True
            break

    if not pattern_replaced:
        # If no pattern found with hyphens, look for the position after the version number
        # Assume format: NAME_VERSION_<pattern>_...
        if len(parts) >= 3:
            parts[2] = '-'.join(pattern_string)

    new_run_name = '_'.join(parts)
    config['run_name'] = new_run_name

    return config


def generate_output_filename(input_path, pattern_string):
    """
    Generate output filename based on input and pattern.

    Parameters
    ----------
    input_path : Path
        Input configuration file path
    pattern_string : str
        Pattern string like "ttttt"

    Returns
    -------
    Path
        Output file path
    """
    stem = input_path.stem  # filename without .json

    # Replace the pattern in filename
    # Expected format: config_COIN-equality_004_tt-f-tt_0.02_fast10+t_1000
    parts = stem.split('_')

    # Find and replace the switch pattern part (looking for part with t/f and hyphens)
    # Start from the end to avoid matching 'config' which might have similar structure
    pattern_replaced = False
    for i in range(len(parts) - 1, -1, -1):
        part = parts[i]
        # Check if this part looks like a switch pattern (has t or f and hyphens)
        if any(c in part for c in ['t', 'f']) and '-' in part and len(part.replace('-', '')) >= 3:
            parts[i] = '-'.join(pattern_string)
            pattern_replaced = True
            break

    if not pattern_replaced:
        # If no pattern found, assume format: config_NAME_VERSION_PATTERN_...
        # Insert pattern at position 3
        if len(parts) >= 4:
            parts[3] = '-'.join(pattern_string)

    new_stem = '_'.join(parts)
    return input_path.parent / f"{new_stem}.json"


def generate_all_variants(input_config_path):
    """
    Generate all 32 variant config files.

    Parameters
    ----------
    input_config_path : str or Path
        Path to input configuration file

    Returns
    -------
    list of Path
        List of generated config file paths
    """
    input_path = Path(input_config_path)

    # Read input config
    with open(input_path, 'r') as f:
        base_config = json.load(f)

    # Generate all 32 combinations of 5 boolean switches
    all_combinations = list(itertools.product([False, True], repeat=5))

    generated_files = []

    print(f"Generating {len(all_combinations)} config variants from {input_path.name}\n")

    for switches in all_combinations:
        # Create a deep copy of base config
        config = json.loads(json.dumps(base_config))

        # Generate pattern string
        pattern_string = generate_pattern_string(switches)

        # Update switches
        config = update_config_with_switches(config, switches)

        # Update run_name
        config = update_run_name(config, pattern_string)

        # Generate output filename
        output_path = generate_output_filename(input_path, pattern_string)

        # Write output file
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=4)

        generated_files.append(output_path)
        print(f"  {pattern_string}: {output_path.name}")

    return generated_files


def main():
    """Main entry point."""

    if len(sys.argv) != 2:
        print("Usage: python generate_switch_variants.py <input_config.json>")
        print("\nExample:")
        print("  python generate_switch_variants.py config_COIN-equality_004_tt-f-tt_0.02_fast10+t_1000.json")
        sys.exit(1)

    input_config = sys.argv[1]

    if not Path(input_config).exists():
        print(f"Error: Input file not found: {input_config}")
        sys.exit(1)

    generated_files = generate_all_variants(input_config)

    print(f"\n{'=' * 80}")
    print(f"Successfully generated {len(generated_files)} configuration files!")
    print(f"{'=' * 80}\n")

    print("Next step: Run all configs in parallel")
    print('  python run_parallel.py "config_COIN-equality_004_*-*-*-*-*_0.02_fast10+t_1000.json"')
    print("\nOr run with reduced evaluations for testing:")
    print('  python run_parallel.py "config_COIN-equality_004_*-*-*-*-*_0.02_fast10+t_1000.json" --optimization_parameters.max_evaluations 100')


if __name__ == '__main__':
    main()
