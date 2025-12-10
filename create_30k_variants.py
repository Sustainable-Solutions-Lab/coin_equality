#!/usr/bin/env python3
"""
Create 30k evaluation variants from 1000 evaluation configs.

Reads JSON files from ./json folder and creates copies in current directory with:
- Filename: _1000.json -> _30k.json
- run_name: _1000 -> _30k
- max_evaluations: 1000 -> 30000
"""

import json
import glob
from pathlib import Path


def create_30k_variant(input_path):
    """
    Create 30k evaluation variant from 1000 evaluation config.

    Parameters
    ----------
    input_path : Path
        Path to input JSON file

    Returns
    -------
    Path
        Path to created output file
    """
    with open(input_path, 'r') as f:
        config = json.load(f)

    # Update run_name: replace _1000 with _30k
    old_run_name = config['run_name']
    new_run_name = old_run_name.replace('_1000', '_30k')
    config['run_name'] = new_run_name

    # Update max_evaluations: 1000 -> 30000
    config['optimization_parameters']['max_evaluations'] = 30000

    # Generate output filename: replace _1000.json with _30k.json
    output_filename = input_path.name.replace('_1000.json', '_30k.json')
    output_path = Path.cwd() / output_filename

    # Write output file
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

    return output_path


def main():
    """Main entry point."""
    # Find all 1000-evaluation config files in ./json
    pattern = './json/config_COIN-equality_004_*-*-*-*-*_0.02_fast10+t_1000.json'
    input_files = sorted(glob.glob(pattern))

    if not input_files:
        print(f"No files found matching pattern: {pattern}")
        return

    print(f"Creating 30k evaluation variants from {len(input_files)} config files\n")

    created_files = []
    for input_path in input_files:
        input_path = Path(input_path)
        output_path = create_30k_variant(input_path)
        created_files.append(output_path)
        print(f"  {input_path.name} -> {output_path.name}")

    print(f"\n{'=' * 80}")
    print(f"Successfully created {len(created_files)} config files!")
    print(f"{'=' * 80}\n")

    print("Next step: Run all configs in parallel")
    print('  python run_parallel.py "config_COIN-equality_004_*-*-*-*-*_0.02_fast10+t_30k.json"')


if __name__ == '__main__':
    main()
