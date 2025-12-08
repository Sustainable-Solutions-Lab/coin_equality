#!/usr/bin/env python
"""
Quick test of lagged damage implementation.
"""

import json
from pathlib import Path
from parameters import load_configuration
from economic_model import integrate_model

# Load a simple configuration
config_path = Path("config_COIN-equality_003_tt-t-tt_0.02_fast10+t_320.json")

print(f"Loading configuration from {config_path}...")
config = load_configuration(str(config_path))

print(f"Running model integration with lagged damage...")
try:
    results = integrate_model(config, store_detailed_output=True)

    print(f"\nSUCCESS! Model integrated successfully.")
    print(f"Final timestep: t = {results['t'][-1]:.2f}")
    print(f"Final utility: U = {results['U'][-1]:.6e}")
    print(f"Final Omega: {results['Omega'][-1]:.6f}")
    print(f"Number of damage iterations per timestep: {results['n_damage_iterations'][0]:.0f} (should be 1 for lagged approach)")

    # Check that all damage iterations are 1
    if all(n == 1 for n in results['n_damage_iterations']):
        print("\n✓ VERIFIED: All timesteps use single-pass lagged calculation (no iteration)")
    else:
        print(f"\n⚠ WARNING: Some timesteps required multiple iterations")
        unique_iterations = set(results['n_damage_iterations'])
        print(f"  Unique iteration counts: {sorted(unique_iterations)}")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
