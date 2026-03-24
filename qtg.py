"""Cross-platform CLI for quant-trading-gym Python tools.

Usage (from quant-trading-gym/):
    python qtg.py check                          # data quality
    python qtg.py train                          # train all models
    python qtg.py shap                           # feature analysis
    python qtg.py shap --output shap_results.json
"""
import importlib
import os
import sys

# Add python/ to import path so `training.*` resolves from project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))

COMMANDS = {
    "check": "training.check_parquet",
    "train": "training.train_models",
    "shap": "training.analyze_shap",
}

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print("Usage: python qtg.py <command> [args...]")
    print()
    for cmd, mod in COMMANDS.items():
        print(f"  {cmd:10} {mod}")
    print()
    print("Pass --help after command for command-specific options.")
    sys.exit(0)

command = sys.argv[1]
if command not in COMMANDS:
    print(f"Unknown command: {command}")
    print(f"Available: {', '.join(COMMANDS)}")
    sys.exit(1)

# Remove the command name so argparse in each module sees correct argv
sys.argv = [sys.argv[0]] + sys.argv[2:]

mod = importlib.import_module(COMMANDS[command])
sys.exit(mod.main() or 0)
