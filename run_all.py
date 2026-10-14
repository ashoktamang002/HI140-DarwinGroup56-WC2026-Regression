"""Run the full pipeline end to end."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
for step in ["parse_fbref.py", "build_datasets.py", "descriptives.py", "modelling.py", "assumptions.py", "class_workflow.py", "figures.py"]:
    print(f"\n>>> {step}")
    subprocess.run([sys.executable, step], cwd=SRC, check=True)
