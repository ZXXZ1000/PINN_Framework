"""
Environment readiness check for PINN_Framework.

This script verifies the imports required by three project paths:
training, Fastscape/xsimlab data generation, and tests. Use --smoke to run the
small operator training verification as well.
"""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


REQUIRED_MODULES = [
    ("torch", "operator training"),
    ("numpy", "numerics"),
    ("scipy", "numerics"),
    ("pandas", "data utilities"),
    ("xarray", "Fastscape/xsimlab data generation"),
    ("dask", "Fastscape/xsimlab data generation"),
    ("distributed", "Fastscape/xsimlab data generation"),
    ("xsimlab", "Fastscape/xsimlab data generation"),
    ("fastscape", "Fastscape teacher data generation"),
    ("matplotlib", "visualization"),
    ("PIL", "image utilities"),
    ("yaml", "YAML config"),
    ("omegaconf", "config interpolation"),
    ("tqdm", "training progress"),
    ("tensorboard", "training logs"),
    ("skimage", "image/data analysis"),
    ("netCDF4", "NetCDF outputs"),
    ("pytest", "test runner"),
]


def import_module(name):
    module = importlib.import_module(name)
    return getattr(module, "__version__", "unknown")


def check_required_modules():
    failures = []
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    for module_name, purpose in REQUIRED_MODULES:
        try:
            version = import_module(module_name)
            print(f"[OK] {module_name}: {version} ({purpose})")
        except Exception as exc:
            failures.append((module_name, purpose, exc))
            print(f"[FAIL] {module_name}: {type(exc).__name__}: {exc} ({purpose})")

    try:
        import torch

        print(f"Torch CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.backends, "mps"):
            print(f"Torch MPS available: {torch.backends.mps.is_available()}")
    except Exception:
        pass

    return failures


def run_smoke_training():
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "verify_operator_training.py")]
    print("Running smoke training:", " ".join(command))
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def main():
    parser = argparse.ArgumentParser(description="Check PINN_Framework environment readiness.")
    parser.add_argument("--smoke", action="store_true", help="Run the small operator training verification.")
    args = parser.parse_args()

    failures = check_required_modules()
    if failures:
        print("\nEnvironment check failed. Missing or broken modules:")
        for module_name, purpose, exc in failures:
            print(f"- {module_name}: {purpose}; {type(exc).__name__}: {exc}")
        return 1

    if args.smoke:
        run_smoke_training()

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
