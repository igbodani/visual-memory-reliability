"""Lightweight setup checks. No model imports/downloads except in explicit verify mode.

This helper is outside vmresearch so the frozen experiment-code digest is unchanged.
Expected core versions come from the supplied successful-run record, not a new lockfile.
"""
from __future__ import annotations
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys

REFERENCE = Path(__file__).resolve().parent / "reference" / "compatibility_pass.json"

def expected_versions() -> dict[str, str]:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))["packages"]

def current_versions(names) -> dict[str, str | None]:
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result

def compare_versions(expected, current) -> dict:
    return {k: {"expected": v, "installed": current.get(k)}
            for k, v in expected.items() if current.get(k) != v}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("core", "gpu", "installed", "snapshot", "verify"))
    args = parser.parse_args()
    expected = expected_versions()
    actual = current_versions(expected)
    if args.mode == "snapshot":
        print(json.dumps({"executable": sys.executable, "python": sys.version,
                          "packages": actual}, indent=2))
        return 0
    if args.mode == "installed":
        return 0 if all(actual.get(k) is not None for k in ("stretch-ai", "sam-2")) else 1
    if args.mode == "gpu":
        expected = {k: expected[k] for k in ("torch", "torchvision")}
    differences = compare_versions(expected, actual)
    if differences or sys.version_info[:2] != (3, 10):
        print(json.dumps({"version_mismatches": differences, "python": sys.version}, indent=2),
              file=sys.stderr)
        return 1
    if args.mode != "verify":
        return 0
    import torch
    import torchvision
    if not torch.cuda.is_available():
        raise RuntimeError("Core versions match, but CUDA is unavailable. Select a GPU runtime; do not reinstall packages.")
    values = (torch.ones(3, device="cuda") * 2).cpu().tolist()
    if values != [2.0, 2.0, 2.0]:
        raise RuntimeError(f"Unexpected GPU calculation: {values}")
    print("Python executable:", sys.executable)
    print("Python:", sys.version.split()[0])
    for name, value in actual.items():
        print(f"{name}: {value}")
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU calculation:", values)
    print("CORE VERSIONS AND GPU CHECK PASSED")
    print("Known dependency-metadata exceptions remain. Run Section 3's full preflight next.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
