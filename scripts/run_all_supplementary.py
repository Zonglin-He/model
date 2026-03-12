import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXPERIMENTS = [
    "scripts/run_structural_corruption.py",
    "scripts/run_online_streaming_analysis.py",
    "scripts/run_latency_benchmark.py",
    "scripts/run_gate_diagnostics.py",
    "scripts/run_ssaw_physical_validation.py",
    "scripts/run_crossdataset_ablation.py",
    "scripts/run_source_sensitivity.py",
    "scripts/run_significance_test.py",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip", nargs="*", default=[], help="Skip scripts whose names contain these substrings.")
    args = parser.parse_args()

    for script in EXPERIMENTS:
        if any(skip in script for skip in args.skip):
            print(f"[Skip] {script}")
            continue
        print("\n" + "=" * 60)
        print(f"[Running] {script}")
        print("=" * 60)
        ret = subprocess.run(
            [sys.executable, script, "--data_path", args.data_path, "--device", args.device],
            check=False,
        )
        if ret.returncode != 0:
            print(f"[Warning] {script} exited with code {ret.returncode}")

    print("\nSupplementary experiment dispatch finished.")


if __name__ == "__main__":
    main()
