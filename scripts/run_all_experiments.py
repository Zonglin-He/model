import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.accup import NuSTAR_ActiveSearch


RESULTS_ROOT = ROOT / "results" / "tta_experiments_logs"

EXPERIMENTS = {
    1: {
        "name": "Controlled Structural Corruption",
        "script": ROOT / "scripts" / "run_structural_corruption.py",
        "output_dir": RESULTS_ROOT / "structural_corruption",
    },
    2: {
        "name": "Online Streaming Analysis",
        "script": ROOT / "scripts" / "run_online_streaming_analysis.py",
        "output_dir": RESULTS_ROOT / "online_streaming",
    },
    3: {
        "name": "Runtime / Latency / Memory",
        "script": ROOT / "scripts" / "run_latency_benchmark.py",
        "output_dir": RESULTS_ROOT / "latency",
    },
    4: {
        "name": "Gate Diagnostics",
        "script": ROOT / "scripts" / "run_gate_diagnostics.py",
        "output_dir": RESULTS_ROOT / "gate_diagnostics",
    },
    5: {
        "name": "SSAW Physical Validation",
        "script": ROOT / "scripts" / "run_ssaw_physical_validation.py",
        "output_dir": RESULTS_ROOT / "ssaw_validation",
    },
    6: {
        "name": "Cross-dataset Ablation",
        "script": ROOT / "scripts" / "run_crossdataset_ablation.py",
        "output_dir": RESULTS_ROOT / "crossdataset_ablation",
    },
    7: {
        "name": "Reserved",
        "script": None,
        "output_dir": None,
    },
    8: {
        "name": "Source Model Sensitivity",
        "script": ROOT / "scripts" / "run_source_sensitivity.py",
        "output_dir": RESULTS_ROOT / "source_sensitivity",
    },
    9: {
        "name": "Significance Test",
        "script": ROOT / "scripts" / "run_significance_test.py",
        "output_dir": RESULTS_ROOT / "significance_tests",
    },
}


def parse_int_list(values):
    if not values:
        return []
    return [int(v) for v in values]


def tail_text(text, limit=4000):
    text = text or ""
    return text[-limit:]


def verify_ssaw_fix():
    output_dir = RESULTS_ROOT / "ssaw_fix"
    output_dir.mkdir(parents=True, exist_ok=True)

    search = NuSTAR_ActiveSearch(num_control_points=10, num_candidates=4, sigma=0.1)
    k_test = torch.ones(4, 10, dtype=torch.float32)
    k_test[0, 4] = 1.2
    upsampled = search._natural_cubic_spline_upsample(k_test, 3000)
    max_adjacent_diff = float((upsampled[:, 1:] - upsampled[:, :-1]).abs().max().item())
    passed = bool(max_adjacent_diff < 0.01)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "max_adjacent_diff": max_adjacent_diff,
        "shape": list(upsampled.shape),
    }
    verify_path = output_dir / "verify_result.json"
    verify_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not passed:
        raise RuntimeError(f"SSAW verification failed: max adjacent diff {max_adjacent_diff:.6f} >= 0.01")
    return verify_path, payload


def summarize_experiment_outputs(exp_num):
    spec = EXPERIMENTS[exp_num]
    out_dir = spec["output_dir"]
    if out_dir is None or not out_dir.exists():
        return {}

    summary = {"output_dir": str(out_dir)}
    if exp_num == 1:
        raw_path = out_dir / "raw_results.csv"
        if raw_path.exists():
            import pandas as pd

            df = pd.read_csv(raw_path)
            means = df.groupby(["dataset", "severity"])["f1"].mean().reset_index()
            mild = means[means["severity"] == "mild"].set_index("dataset")["f1"]
            severe = means[means["severity"] == "severe"].set_index("dataset")["f1"]
            common = mild.index.intersection(severe.index)
            drops = {dataset: float(mild[dataset] - severe[dataset]) for dataset in common}
            summary["mean_f1_drop_mild_to_severe"] = drops
    elif exp_num == 3:
        latency_path = out_dir / "latency_results.csv"
        if latency_path.exists():
            import pandas as pd

            df = pd.read_csv(latency_path)
            summary["latency_mean_ms"] = {
                f"{row.method}_{row.dataset}": float(row.latency_mean_ms) for row in df.itertuples()
            }
    elif exp_num == 9:
        sig_path = out_dir / "significance_test_results.csv"
        if sig_path.exists():
            import pandas as pd

            df = pd.read_csv(sig_path)
            summary["significant_pairs"] = int((df["significance"].fillna("") != "").sum())
            summary["tests"] = df[["dataset", "scenario", "method_b", "significance"]].to_dict("records")
    return summary


def write_final_summary(summary_payload):
    md_path = RESULTS_ROOT / "FINAL_SUMMARY.md"
    lines = [
        "# Final Summary",
        "",
        f"- Timestamp: {summary_payload['timestamp']}",
        f"- Total elapsed seconds: {summary_payload['total_elapsed_sec']:.2f}",
        "",
        "## SSAW Fix",
        f"- Status: {summary_payload['ssaw_fix']['status']}",
        f"- Verify path: {summary_payload['ssaw_fix'].get('verify_path', '')}",
        f"- Max adjacent diff: {summary_payload['ssaw_fix'].get('details', {}).get('max_adjacent_diff', 'n/a')}",
        "",
        "## Experiment Status",
    ]
    for item in summary_payload["experiments"]:
        lines.append(
            f"- Exp {item['id']}: {item['name']} | {item['status']} | {item['elapsed_sec']:.2f}s | {item.get('output_dir', '')}"
        )
        if item.get("error_summary"):
            lines.append(f"  - Error: {item['error_summary']}")
    lines.append("")
    lines.append("## Key Outputs")
    for item in summary_payload["experiments"]:
        if item.get("highlights"):
            lines.append(f"- Exp {item['id']} highlights: {json.dumps(item['highlights'], ensure_ascii=False)}")
    lines.append("")
    lines.append("## Output Paths")
    for item in summary_payload["experiments"]:
        if item.get("output_dir"):
            lines.append(f"- Exp {item['id']}: {item['output_dir']}")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    requested_only = set(parse_int_list(args.only))
    requested_skip = set(parse_int_list(args.skip))

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable, falling back to cpu")
        device = "cpu"

    overall_start = time.perf_counter()
    summary_payload = {
        "timestamp": datetime.now().isoformat(),
        "data_path": args.data_path,
        "device": device,
        "seeds": args.seeds,
        "backbone": args.backbone,
        "ssaw_fix": {},
        "experiments": [],
    }

    print("=" * 80)
    print("[Part A] SSAW fix verification")
    print("=" * 80)
    try:
        verify_path, verify_payload = verify_ssaw_fix()
        summary_payload["ssaw_fix"] = {
            "status": "SUCCESS",
            "verify_path": str(verify_path),
            "details": verify_payload,
        }
        print(f"[SUCCESS] SSAW verification passed in {verify_path}")
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        summary_payload["ssaw_fix"] = {
            "status": "FAILED",
            "verify_path": "",
            "details": {"traceback": tb},
        }

    for exp_num in sorted(EXPERIMENTS):
        spec = EXPERIMENTS[exp_num]
        if requested_only and exp_num not in requested_only:
            summary_payload["experiments"].append(
                {
                    "id": exp_num,
                    "name": spec["name"],
                    "status": "SKIPPED",
                    "elapsed_sec": 0.0,
                    "output_dir": str(spec["output_dir"]) if spec["output_dir"] else "",
                }
            )
            continue
        if exp_num in requested_skip:
            summary_payload["experiments"].append(
                {
                    "id": exp_num,
                    "name": spec["name"],
                    "status": "SKIPPED",
                    "elapsed_sec": 0.0,
                    "output_dir": str(spec["output_dir"]) if spec["output_dir"] else "",
                }
            )
            continue
        if spec["script"] is None:
            summary_payload["experiments"].append(
                {
                    "id": exp_num,
                    "name": spec["name"],
                    "status": "SKIPPED",
                    "elapsed_sec": 0.0,
                    "output_dir": "",
                    "error_summary": "Experiment slot not defined in prompt.",
                }
            )
            continue

        print("\n" + "=" * 80)
        print(f"[Experiment {exp_num}] {spec['name']}")
        print("=" * 80)
        start = time.perf_counter()
        cmd = [
            sys.executable,
            str(spec["script"]),
            "--data_path",
            args.data_path,
            "--device",
            device,
            "--seeds",
            args.seeds,
            "--backbone",
            args.backbone,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            elapsed = time.perf_counter() - start
            print(proc.stdout)
            if proc.stderr:
                print(proc.stderr)
            status = "SUCCESS" if proc.returncode == 0 else "FAILED"
            error_summary = tail_text(proc.stderr) if proc.returncode != 0 else ""
            item = {
                "id": exp_num,
                "name": spec["name"],
                "status": status,
                "elapsed_sec": elapsed,
                "output_dir": str(spec["output_dir"]),
                "error_summary": error_summary,
                "stdout_tail": tail_text(proc.stdout),
                "stderr_tail": tail_text(proc.stderr),
            }
            item["highlights"] = summarize_experiment_outputs(exp_num)
            summary_payload["experiments"].append(item)
            print(f"[{status}] elapsed={elapsed:.2f}s")
        except Exception:
            elapsed = time.perf_counter() - start
            tb = traceback.format_exc()
            print(tb)
            summary_payload["experiments"].append(
                {
                    "id": exp_num,
                    "name": spec["name"],
                    "status": "FAILED",
                    "elapsed_sec": elapsed,
                    "output_dir": str(spec["output_dir"]),
                    "error_summary": tail_text(tb),
                    "highlights": {},
                }
            )

    summary_payload["total_elapsed_sec"] = time.perf_counter() - overall_start
    summary_json_path = RESULTS_ROOT / "experiment_summary.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = write_final_summary(summary_payload)

    print("\n" + "=" * 80)
    print("All experiments finished.")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Final summary: {md_path}")


if __name__ == "__main__":
    main()
