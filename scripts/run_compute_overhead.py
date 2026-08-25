"""Same-hardware latency, throughput, memory, and profiler-FLOP audit."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.supplementary_utils import (
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    enforce_common_batch_size,
    ensure_dir,
    move_data_to_device,
)


SCENARIOS = {"EEG": ("16", "1"), "HAR": ("12", "16"), "FD": ("2", "3")}


def parse_list(text, cast=str):
    return [cast(value.strip()) for value in str(text).split(",") if value.strip()]


def synchronize(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def invoke(model, data, source_only):
    if source_only:
        with torch.inference_mode():
            return model({"data": data})
    return model({"data": data})


def profile_flops(model, data, source_only, device):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(device).startswith("cuda") and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as profiler:
            invoke(model, data, source_only)
            synchronize(device)
        flops = sum(float(event.flops or 0.0) for event in profiler.key_averages())
        return flops
    except Exception:
        return float("nan")


def measure(args, dataset, method, override=None):
    src_id, trg_id = SCENARIOS[dataset]
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=dataset,
        da_method=method,
        backbone=args.backbone,
        exp_name=f"compute_{method}",
        seed=args.stream_seed,
        source_seed=args.source_seed,
        pretrain_cache_dir=args.pretrain_cache_dir,
    )
    common_batch_size = enforce_common_batch_size(trainer, src_id, trg_id)
    if override:
        trainer.set_runtime_hparams(override)
    tta_model = pre_trained_model = None
    try:
        tta_model, pre_trained_model = create_tta_model(
            trainer, src_id, trg_id, run_seed=args.stream_seed
        )
        target_loader = trainer.trg_whole_dl
        source_only = method == "NoAdap"
        total_parameters = sum(parameter.numel() for parameter in tta_model.parameters())
        requires_grad_parameters = sum(
            parameter.numel()
            for parameter in tta_model.parameters()
            if parameter.requires_grad
        )
        if source_only:
            optimizer_parameters = 0
        else:
            optimizer_parameter_ids = {
                id(parameter)
                for group in tta_model.optimizer.param_groups
                for parameter in group["params"]
            }
            optimizer_parameters = sum(
                parameter.numel()
                for parameter in tta_model.parameters()
                if id(parameter) in optimizer_parameter_ids
            )
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(trainer.device)

        # First pass: a true online stream starting from the untouched source
        # checkpoint. Retain predictions so runtime and quality are reported
        # from the same online pass.
        stream_start = time.perf_counter()
        stream_samples = 0
        stream_logits = []
        stream_labels = []
        for data, labels, _ in target_loader:
            data = move_data_to_device(data, trainer.device)
            stream_samples += data[0].size(0) if isinstance(data, (list, tuple)) else data.size(0)
            outputs = invoke(tta_model, data, source_only)
            stream_logits.append(outputs.detach().cpu())
            stream_labels.append(labels.view(-1).long().cpu())
        synchronize(args.device)
        total_stream_seconds = time.perf_counter() - stream_start
        all_logits = torch.cat(stream_logits)
        all_labels = torch.cat(stream_labels)
        all_predictions = all_logits.argmax(dim=1)
        stream_accuracy = float((all_predictions == all_labels).float().mean().item())
        stream_macro_f1 = float(
            f1_score(
                all_labels.numpy(),
                all_predictions.numpy(),
                average="macro",
                zero_division=0,
            )
        )

        # Second pass: steady-state per-batch latency. Iterate the loader
        # directly instead of materialising the complete three-view stream.
        timings = []
        samples = []
        total_iterations = args.warmup_batches + args.measure_batches
        timing_iterator = iter(target_loader)
        for iteration in range(total_iterations):
            try:
                data, _, _ = next(timing_iterator)
            except StopIteration:
                timing_iterator = iter(target_loader)
                data, _, _ = next(timing_iterator)
            data = move_data_to_device(data, trainer.device)
            batch_size = data[0].size(0) if isinstance(data, (list, tuple)) else data.size(0)
            synchronize(args.device)
            start = time.perf_counter()
            invoke(tta_model, data, source_only)
            synchronize(args.device)
            elapsed = time.perf_counter() - start
            if iteration >= args.warmup_batches:
                timings.append(elapsed)
                samples.append(batch_size)

        # Record the adaptation peak before running the profiler.  The profiler
        # performs an additional forward/update and may allocate its own buffers,
        # which would otherwise inflate the reported runtime memory footprint.
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(trainer.device) / (1024 ** 2)
            if torch.cuda.is_available() and str(args.device).startswith("cuda")
            else float("nan")
        )

        profile_data = move_data_to_device(next(iter(target_loader))[0], trainer.device)
        profiler_flops = profile_flops(tta_model, profile_data, source_only, args.device)
        total_timed_samples = sum(samples)
        total_timed_seconds = sum(timings)
        return {
            "dataset": dataset,
            "scenario": f"{src_id}->{trg_id}",
            "method": method,
            "common_batch_size": int(common_batch_size),
            "batch_size": int(round(statistics.mean(samples))),
            "latency_mean_ms": statistics.mean(timings) * 1000.0,
            "latency_std_ms": statistics.pstdev(timings) * 1000.0,
            "throughput_samples_s": total_timed_samples / total_timed_seconds,
            "full_stream_seconds": total_stream_seconds,
            "full_stream_samples_s": stream_samples / total_stream_seconds,
            "full_stream_accuracy": stream_accuracy,
            "full_stream_macro_f1": stream_macro_f1,
            "peak_memory_mb": peak_memory_mb,
            "total_parameters": total_parameters,
            # Reviewer-facing definition: parameters that can actually receive
            # an optimizer update.  Source-only inference therefore reports 0,
            # even if its tensors retain requires_grad=True internally.
            "trainable_parameters": optimizer_parameters,
            "optimizer_parameters": optimizer_parameters,
            "requires_grad_parameters": requires_grad_parameters,
            "profiler_flops_per_batch": profiler_flops,
            "profiler_macs_per_batch_approx": profiler_flops / 2.0,
            **(override or {}),
        }
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--methods",
        default="NoAdap,DuSafe",
    )
    parser.add_argument("--datasets", default="EEG,HAR,FD")
    parser.add_argument("--source_seed", type=int, default=1)
    parser.add_argument("--stream_seed", type=int, default=42)
    parser.add_argument("--warmup_batches", type=int, default=5)
    parser.add_argument("--measure_batches", type=int, default=20)
    parser.add_argument(
        "--pretrain_cache_dir",
        default=str(ROOT / "results" / "pretrain_cache" / "reviewer_rerun"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(ROOT / "results" / "tta_experiments_logs" / "reviewer_rerun" / "compute_overhead"),
    )
    args = parser.parse_args()
    output_dir = ensure_dir(args.output_dir)
    method_path = output_dir / "method_overhead.csv"
    rows = pd.read_csv(method_path).to_dict("records") if method_path.exists() else []
    completed_methods = {(row["dataset"], row["method"]) for row in rows}
    for dataset in parse_list(args.datasets):
        for method in parse_list(args.methods):
            if (dataset, method) in completed_methods:
                continue
            print(f"[Compute] {dataset} {method}", flush=True)
            rows.append(measure(args, dataset, method))
            completed_methods.add((dataset, method))
            pd.DataFrame(rows).to_csv(method_path, index=False)

    manifest = {
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch": torch.__version__,
        "flop_source": "torch.profiler with_flops; MACs reported as FLOPs/2 approximation",
        "full_stream_state": "starts from the source checkpoint; predictions and end-to-end transfer cost included",
        "batch_latency_state": "measured after the full stream and warm-up; host-to-device transfer excluded",
        "warmup_batches": args.warmup_batches,
        "measure_batches": args.measure_batches,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
