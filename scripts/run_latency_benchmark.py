import argparse
import json
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.accup_instrumented import ACCUPInstrumented
from scripts.perturbation_analysis_utils import pgd_entropy_attack
from scripts.supplementary_utils import (
    RESULTS_ROOT,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    dataset_scenarios,
    ensure_dir,
    move_data_to_device,
)


CURRENT_METHODS = {
    "NoAdapt": {"da_method": "NoAdap", "tta_model_class": None},
    "NuSTAR": {"da_method": "ACCUP", "tta_model_class": ACCUPInstrumented},
}
DATASETS = ("EEG", "HAR", "FD")
DATASET_DISPLAY = {"EEG": "Sleep-EDF", "HAR": "UCI-HAR", "FD": "MFD"}
REPO_ROOTS = {
    "ACCUP": Path(r"D:\PyCharm Project\ACCUP"),
    "EATA": Path(r"D:\PyCharm Project\EATA"),
}
NUM_WARMUP_BATCHES = 10
NUM_MEASURE_BATCHES = 50
SEED = 42
NCAND_VALUES = [1, 4, 8, 16, 32, 64]
BATCH_SIZES = [16, 32, 64, 128, 256]
DEFAULT_PGD_STEPS = 10


def sync_if_needed(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def collect_batches(loader):
    return list(loader)


def representative_scenario(trainer):
    return dataset_scenarios(trainer)[0]


def extract_json(stdout):
    lines = [line.strip() for line in str(stdout).splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("External benchmark produced no stdout.")
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"Could not parse JSON from external benchmark output:\n{stdout}")


def run_external_probe(workdir, payload):
    code = textwrap.dedent(
        r"""
        import json
        import statistics
        import sys
        import time
        from pathlib import Path

        import torch


        def sync_if_needed(device):
            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()


        payload = json.loads(sys.argv[1])
        mode = payload["mode"]
        repo_root = Path(payload["repo_root"])
        sys.path.insert(0, str(repo_root))
        sys.path.insert(0, str(repo_root / "trainers"))

        if mode == "accup":
            import argparse
            import collections
            import os

            from optim.optimizer import build_optimizer
            from trainers.tta_trainer import TTATrainer
            from utils.utils import AverageMeter, fix_randomness, starting_logs

            os.chdir(str(repo_root))

            def move_data_to_device(data, device):
                if isinstance(data, tuple):
                    return tuple(move_data_to_device(item, device) for item in data)
                if isinstance(data, list):
                    return [move_data_to_device(item, device) for item in data]
                if torch.is_tensor(data):
                    return data.float().to(device)
                return data

            args = argparse.Namespace(
                save_dir="results/tta_experiments_logs",
                exp_name="latency_external",
                da_method="ACCUP",
                data_path=payload["data_path"],
                dataset=payload["dataset"],
                backbone=payload["backbone"],
                num_runs=1,
                device=payload["device"],
            )
            trainer = TTATrainer(args)
            trainer.run_id = 0
            src_id, trg_id = trainer.dataset_configs.scenarios[0]
            fix_randomness(payload["seed"])
            trainer.logger, trainer.scenario_log_dir = starting_logs(
                trainer.dataset,
                trainer.da_method,
                trainer.exp_log_dir,
                src_id,
                trg_id,
                0,
            )
            trainer.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
            trainer.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
            trainer.load_data_demo(src_id, trg_id, payload["seed"])

            pre_trained_model = None
            cache_path = trainer._resolve_pretrain_cache(src_id, trg_id)
            if cache_path:
                pre_trained_model = trainer.initialize_pretrained_model()
                cache_state = torch.load(cache_path, map_location=trainer.device)
                if isinstance(cache_state, dict) and "non_adapted" in cache_state:
                    pre_trained_model.network.load_state_dict(cache_state["non_adapted"])
                elif isinstance(cache_state, dict) and "model_state" in cache_state:
                    pre_trained_model.load_state_dict(cache_state["model_state"])
                else:
                    pre_trained_model.network.load_state_dict(cache_state)
            else:
                non_adapted_model_state, pre_trained_model = trainer.pre_train()
                load_pretrained_checkpoint_path = os.path.join(
                    trainer.load_pretrained_checkpoint, src_id + "_to_" + trg_id
                )
                os.makedirs(load_pretrained_checkpoint_path, exist_ok=True)
                trainer.save_checkpoint(trainer.home_path, load_pretrained_checkpoint_path, non_adapted_model_state)

            optimizer = build_optimizer(trainer.hparams)
            tta_model = trainer.get_tta_model_class()(
                trainer.dataset_configs,
                trainer.hparams,
                pre_trained_model,
                optimizer,
            ).to(trainer.device)
            pre_trained_model.eval()

            batches = list(trainer.trg_whole_dl)
            if str(payload["device"]).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device=trainer.device)

            timings = []
            total_needed = payload["warmup_batches"] + payload["measure_batches"]
            for batch_idx in range(total_needed):
                data, _, _ = batches[batch_idx % len(batches)]
                data = move_data_to_device(data, trainer.device)
                sync_if_needed(payload["device"])
                start = time.perf_counter()
                _ = tta_model(data)
                sync_if_needed(payload["device"])
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if batch_idx >= payload["warmup_batches"]:
                    timings.append(elapsed_ms)

            peak_gpu_mb = float("nan")
            if str(payload["device"]).startswith("cuda") and torch.cuda.is_available():
                peak_gpu_mb = torch.cuda.max_memory_allocated(device=trainer.device) / (1024 ** 2)

            result = {
                "method": "ACCUP",
                "dataset": payload["dataset"],
                "scenario": f"{src_id}->{trg_id}",
                "latency_mean_ms": float(statistics.mean(timings)),
                "latency_std_ms": float(statistics.pstdev(timings)),
                "peak_gpu_mb": float(peak_gpu_mb),
                "source_repo": "ACCUP",
            }
            trainer.summary_f1_scores.close()
            print(json.dumps(result))
        elif mode == "eata_timeseries":
            import argparse
            import math
            import os

            import eata
            import main_timeseries

            os.chdir(str(repo_root))

            args = argparse.Namespace()
            args.data_root = payload["data_path"]
            args.algorithm = payload["algorithm"].lower()
            args.batch_size = payload["batch_size"]
            args.workers = 2
            args.normalize = False
            args.base_channels = 64
            args.dropout = 0.2
            args.epochs_pretrain = 10
            args.lr_pretrain = 1e-3
            args.weight_decay = 0.0
            args.lr_adapt = 2.5e-4
            args.pretrained_seed = 0
            args.device = payload["device"]
            args.use_pretrain_cache = True
            args.pretrain_cache = str(repo_root / "pretrain_cache")
            args.fisher_size = 2000
            args.fisher_alpha = 2000.0
            args.fisher_clip_by_norm = 10.0
            args.e_margin = -1.0
            args.d_margin = 0.05

            dataset_name = payload["dataset"]
            src, tgt = main_timeseries.DEFAULT_PAIRS[dataset_name][0]
            augment = main_timeseries.StandardTimeSeriesAugment()
            train_set, train_loader, fisher_loader, test_set, test_loader = main_timeseries.build_loaders(
                main_timeseries.resolve_data_root(args.data_root),
                dataset_name,
                src,
                tgt,
                args.batch_size,
                args.workers,
                args.normalize,
                augment,
            )
            num_classes = int(torch.max(torch.cat([train_set.labels, test_set.labels])).item()) + 1
            if args.e_margin <= 0:
                args.e_margin = math.log(num_classes) * 0.4

            pretrained_state, _ = main_timeseries._load_pretrained_state(
                args.pretrain_cache,
                dataset_name,
                src,
                logger=type("Logger", (), {"warning": lambda *a, **k: None})(),
            )
            if pretrained_state is None:
                raise RuntimeError(
                    f"No pretrain cache found for {dataset_name} src{src} in {args.pretrain_cache}."
                )
            model_spec = main_timeseries.infer_timeseries_cnn_spec(pretrained_state)
            model_builder = lambda: main_timeseries.build_timeseries_model_from_spec(model_spec)

            fishers = None
            if args.algorithm == "eata":
                fisher_model = model_builder()
                fisher_model.load_state_dict(pretrained_state)
                fisher_model = eata.configure_model(fisher_model)
                fishers = main_timeseries.compute_fisher(
                    fisher_model,
                    fisher_loader,
                    args.device,
                    args.fisher_size,
                    args.fisher_clip_by_norm,
                )

            model = model_builder()
            model.load_state_dict(pretrained_state)
            adapt_model = main_timeseries.build_adapt_model(
                model,
                args.algorithm,
                args,
                fisher_loader,
                fishers_cache=fishers,
            ).to(args.device)

            batches = list(test_loader)
            if str(payload["device"]).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device=torch.device(args.device))

            timings = []
            total_needed = payload["warmup_batches"] + payload["measure_batches"]
            for batch_idx in range(total_needed):
                x, _ = batches[batch_idx % len(batches)]
                x = x.to(args.device, non_blocking=True)
                sync_if_needed(payload["device"])
                start = time.perf_counter()
                _ = adapt_model(x)
                sync_if_needed(payload["device"])
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if batch_idx >= payload["warmup_batches"]:
                    timings.append(elapsed_ms)

            peak_gpu_mb = float("nan")
            if str(payload["device"]).startswith("cuda") and torch.cuda.is_available():
                peak_gpu_mb = torch.cuda.max_memory_allocated(device=torch.device(args.device)) / (1024 ** 2)

            result = {
                "method": payload["algorithm"].upper(),
                "dataset": dataset_name,
                "scenario": f"{src}->{tgt}",
                "latency_mean_ms": float(statistics.mean(timings)),
                "latency_std_ms": float(statistics.pstdev(timings)),
                "peak_gpu_mb": float(peak_gpu_mb),
                "source_repo": "EATA",
            }
            print(json.dumps(result))
        else:
            raise ValueError(f"Unsupported external benchmark mode: {mode}")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(payload)],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"External benchmark failed for {payload['mode']} ({payload.get('dataset')}).\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return extract_json(result.stdout)


def measure_internal_method(
    data_path,
    device,
    dataset,
    method_name,
    backbone,
    da_method,
    tta_model_class=None,
    override=None,
):
    trainer = build_trainer(
        data_path=data_path,
        device=device,
        dataset=dataset,
        da_method=da_method,
        tta_model_class=tta_model_class,
        exp_name="latency",
        seed=SEED,
        backbone=backbone,
    )
    override = override or {}
    tta_model = None
    pre_trained_model = None
    try:
        src_id, trg_id = representative_scenario(trainer)
        if override:
            trainer.store_scenario_override(src_id, trg_id, override)
        tta_model, pre_trained_model = create_tta_model(trainer, src_id, trg_id, run_seed=SEED)
        batches = collect_batches(trainer.trg_whole_dl)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=trainer.device)

        timings = []
        total_needed = NUM_WARMUP_BATCHES + NUM_MEASURE_BATCHES
        for batch_idx in range(total_needed):
            data, _, _ = batches[batch_idx % len(batches)]
            data = move_data_to_device(data, trainer.device)
            sync_if_needed(device)
            start = time.perf_counter()
            _ = tta_model(data)
            sync_if_needed(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if batch_idx >= NUM_WARMUP_BATCHES:
                timings.append(elapsed_ms)

        peak_gpu_mb = float("nan")
        if str(device).startswith("cuda") and torch.cuda.is_available():
            peak_gpu_mb = torch.cuda.max_memory_allocated(device=trainer.device) / (1024 ** 2)

        return {
            "method": method_name,
            "dataset": dataset,
            "scenario": f"{src_id}->{trg_id}",
            "latency_mean_ms": float(statistics.mean(timings)),
            "latency_std_ms": float(statistics.pstdev(timings)),
            "peak_gpu_mb": peak_gpu_mb,
            "source_repo": "ACCUP + EATA",
        }
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)


def measure_external_accup(data_path, device, dataset, backbone, batch_size):
    payload = {
        "mode": "accup",
        "repo_root": str(REPO_ROOTS["ACCUP"]),
        "data_path": str(data_path),
        "dataset": dataset,
        "backbone": backbone,
        "device": device,
        "seed": SEED,
        "batch_size": batch_size,
        "warmup_batches": NUM_WARMUP_BATCHES,
        "measure_batches": NUM_MEASURE_BATCHES,
    }
    return run_external_probe(REPO_ROOTS["ACCUP"], payload)


def measure_external_eata_family(data_path, device, dataset, backbone, algorithm, batch_size):
    payload = {
        "mode": "eata_timeseries",
        "repo_root": str(REPO_ROOTS["EATA"]),
        "data_path": str(data_path),
        "dataset": dataset,
        "backbone": backbone,
        "device": device,
        "algorithm": algorithm,
        "batch_size": batch_size,
        "warmup_batches": NUM_WARMUP_BATCHES,
        "measure_batches": NUM_MEASURE_BATCHES,
    }
    return run_external_probe(REPO_ROOTS["EATA"], payload)


def measure_search_cost(data_path, device, dataset, backbone, search_method, pgd_steps=DEFAULT_PGD_STEPS):
    trainer = build_trainer(
        data_path=data_path,
        device=device,
        dataset=dataset,
        da_method="ACCUP",
        tta_model_class=ACCUPInstrumented,
        exp_name="latency_search",
        seed=SEED,
        backbone=backbone,
    )
    tta_model = None
    pre_trained_model = None
    try:
        src_id, trg_id = representative_scenario(trainer)
        tta_model, pre_trained_model = create_tta_model(trainer, src_id, trg_id, run_seed=SEED)
        batches = collect_batches(trainer.trg_whole_dl)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=trainer.device)

        timings = []
        total_needed = NUM_WARMUP_BATCHES + NUM_MEASURE_BATCHES
        for batch_idx in range(total_needed):
            data, _, _ = batches[batch_idx % len(batches)]
            batch_data = move_data_to_device(data, trainer.device)
            raw_x = tta_model._extract_primary_tensor(batch_data)
            sync_if_needed(device)
            start = time.perf_counter()
            if search_method == "SSAW":
                _ = tta_model.get_adversarial_view(raw_x, tta_model.model)
            elif str(search_method).startswith("PGD"):
                _ = pgd_entropy_attack(
                    tta_model.model,
                    raw_x,
                    eps=float(getattr(tta_model, "adv_sigma", 0.1) or 0.1),
                    steps=int(pgd_steps),
                )
            else:
                raise ValueError(f"Unsupported search method: {search_method}")
            sync_if_needed(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if batch_idx >= NUM_WARMUP_BATCHES:
                timings.append(elapsed_ms)

        peak_gpu_mb = float("nan")
        if str(device).startswith("cuda") and torch.cuda.is_available():
            peak_gpu_mb = torch.cuda.max_memory_allocated(device=trainer.device) / (1024 ** 2)

        return {
            "dataset": dataset,
            "scenario": f"{src_id}->{trg_id}",
            "search_method": search_method,
            "latency_mean_ms": float(statistics.mean(timings)),
            "latency_std_ms": float(statistics.pstdev(timings)),
            "peak_gpu_mb": peak_gpu_mb,
            "pgd_steps": int(pgd_steps) if search_method.startswith("PGD") else 0,
            "adv_num_candidates": int(getattr(tta_model, "adv_num_candidates", 0)),
        }
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)


def build_latency_results(data_path, device, backbone):
    rows = []
    for dataset in DATASETS:
        dataset_root = Path(data_path) / dataset
        if not dataset_root.exists():
            print(f"[Skip] {dataset} not found at {dataset_root}")
            continue

        print(f"[Latency] {dataset} NoAdapt", flush=True)
        rows.append(
            measure_internal_method(
                data_path=data_path,
                device=device,
                dataset=dataset,
                method_name="NoAdapt",
                backbone=backbone,
                da_method="NoAdap",
                tta_model_class=None,
            )
        )

        print(f"[Latency] {dataset} TENT", flush=True)
        rows.append(
            measure_external_eata_family(
                data_path=data_path,
                device=device,
                dataset=dataset,
                backbone=backbone,
                algorithm="tent",
                batch_size=64,
            )
        )

        print(f"[Latency] {dataset} EATA", flush=True)
        rows.append(
            measure_external_eata_family(
                data_path=data_path,
                device=device,
                dataset=dataset,
                backbone=backbone,
                algorithm="eata",
                batch_size=64,
            )
        )

        print(f"[Latency] {dataset} ACCUP", flush=True)
        rows.append(
            measure_external_accup(
                data_path=data_path,
                device=device,
                dataset=dataset,
                backbone=backbone,
                batch_size=64,
            )
        )

        print(f"[Latency] {dataset} NuSTAR", flush=True)
        rows.append(
            measure_internal_method(
                data_path=data_path,
                device=device,
                dataset=dataset,
                method_name="NuSTAR",
                backbone=backbone,
                da_method="ACCUP",
                tta_model_class=ACCUPInstrumented,
            )
        )

    latency_df = pd.DataFrame(rows)
    if latency_df.empty:
        return latency_df

    no_adapt_lookup = latency_df[latency_df["method"] == "NoAdapt"].set_index("dataset")["latency_mean_ms"].to_dict()
    latency_df["overhead_vs_noadapt_ms"] = latency_df["dataset"].map(no_adapt_lookup)
    latency_df["overhead_vs_noadapt_ms"] = latency_df["latency_mean_ms"] - latency_df["overhead_vs_noadapt_ms"]
    latency_df["overhead_vs_noadapt_x"] = latency_df.apply(
        lambda row: float("nan")
        if no_adapt_lookup.get(row["dataset"], 0.0) == 0.0
        else row["latency_mean_ms"] / no_adapt_lookup[row["dataset"]],
        axis=1,
    )
    latency_df["dataset_display"] = latency_df["dataset"].map(DATASET_DISPLAY).fillna(latency_df["dataset"])
    return latency_df


def build_nustar_vs_baselines(latency_df):
    rows = []
    for dataset, dataset_df in latency_df.groupby("dataset"):
        nustar_df = dataset_df[dataset_df["method"] == "NuSTAR"]
        if nustar_df.empty:
            continue
        nustar_latency = float(nustar_df.iloc[0]["latency_mean_ms"])
        for baseline in ("TENT", "EATA", "ACCUP", "NoAdapt"):
            baseline_df = dataset_df[dataset_df["method"] == baseline]
            if baseline_df.empty:
                continue
            baseline_latency = float(baseline_df.iloc[0]["latency_mean_ms"])
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_display": DATASET_DISPLAY.get(dataset, dataset),
                    "scenario": str(nustar_df.iloc[0]["scenario"]),
                    "baseline": baseline,
                    "baseline_latency_ms": baseline_latency,
                    "nustar_latency_ms": nustar_latency,
                    "extra_overhead_ms": nustar_latency - baseline_latency,
                    "extra_overhead_ratio": nustar_latency / baseline_latency if baseline_latency > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_search_results(data_path, device, backbone, pgd_steps):
    rows = []
    for dataset in DATASETS:
        dataset_root = Path(data_path) / dataset
        if not dataset_root.exists():
            continue
        pgd_label = f"PGD-{pgd_steps}"
        print(f"[Search] {dataset} SSAW", flush=True)
        rows.append(measure_search_cost(data_path, device, dataset, backbone, "SSAW", pgd_steps=pgd_steps))
        print(f"[Search] {dataset} {pgd_label}", flush=True)
        rows.append(measure_search_cost(data_path, device, dataset, backbone, pgd_label, pgd_steps=pgd_steps))

    search_df = pd.DataFrame(rows)
    if search_df.empty:
        return search_df, pd.DataFrame()

    search_df["dataset_display"] = search_df["dataset"].map(DATASET_DISPLAY).fillna(search_df["dataset"])
    paired_rows = []
    pgd_label = f"PGD-{pgd_steps}"
    for dataset, dataset_df in search_df.groupby("dataset"):
        ssaw_df = dataset_df[dataset_df["search_method"] == "SSAW"]
        pgd_df = dataset_df[dataset_df["search_method"] == pgd_label]
        if ssaw_df.empty or pgd_df.empty:
            continue
        ssaw_latency = float(ssaw_df.iloc[0]["latency_mean_ms"])
        pgd_latency = float(pgd_df.iloc[0]["latency_mean_ms"])
        paired_rows.append(
            {
                "dataset": dataset,
                "dataset_display": DATASET_DISPLAY.get(dataset, dataset),
                "scenario": str(ssaw_df.iloc[0]["scenario"]),
                "ssaw_latency_ms": ssaw_latency,
                "pgd_latency_ms": pgd_latency,
                "saved_ms": pgd_latency - ssaw_latency,
                "speedup_vs_pgd": pgd_latency / ssaw_latency if ssaw_latency > 0 else float("nan"),
                "ssaw_peak_gpu_mb": float(ssaw_df.iloc[0]["peak_gpu_mb"]),
                "pgd_peak_gpu_mb": float(pgd_df.iloc[0]["peak_gpu_mb"]),
                "ssaw_num_candidates": int(ssaw_df.iloc[0]["adv_num_candidates"]),
                "pgd_steps": int(pgd_df.iloc[0]["pgd_steps"]),
            }
        )
    return search_df, pd.DataFrame(paired_rows)


def build_ncand_curve(data_path, device, backbone):
    rows = []
    dataset = "EEG"
    if not (Path(data_path) / dataset).exists():
        return pd.DataFrame()
    for ncand in NCAND_VALUES:
        rows.append(
            {
                **measure_internal_method(
                    data_path=data_path,
                    device=device,
                    dataset=dataset,
                    method_name="NuSTAR",
                    backbone=backbone,
                    da_method="ACCUP",
                    tta_model_class=ACCUPInstrumented,
                    override={
                        "adv_num_candidates": ncand,
                        "enable_ssaw": True,
                        "adv_sigma": 0.1,
                    },
                ),
                "adv_num_candidates": ncand,
            }
        )
    return pd.DataFrame(rows)


def build_batchsize_curve(data_path, device, backbone):
    rows = []
    dataset = "EEG"
    if not (Path(data_path) / dataset).exists():
        return pd.DataFrame()
    for batch_size in BATCH_SIZES:
        for method_name, method_cfg in CURRENT_METHODS.items():
            rows.append(
                {
                    **measure_internal_method(
                        data_path=data_path,
                        device=device,
                        dataset=dataset,
                        method_name=method_name,
                        backbone=backbone,
                        da_method=method_cfg["da_method"],
                        tta_model_class=method_cfg["tta_model_class"],
                        override={"batch_size": batch_size},
                    ),
                    "batch_size": batch_size,
                }
            )
    return pd.DataFrame(rows)


def plot_latency_comparison(df, output_path):
    if df.empty:
        return
    pivot = df.pivot(index="dataset_display", columns="method", values="latency_mean_ms")
    method_order = [method for method in ("NoAdapt", "TENT", "EATA", "ACCUP", "NuSTAR") if method in pivot.columns]
    pivot = pivot[method_order]
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Latency (ms/batch)")
    ax.set_title("Per-batch adaptation latency")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_search_comparison(df, output_path):
    if df.empty:
        return
    plot_df = df.set_index("dataset_display")[["ssaw_latency_ms", "pgd_latency_ms"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylabel("Latency (ms/batch)")
    ax.set_title("SSAW search vs PGD search cost")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_ncand_curve(df, output_path):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(df["adv_num_candidates"], df["latency_mean_ms"], yerr=df["latency_std_ms"], marker="o")
    ax.set_xlabel("Ncand")
    ax.set_ylabel("Latency (ms/batch)")
    ax.set_title("NuSTAR latency vs number of sampled candidates")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_batchsize_curve(df, output_path):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, sub_df in df.groupby("method"):
        ax.plot(sub_df["batch_size"], sub_df["latency_mean_ms"], marker="o", label=method)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency (ms/batch)")
    ax.set_title("Batch-size sensitivity")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_markdown_summary(output_dir, latency_df, paired_df, search_pair_df, pgd_steps):
    lines = ["# Latency Benchmark Summary", ""]
    if not latency_df.empty:
        latency_view = latency_df[
            [
                "dataset_display",
                "scenario",
                "method",
                "latency_mean_ms",
                "latency_std_ms",
                "peak_gpu_mb",
                "overhead_vs_noadapt_ms",
            ]
        ].copy()
        latency_view = latency_view.rename(
            columns={
                "dataset_display": "Dataset",
                "scenario": "Scenario",
                "method": "Method",
                "latency_mean_ms": "Latency (ms)",
                "latency_std_ms": "Std (ms)",
                "peak_gpu_mb": "GPU Mem (MB)",
                "overhead_vs_noadapt_ms": "Overhead vs NoAdapt (ms)",
            }
        )
        lines.append("## Absolute latency")
        lines.append("")
        lines.append(latency_view.to_markdown(index=False, floatfmt=".2f"))
        lines.append("")

    if not paired_df.empty:
        paired_view = paired_df[
            [
                "dataset_display",
                "scenario",
                "baseline",
                "baseline_latency_ms",
                "nustar_latency_ms",
                "extra_overhead_ms",
                "extra_overhead_ratio",
            ]
        ].copy()
        paired_view = paired_view.rename(
            columns={
                "dataset_display": "Dataset",
                "scenario": "Scenario",
                "baseline": "Baseline",
                "baseline_latency_ms": "Baseline (ms)",
                "nustar_latency_ms": "NuSTAR (ms)",
                "extra_overhead_ms": "Extra Overhead (ms)",
                "extra_overhead_ratio": "NuSTAR / Baseline",
            }
        )
        lines.append("## NuSTAR extra overhead against baselines")
        lines.append("")
        lines.append(paired_view.to_markdown(index=False, floatfmt=".2f"))
        lines.append("")

    if not search_pair_df.empty:
        search_view = search_pair_df[
            [
                "dataset_display",
                "scenario",
                "ssaw_latency_ms",
                "pgd_latency_ms",
                "saved_ms",
                "speedup_vs_pgd",
                "ssaw_num_candidates",
                "pgd_steps",
            ]
        ].copy()
        search_view = search_view.rename(
            columns={
                "dataset_display": "Dataset",
                "scenario": "Scenario",
                "ssaw_latency_ms": "SSAW (ms)",
                "pgd_latency_ms": f"PGD-{pgd_steps} (ms)",
                "saved_ms": "Saved (ms)",
                "speedup_vs_pgd": "PGD / SSAW",
                "ssaw_num_candidates": "Ncand",
                "pgd_steps": "PGD steps",
            }
        )
        lines.append(f"## SSAW vs PGD-{pgd_steps} search cost")
        lines.append("")
        lines.append(search_view.to_markdown(index=False, floatfmt=".2f"))
        lines.append("")

    (output_dir / "latency_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--pgd_steps", default=DEFAULT_PGD_STEPS, type=int)
    args = parser.parse_args()

    output_dir = ensure_dir(RESULTS_ROOT / "latency")

    latency_df = build_latency_results(args.data_path, args.device, args.backbone)
    latency_df.to_csv(output_dir / "latency_results.csv", index=False)

    paired_df = build_nustar_vs_baselines(latency_df)
    paired_df.to_csv(output_dir / "nustar_extra_overhead_vs_baselines.csv", index=False)

    search_df, search_pair_df = build_search_results(args.data_path, args.device, args.backbone, args.pgd_steps)
    search_df.to_csv(output_dir / "search_cost_results.csv", index=False)
    search_pair_df.to_csv(output_dir / "search_cost_pairwise.csv", index=False)

    ncand_df = build_ncand_curve(args.data_path, args.device, args.backbone)
    ncand_df.to_csv(output_dir / "ncand_cost_curve.csv", index=False)

    batch_df = build_batchsize_curve(args.data_path, args.device, args.backbone)
    batch_df.to_csv(output_dir / "batchsize_cost_curve.csv", index=False)

    plot_latency_comparison(latency_df, output_dir / "latency_comparison.pdf")
    plot_search_comparison(search_pair_df, output_dir / "search_cost_comparison.pdf")
    plot_ncand_curve(ncand_df, output_dir / "ncand_cost_curve.pdf")
    plot_batchsize_curve(batch_df, output_dir / "batchsize_cost_curve.pdf")
    write_markdown_summary(output_dir, latency_df, paired_df, search_pair_df, args.pgd_steps)

    print("Latency benchmark completed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
