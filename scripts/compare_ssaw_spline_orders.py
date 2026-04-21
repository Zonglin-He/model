import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import make_interp_spline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.accup import NuSTAR_ActiveSearch
from scripts.supplementary_utils import (
    RESULTS_ROOT,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    dataset_scenarios,
    ensure_dir,
)


ORDER_NAMES = {
    2: "quadratic",
    3: "cubic",
    4: "quartic",
}


def parse_int_list(text):
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_dataset_list(text):
    values = [item.strip().upper() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one dataset must be provided.")
    return values


def parse_order_list(text):
    orders = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not orders:
        raise ValueError("At least one spline order must be provided.")
    invalid = [order for order in orders if order not in ORDER_NAMES]
    if invalid:
        raise ValueError(f"Unsupported spline orders: {invalid}. Supported: {sorted(ORDER_NAMES)}")
    return orders


def parse_scenario_filters(entries):
    if not entries:
        return {}
    filters = {}
    for entry in entries:
        text = str(entry).strip()
        if ":" not in text or "->" not in text:
            raise ValueError(
                f"Invalid --scenario '{entry}'. Expected DATASET:src->trg, e.g. EEG:0->11."
            )
        dataset, scenario = text.split(":", 1)
        src, trg = scenario.split("->", 1)
        filters.setdefault(dataset.strip().upper(), set()).add((src.strip(), trg.strip()))
    return filters


class SciPySplineOrderActiveSearch(NuSTAR_ActiveSearch):
    def __init__(self, spline_order: int, num_control_points: int = 10, num_candidates: int = 16, sigma: float = 0.1):
        super().__init__(
            num_control_points=num_control_points,
            num_candidates=num_candidates,
            sigma=sigma,
        )
        self.spline_order = int(spline_order)

    def _spline_upsample(self, controls: torch.Tensor, target_len: int) -> torch.Tensor:
        if controls.dim() != 2:
            raise ValueError(f"Expected control tensor with shape [N, M], got {tuple(controls.shape)}")
        if target_len <= 0:
            raise ValueError("target_len must be positive")
        if controls.size(1) == 1:
            return controls.repeat(1, target_len)

        device = controls.device
        dtype = controls.dtype
        ctrl_np = controls.detach().cpu().numpy().astype(np.float64, copy=False)
        num_ctrl = ctrl_np.shape[1]
        order = min(int(self.spline_order), num_ctrl - 1)
        ctrl_x = np.linspace(0.0, float(target_len - 1), num_ctrl, dtype=np.float64)
        eval_x = np.linspace(0.0, float(target_len - 1), target_len, dtype=np.float64)

        curves = []
        for row in ctrl_np:
            if np.allclose(row, row[0]):
                curve = np.full(target_len, row[0], dtype=np.float64)
            else:
                spline = make_interp_spline(ctrl_x, row, k=order)
                curve = np.asarray(spline(eval_x), dtype=np.float64)
                curve = np.nan_to_num(curve, nan=1.0, posinf=1.0, neginf=1.0)
            curves.append(curve)

        stacked = np.stack(curves, axis=0)
        return torch.from_numpy(stacked).to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, model, sigma=None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B, C, T], got {tuple(x.shape)}")

        sigma = self.sigma if sigma is None else float(sigma)
        batch_size, channels, target_len = x.shape
        if self.num_candidates <= 0 or sigma <= 0.0:
            warp = torch.ones(batch_size, 1, target_len, device=x.device, dtype=x.dtype)
            self.last_warp_curves = warp.detach().cpu()
            self.last_warp_curve = warp[:1].detach().cpu()
            self.last_metadata = {
                "mode": "identity",
                "curve": warp.squeeze(1).detach().cpu(),
                "control_points": torch.ones(batch_size, 1, device=x.device, dtype=x.dtype).cpu(),
                "spline_order": self.spline_order,
            }
            return x

        controls = self._sample_control_points(batch_size, x.device, x.dtype, sigma)
        flat_controls = controls.reshape(batch_size * self.num_candidates, self.num_control_points)
        upsampled = self._spline_upsample(flat_controls, target_len)
        warps = upsampled.reshape(batch_size, self.num_candidates, target_len)
        warped_x = x.unsqueeze(1) * warps.unsqueeze(2)
        flat_x = warped_x.reshape(batch_size * self.num_candidates, channels, target_len)

        feats = self._extract_features(model, flat_x)
        logits = model.classifier(feats)
        entropy = torch.sum(-torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1), dim=1)
        entropy = entropy.reshape(batch_size, self.num_candidates)
        best_idx = entropy.argmax(dim=1)
        best_warp = warps[torch.arange(batch_size, device=x.device), best_idx]

        self.last_warp_curves = best_warp.unsqueeze(1).detach().cpu()
        self.last_warp_curve = best_warp[:1].unsqueeze(1).detach().cpu()
        self.last_metadata = {
            "mode": f"spline_order_{self.spline_order}",
            "curve": best_warp.detach().cpu(),
            "control_points": controls[torch.arange(batch_size, device=x.device), best_idx].detach().cpu(),
            "score": entropy[torch.arange(batch_size, device=x.device), best_idx].detach().cpu(),
            "spline_order": self.spline_order,
        }
        return x * best_warp.unsqueeze(1)


def configure_spline_order(tta_model, spline_order: int):
    if spline_order == 3:
        return
    current = tta_model.active_search
    tta_model.active_search = SciPySplineOrderActiveSearch(
        spline_order=spline_order,
        num_control_points=current.num_control_points,
        num_candidates=current.num_candidates,
        sigma=current.sigma,
    )


def evaluate_order_scenario(data_path, device, dataset, seed, backbone, spline_order, src_id, trg_id):
    trainer = build_trainer(
        data_path=data_path,
        device=device,
        dataset=dataset,
        da_method="ACCUP",
        exp_name="ssaw_spline_order_compare",
        seed=seed,
        backbone=backbone,
    )
    tta_model = None
    pre_trained_model = None
    try:
        tta_model, pre_trained_model = create_tta_model(trainer, src_id, trg_id, run_seed=seed)
        configure_spline_order(tta_model, spline_order)
        f1 = float(trainer.calculate_metrics(tta_model)[1])
        return {
            "dataset": dataset,
            "scenario": f"{src_id}->{trg_id}",
            "seed": seed,
            "spline_order": spline_order,
            "spline_name": ORDER_NAMES[spline_order],
            "f1": f1,
            "adv_sigma": float(trainer.hparams.get("adv_sigma", 0.0)),
            "adv_num_candidates": int(trainer.hparams.get("adv_num_candidates", 0)),
            "batch_size": int(trainer.hparams.get("batch_size", 0)),
            "learning_rate": float(trainer.hparams.get("learning_rate", 0.0)),
            "pre_learning_rate": float(trainer.hparams.get("pre_learning_rate", 0.0)),
        }
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--datasets", default="EEG,HAR,FD")
    parser.add_argument("--orders", default="2,3,4")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Optional filter DATASET:src->trg. May be passed multiple times.",
    )
    args = parser.parse_args()

    output_dir = ensure_dir(RESULTS_ROOT / "ssaw_spline_order_compare")
    seeds = parse_int_list(args.seeds)
    datasets = parse_dataset_list(args.datasets)
    orders = parse_order_list(args.orders)
    scenario_filters = parse_scenario_filters(args.scenario)

    raw_rows = []
    for dataset in datasets:
        if not (Path(args.data_path) / dataset).exists():
            print(f"[Skip] {dataset} not found", flush=True)
            continue

        probe_trainer = build_trainer(
            data_path=args.data_path,
            device=args.device,
            dataset=dataset,
            da_method="ACCUP",
            exp_name="ssaw_spline_order_compare_probe",
            seed=seeds[0],
            backbone=args.backbone,
        )
        scenarios = dataset_scenarios(probe_trainer)
        cleanup_trainer(probe_trainer, close_summary=True)

        if dataset in scenario_filters:
            wanted = scenario_filters[dataset]
            scenarios = [scenario for scenario in scenarios if scenario in wanted]

        for spline_order in orders:
            for seed in seeds:
                print(
                    f"[Run] dataset={dataset} order={spline_order} seed={seed} scenarios={len(scenarios)}",
                    flush=True,
                )
                for src_id, trg_id in scenarios:
                    raw_rows.append(
                        evaluate_order_scenario(
                            data_path=args.data_path,
                            device=args.device,
                            dataset=dataset,
                            seed=seed,
                            backbone=args.backbone,
                            spline_order=spline_order,
                            src_id=src_id,
                            trg_id=trg_id,
                        )
                    )

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(output_dir / "raw_results.csv", index=False)

    scenario_pivot = (
        raw_df.groupby(["dataset", "scenario", "spline_name"])["f1"]
        .mean()
        .reset_index()
        .pivot(index=["dataset", "scenario"], columns="spline_name", values="f1")
        .reset_index()
    )
    scenario_pivot.to_csv(output_dir / "scenario_results.csv", index=False)

    dataset_pivot = (
        raw_df.groupby(["dataset", "spline_name"])["f1"]
        .mean()
        .reset_index()
        .pivot(index="dataset", columns="spline_name", values="f1")
        .reset_index()
    )
    dataset_pivot.to_csv(output_dir / "dataset_results.csv", index=False)

    cubic_baseline = (
        raw_df[raw_df["spline_order"] == 3]
        .groupby(["dataset", "scenario"])["f1"]
        .mean()
        .reset_index()
        .rename(columns={"f1": "cubic"})
    )
    delta_df = (
        raw_df.groupby(["dataset", "scenario", "spline_name"])["f1"]
        .mean()
        .reset_index()
        .merge(cubic_baseline, on=["dataset", "scenario"], how="left")
    )
    delta_df["delta_vs_cubic"] = delta_df["f1"] - delta_df["cubic"]
    delta_df.to_csv(output_dir / "delta_vs_cubic.csv", index=False)

    print("SSAW spline-order comparison completed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
