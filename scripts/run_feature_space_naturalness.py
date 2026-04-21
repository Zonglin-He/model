import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.accup_instrumented import ACCUPInstrumented
from scripts.perturbation_analysis_utils import (
    gaussian_noise_view,
    magnitude_warp_view,
    pgd_entropy_attack,
    time_warp_view,
)
from scripts.supplementary_utils import (
    RESULTS_ROOT,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    ensure_dir,
    extract_primary_tensor,
)


GENERIC_VIEW_BUILDERS = {
    "gaussian_noise": lambda model, x, args: gaussian_noise_view(x),
    "magnitude_warp": lambda model, x, args: magnitude_warp_view(x),
    "time_warp": lambda model, x, args: time_warp_view(x),
    "pgd_entropy": lambda model, x, args: pgd_entropy_attack(model, x, eps=args.pgd_eps, steps=args.pgd_steps),
}
VIEW_MARKERS = {"original": "o", "generic": "^", "ssaw": "s"}


def collect_samples(loader, limit):
    x_chunks, y_chunks = [], []
    total = 0
    for data, labels, _ in loader:
        primary = extract_primary_tensor(data)
        x_chunks.append(primary)
        y_chunks.append(labels)
        total += primary.size(0)
        if total >= limit:
            break
    return torch.cat(x_chunks, dim=0)[:limit], torch.cat(y_chunks, dim=0)[:limit]


def reduce_features(features, method="tsne"):
    features_np = features.detach().cpu().numpy()
    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42)
            return reducer.fit_transform(features_np)
        except Exception:
            method = "tsne"
    perplexity = min(30, max(5, (features_np.shape[0] - 1) // 3))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
    )
    return reducer.fit_transform(features_np)


def extract_features(model, x):
    feats, _ = model.feature_extractor(x)
    return feats


def feature_distance(a, b):
    return 1.0 - F.cosine_similarity(F.normalize(a, dim=1), F.normalize(b, dim=1), dim=1)


def plot_embedding(embedding_df, class_names, output_path, title):
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(class_names))))
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for class_idx, class_name in enumerate(class_names):
        class_df = embedding_df[embedding_df["class_id"] == class_idx]
        if class_df.empty:
            continue
        for view_name, marker in VIEW_MARKERS.items():
            sub_df = class_df[class_df["view_type"] == view_name]
            if sub_df.empty:
                continue
            ax.scatter(
                sub_df["x"],
                sub_df["y"],
                s=18,
                alpha=0.65,
                c=[colors[class_idx]],
                marker=marker,
                label=f"{class_name}-{view_name}",
            )
    handles, labels = ax.get_legend_handles_labels()
    keep = {}
    for handle, label in zip(handles, labels):
        keep.setdefault(label, handle)
    ax.legend(keep.values(), keep.keys(), fontsize=7, frameon=False, ncol=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset", default="EEG")
    parser.add_argument("--scenario", required=True, help="src->trg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--max_samples", type=int, default=600)
    parser.add_argument("--generic_view", default="magnitude_warp", choices=sorted(GENERIC_VIEW_BUILDERS.keys()))
    parser.add_argument("--embedding", default="tsne", choices=["tsne", "umap"])
    parser.add_argument("--pgd_eps", type=float, default=0.1)
    parser.add_argument("--pgd_steps", type=int, default=10)
    args = parser.parse_args()

    output_dir = ensure_dir(RESULTS_ROOT / "feature_space_naturalness" / f"{args.dataset}_{args.scenario.replace('->', 'to')}_{args.generic_view}")
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=args.dataset,
        da_method="ACCUP",
        backbone=args.backbone,
        exp_name="feature_space_naturalness",
        seed=args.seed,
        tta_model_class=ACCUPInstrumented,
    )
    tta_model = None
    pre_trained_model = None
    try:
        src_id, trg_id = args.scenario.split("->", 1)
        tta_model, pre_trained_model = create_tta_model(trainer, src_id, trg_id, run_seed=args.seed)
        raw_x, labels = collect_samples(trainer.trg_whole_dl, args.max_samples)
        raw_x = raw_x.to(trainer.device)
        labels = labels.to(trainer.device)

        generic_x = GENERIC_VIEW_BUILDERS[args.generic_view](tta_model.model, raw_x, args)
        ssaw_x = tta_model.get_adversarial_view(raw_x, tta_model.model)

        with torch.no_grad():
            raw_feats = extract_features(tta_model.model, raw_x)
            generic_feats = extract_features(tta_model.model, generic_x)
            ssaw_feats = extract_features(tta_model.model, ssaw_x)

        stacked_feats = torch.cat([raw_feats, generic_feats, ssaw_feats], dim=0)
        embedding = reduce_features(stacked_feats, method=args.embedding)
        split = raw_feats.size(0)
        raw_embed = embedding[:split]
        generic_embed = embedding[split : 2 * split]
        ssaw_embed = embedding[2 * split :]

        labels_np = labels.detach().cpu().numpy()
        embedding_df = pd.concat(
            [
                pd.DataFrame({"x": raw_embed[:, 0], "y": raw_embed[:, 1], "class_id": labels_np, "view_type": "original"}),
                pd.DataFrame({"x": generic_embed[:, 0], "y": generic_embed[:, 1], "class_id": labels_np, "view_type": "generic"}),
                pd.DataFrame({"x": ssaw_embed[:, 0], "y": ssaw_embed[:, 1], "class_id": labels_np, "view_type": "ssaw"}),
            ],
            ignore_index=True,
        )
        embedding_df.to_csv(output_dir / "embedding_points.csv", index=False)

        stats_df = pd.DataFrame(
            [
                {
                    "comparison": f"raw_vs_{args.generic_view}",
                    "mean_feature_distance": float(feature_distance(raw_feats, generic_feats).mean().item()),
                },
                {
                    "comparison": "raw_vs_ssaw",
                    "mean_feature_distance": float(feature_distance(raw_feats, ssaw_feats).mean().item()),
                },
            ]
        )
        stats_df.to_csv(output_dir / "feature_distance_summary.csv", index=False)

        plot_embedding(
            embedding_df,
            trainer.dataset_configs.class_names,
            output_dir / f"{args.embedding}_{args.generic_view}_vs_ssaw.pdf",
            title=f"{args.dataset} {args.scenario} | original vs {args.generic_view} vs ssaw",
        )
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)

    print(f"Feature-space naturalness visualization completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
