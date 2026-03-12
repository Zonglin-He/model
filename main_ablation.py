import argparse

from trainers.tta_trainer import TTATrainer


parser = argparse.ArgumentParser(description="ACCUP ablations (single entry with ablation_mode override)")
parser.add_argument('--save_dir', default='results/tta_experiments_logs', type=str, help='Directory containing all experiments')
parser.add_argument('--exp_name', default='All_Trg', type=str, help='experiment name')
parser.add_argument('--da_method', default='ACCUP', type=str, help='ACCUP, NoAdap')
parser.add_argument('--data-path', default=r'D:\PyCharm Project\ACCUP + EATA\data\Dataset', type=str)
parser.add_argument('--dataset', default='EEG', type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')
parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')
parser.add_argument('--num_runs', default=1, type=int, help='Number of consecutive run with different seeds')
parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')
parser.add_argument('--seed', default=42, type=int, help='Random seed applied to every run in this invocation')
parser.add_argument(
    '--seeds',
    type=str,
    default=None,
    help="Comma-separated seeds to run sequentially (e.g., '41,42,43'). Overrides --seed when provided.",
)
parser.add_argument(
    '--pretrain_cache_dir',
    type=str,
    default=None,
    help="Optional directory to cache/reuse pre-training weights across runs.",
)
parser.add_argument(
    '--disable_pretrain_cache',
    action='store_true',
    help="Force pre-training from scratch even if a cache directory is provided.",
)
parser.add_argument(
    '--scenario',
    action='append',
    default=None,
    help=(
        "Optional src->trg scenario filter. "
        "Example: --scenario 7->18 --scenario 16->1. "
        "If omitted, all dataset-defined scenarios will be evaluated."
    ),
)
parser.add_argument('--ablation_mode', type=str, default=None,
                    help="Ablation mode: full, no_warping, no_gate, naive_combo, inference_only")


def _run_single(seed_args):
    trainer = TTATrainer(seed_args)
    if seed_args.scenario:
        selected_pairs = []
        for entry in seed_args.scenario:
            if '->' in entry:
                src, trg = entry.split('->', 1)
            elif ',' in entry:
                src, trg = entry.split(',', 1)
            else:
                raise ValueError(f"Invalid scenario format '{entry}'. Expected 'src->trg'.")
            selected_pairs.append((str(src), str(trg)))
        trainer.dataset_configs.scenarios = selected_pairs
    trainer.test_time_adaptation()


if __name__ == "__main__":
    args = parser.parse_args()

    if args.seeds:
        try:
            seed_list = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        except Exception as exc:
            raise ValueError(f"Unable to parse --seeds='{args.seeds}'") from exc
    else:
        seed_list = [getattr(args, 'seed', 42)]

    base_exp_name = args.exp_name
    multiple = len(seed_list) > 1
    for seed in seed_list:
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = seed
        if multiple:
            seed_args.exp_name = f"{base_exp_name}_seed{seed}"
        _run_single(seed_args)
