# Repository Guidelines

## Project Structure & Module Organization
- `algorithms/` holds DuSafe adaptation logic; expose production variants through `get_tta_class.py`.
- `trainers/tta_trainer.py` is the CLI entry point that orchestrates data loading, adaptation, and logging; shared trainer utilities live in `trainers/tta_abstract_trainer.py`.
- `configs/` defines dataset pairings and hyperparameter defaults; adjust these instead of hard-coding values in trainers.
- `models/`, `optim/`, and `utils/` group reusable building blocks; prefer extending these rather than duplicating code inside algorithms.
- `results/tta_experiments_logs/` is filled at runtime with metrics and checkpoints; keep large artifacts out of version control.

## Build, Test, and Development Commands
- Create a virtual environment and install dependencies with `python -m venv .venv` followed by `.venv\Scripts\activate` and `python -m pip install -r require.txt`.
- Run a full adaptation pass using `python trainers/tta_trainer.py --da_method DuSafe --dataset EEG --data-path <dataset_root> --num_runs 1`; swap flags to target other scenarios.
- For quicker smoke checks, set `--da_method NoAdap` to validate data ingestion without adaptation updates.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, snake_case for functions and modules, PascalCase for classes, and descriptive argument names mirroring CLI flags.
- Keep configuration in Python dicts inside `configs/`; document new parameters inline and surface them through `argparse` in `tta_trainer.py`.
- Add concise English comments where behavior is non-obvious (existing Chinese notes are acceptable but provide an English summary nearby).

## Testing Guidelines
- Use `pytest` with files named `test_<module>.py`; keep protocol-critical coverage in `tests/test_tta_protocol.py`.
- Until automated tests land, capture before/after metrics from `results/tta_experiments_logs/*summary_f1_scores.txt` and share the command used to reproduce them.
- When touching dataloaders or configs, run the protocol tests and, when data are available, one DuSafe scenario with its metrics table.

## Commit & Pull Request Guidelines
- Keep commit subjects short and imperative as in the existing history (`add timesnet`, `Update README.md`); group related changes logically.
- Reference relevant configs or scripts in the body and note any required dataset layout changes.
- Pull requests should state the motivation, the command used for validation, and paste key metrics; include screenshots for tensorboard or tables when useful.

## Configuration Tips
- Default paths assume datasets live outside the repo (e.g., `E:\Dataset`); never commit raw data, and use environment variables or `.env` files for machine-specific overrides.
- Tune hyperparameters in `configs/tta_hparams_new.py` and new model definitions in `models/` so collaborators can reproduce runs without manual edits elsewhere.
