# Robust TTA baseline source audit

Audit date: 2026-08-18 (Asia/Shanghai)

This file records the upstream sources used for the reviewer-baseline audit.
Only author-maintained repositories or repositories named by the paper itself
were accepted.  The four robust baselines requested for this audit, plus the
five existing baseline sources that were missing from the current checkout,
were cloned with `git clone --depth 1`; each clone is unmodified and keeps its own `.git`
directory.  These repositories are not imported into `algorithms/` and are not
drop-in implementations for the production fixed-source time-series trainer.

## Official robust baseline clones

| Method | Author/paper source | Local path | Clone state and commit | License | Official TTA defaults | Current protocol interface difference |
|---|---|---|---|---|---|---|
| CoTTA | [qinenergy/cotta](https://github.com/qinenergy/cotta); [paper](https://arxiv.org/abs/2203.13591) (paper also links `https://qin.ee/cotta`) | `third_party/official_baselines/CoTTA` | Shallow clone (`main`), `c212a204b32be4005092e4323105a24a29ad2952` | MIT in `LICENSE`; the same file also reproduces the TENT MIT notice. | Yes. CIFAR configs: `cifar/cfgs/cifar10/cotta.yaml` and `cifar/cfgs/cifar100/cotta.yaml` (Adam, one step, LR `1e-3`, MT `0.999`, restore `0.01`, AP `0.92`/`0.72`, batch `200`). ImageNet uses `imagenet/cfgs/10orders/cotta/cotta*.yaml` through `imagenet/run.sh`. | Official loop is image-only (CIFAR/ImageNet/Cityscapes), expects RobustBench/torchvision image models and its YAML/`imagenetc.py` entry point. It does not consume this repository's EEG/HAR/FD loaders, TimesNet-style model API, `--data-path`, source-seed checkpoint cache, or `BaseTestTimeAlgorithm` call shape. |
| SoTTA | [taeckyung/SoTTA](https://github.com/taeckyung/SoTTA); [paper](https://arxiv.org/abs/2310.10074) | `third_party/official_baselines/SoTTA` | Shallow clone (`main`), `09d568f467cd0343d1af2d751fb7186d839817ae` | MIT in `LICENSE` | Yes, but script-defined rather than a single YAML: `tta.sh` sets online mode, `update_every_x=64`, `memory_size=64`, other-baseline LR `0.001`, and SoTTA HUS/high-threshold variants; `conf.py`/`sotta.yml` provide environment and dataset defaults. | Official code expects image datasets and its `main.py` noisy-stream protocol, source checkpoints under `log/`, and method-specific memory/BN flags. It has no adapter for the production `EEG`/`HAR`/`FD` loaders, fixed-source cache semantics, or current trainer's batch/diagnostic interface. |
| RoTTA | [BIT-DA/RoTTA](https://github.com/BIT-DA/RoTTA); [paper](https://arxiv.org/abs/2303.13899) (paper names the same repository) | `third_party/official_baselines/RoTTA` | Shallow clone (`main`), `67e34c900cdd355fc07e55edd4c577ea7b8ebcc9` | MIT in `LICENSE` | Yes. `configs/adapter/rotta.yaml` defaults to batch `64`, one Adam step, LR `1e-3`, memory `64`, update frequency `64`, `nu=0.001`, `alpha=0.05`, `lambda_t=lambda_u=1`; `configs/dataset/cifar10.yaml`/`cifar100.yaml` and `ptta.py` are the official entry point. | Official `ptta.py` consumes CIFAR-C image batches and package-specific config dictionaries (`image`, `label`, `domain`), with temporal sampling and a memory bank. It is not wired to `trainers/tta_trainer.py`, current time-series backbones, source normalization/checkpoint metadata, or this repository's fixed-source scenario loop. |
| COME | [BlueWhaleLab/COME](https://github.com/BlueWhaleLab/COME); [paper](https://arxiv.org/abs/2410.10894) | `third_party/official_baselines/COME` | Shallow clone (`main`), `409a19b71f62c765b1a5be62347a9455524ec176` | No `LICENSE` file; GitHub API reports no declared license. Treat reuse as license-unresolved until authors clarify. | Yes, script-defined in `start.sh`/`start-open.sh`: ImageNet-C level `5` (normal) or level `3` (open-world), ViT-Base, batch `64`, one step, seed `2024`, and `no_adapt`/Tent/EATA/SAR plus COME variants. | Official code is an ImageNet/OpenOOD image experiment around `main.py`, timm ResNet/ViT models, and `Tent_COME`/`EATA_COME`/`SAR_COME`. It does not accept the current time-series loaders, fixed-source source-seed/cache protocol, or the production trainer's adapter registry and logging contract. |

The commit hashes above were read from each local clone with
`git -C <path> rev-parse HEAD` after cloning.  No long experiment was run.

## Existing TENT/EATA/SAR/NOTE/ACCUPOfficial audit

The current production `HEAD` contains only DuSafe in
`algorithms/get_tta_class.py`; the baseline ports were removed by
`7345444b5abffc5de724312d815fd652f2c1aa7f` (`Prune repository to DuSafe
core`).  The repository README states that they remain recoverable from local
commit `4de8bad86c1b2a93be2ae313625617003e2bfc0c`.  The table below therefore
distinguishes code present in the current checkout from code recoverable in
local Git history and from the newly retained official clones.  The official
clones are source snapshots; the historical ports are not silently
reintroduced into production.

| Method | Official source | Local official clone and commit | Official license | Official default/config evidence | Local current checkout | Local-history/source-provenance result |
|---|---|---|---|---|---|---|
| TENT | [DequanWang/tent](https://github.com/DequanWang/tent); [paper](https://arxiv.org/abs/2006.10726) | `third_party/official_baselines/TENT`, shallow `master`, `e9e926a668d85244c66a6d5c006efbd2b82e83e8` | MIT (`LICENSE`) | Yes: `cfgs/tent.yaml` plus `cifar10c.py`/`conf.py` defaults. | Absent. | A local `Tent` port existed as `algorithms/tent_tta.py` in `4de8bad8`; the file has no upstream URL, so the official clone now provides the auditable source. |
| EATA | [mr-eggplant/EATA](https://github.com/mr-eggplant/EATA); [paper](https://arxiv.org/abs/2204.02610) | `third_party/official_baselines/EATA`, shallow `main`, `f739b3668cc7617e9b9f1979c1a358497a3472c3` | MIT (`LICENSE`) | No standalone YAML was found in the official tree; defaults are in `main.py` (batch `64`, Fisher size `2000`, `e_margin=0.4 log(1000)`, `d_margin=0.05`). | Absent. | A local `EATA` port existed as `algorithms/eata_accup.py` in `4de8bad8`; no upstream URL is embedded in that file, so the official clone now provides the auditable source. |
| SAR | [mr-eggplant/SAR](https://github.com/mr-eggplant/SAR); [paper](https://arxiv.org/abs/2302.12400) | `third_party/official_baselines/SAR`, shallow `main`, `20f6e24b17525f34503510afccedc0629b67b7c4` | BSD-3-Clause (`LICENSE`) | No standalone YAML was found in the official tree; defaults are in `main.py`/the included ImageNet-C scripts. | Absent. | A local `SAR` port existed as `algorithms/sar_tta.py` (plus instrumentation) in `4de8bad8`; no upstream URL is embedded in that file, so the official clone now provides the auditable source. |
| NOTE | [TaesikGong/NOTE](https://github.com/TaesikGong/NOTE); [paper](https://arxiv.org/abs/2208.05117) | `third_party/official_baselines/NOTE`, shallow `main`, `a714a2a2a9406903ba787b0bc240a95dd0342de5` | MIT (`LICENSE`) | Yes: `note.yml` and `tta.sh`; CLI defaults include online update controls and PBRS memory settings. | Absent. | No `note_tta.py`, NOTE adapter, or equivalent local implementation appears in `git log --all`; the official clone is the only local source snapshot. This was not locally reproduced in the production trainer. |
| ACCUPOfficial | [Tokenmw/ACCUP-main](https://github.com/Tokenmw/ACCUP-main); [paper](https://arxiv.org/abs/2501.01472) | `third_party/official_baselines/ACCUPOfficial`, shallow `main`, `920c43c092c6aa96a7950d2e3c0df5c2e4216f99` | MIT (`LICENSE`) | Yes: `configs/` (including `configs/tta_hparams_new.py`); official defaults use batch `32`, one step, Adam, with dataset-specific ACCUP `filter_K`, `tau`, and temperature. | Absent. | A local `ACCUPOfficial` port existed as `algorithms/accup_official.py` in `4de8bad8`, and explicitly records `https://github.com/Tokenmw/ACCUP-main/blob/main/algorithms/accup.py`; the official clone preserves the cited source and the historical port remains non-production. |

The historical ports above were adapted to the old shared trainer and are not
the official repositories themselves.  They must not be described as current
production baselines unless restored deliberately from the historical commit
and re-audited against the fixed-source protocol.

## Historical trainer rerun status

The existing reviewer-rerun artifact
`results/tta_experiments_logs/reviewer_rerun/paired_significance_final/` records
the ten source seeds `101, 202, 303, 404, 505, 606, 707, 808, 909, 1010`.
Its `per_source_seed_results.csv` contains 1,500 rows: ten methods, three
datasets (EEG, FD, HAR), fifteen scenarios, and ten source seeds.  Thus the
following methods have ten-seed paired results in that artifact: `Tent`,
`EATA`, `SAR`, `ACCUPOfficial`, `CoTTA`, `SoTTA`, `RoTTA`, and `COME` (as well
as `NoAdap` and `DuSafe`).  These results are historical/shared-trainer
artifacts; they do not make the adapters current production code.  In the
current checkout, all eight baseline adapters remain absent and are recoverable
only from `4de8bad8` as described above.

`NOTE` was not rerun in the old shared trainer: it has no row in the paired
results artifact and no local historical adapter in Git history.  It is
represented locally only by the official shallow clone listed above.

## Clone and parent-repository hygiene

- All nine official clones (`CoTTA`, `SoTTA`, `RoTTA`, `COME`, `TENT`,
  `EATA`, `SAR`, `NOTE`, and `ACCUPOfficial`) each report
  `git rev-parse --is-shallow-repository` = `true`, one commit in the shallow history,
  and a nested `.git` directory.
- Worktree sizes excluding each nested `.git` are approximately: CoTTA
  908,041 bytes, SoTTA 2,795,397 bytes, RoTTA 624,441 bytes, COME 137,404
  bytes, TENT 26,982 bytes, EATA 615,733 bytes, SAR 17,952,611 bytes, NOTE
  160,374 bytes, and ACCUPOfficial 25,346,552 bytes.  EATA includes an
  upstream figure PNG; SAR includes upstream dataset `.npy` files; and
  ACCUPOfficial includes upstream result/checkpoint artifacts.  These files
  remain inside ignored clones and are not parent-repository additions.
- Parent `.gitignore` now ignores `third_party/official_baselines/*/`, which
  keeps all nine clone trees and their nested Git metadata out of the parent
  index while leaving this root audit file trackable.
- No production algorithm, trainer, model, dataloader, or experiment result
  was modified, and no long TTA experiment was run.
