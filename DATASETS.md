# Dataset archive

Datasets and checkpoints are intentionally excluded from Git history. The
[dataset release](https://github.com/Zonglin-He/model/releases/tag/datasets-20260825)
asset `DuSafe_processed_datasets_20260825.zip` contains only the processed
tensors needed by the loaders:

```text
data/Dataset/
  EEG/train_*.pt, test_*.pt
  HAR/train_*.pt, test_*.pt
  FD/train_*.pt, test_*.pt
  HHAR/train_*.pt, test_*.pt, conversion manifests
```

The archive excludes raw HHAR CSV files, duplicate extraction directories,
source checkpoints, results, caches, and model outputs. It has a protective
top-level directory. From the repository root, download, verify, and install
it with:

```powershell
$url = "https://github.com/Zonglin-He/model/releases/download/datasets-20260825/DuSafe_processed_datasets_20260825.zip"
curl.exe -L -C - -o DuSafe_processed_datasets_20260825.zip $url
(Get-FileHash .\DuSafe_processed_datasets_20260825.zip -Algorithm SHA256).Hash
Expand-Archive .\DuSafe_processed_datasets_20260825.zip -DestinationPath .\_dataset_extract
New-Item -ItemType Directory -Path .\data -Force
Move-Item `
  .\_dataset_extract\DuSafe_processed_datasets_20260825\data\Dataset `
  .\data\Dataset
```

The expected archive SHA256 is
`086719267ec2d2cfa0cdaf542a59844c383eab751a8dcd663d6bd8493394a493`.
The archive also contains `DATASET_MANIFEST.csv`, with a SHA256 and byte size
for every payload file.

## Sources and licenses

| Directory | Source | License |
|---|---|---|
| `EEG` | Subject-wise Sleep Stage Data, DOI `10.21979/N9/UD1IM9` | CC BY-NC 4.0 |
| `HAR` | UCI HAR Dataset Processed, DOI `10.21979/N9/0SYHTZ` | CC BY-NC 4.0 |
| `FD` | Machine Fault Diagnosis, DOI `10.21979/N9/PU85XN` | CC BY-NC 4.0 |
| `HHAR` | UCI Heterogeneity Activity Recognition, DOI `10.24432/C5689X` | CC BY 4.0 |

The repository's MIT license applies to the source code, not to these dataset
files. Dataset users must retain attribution and comply with the corresponding
dataset license; in particular, the processed EEG, HAR, and FD distributions
are limited to non-commercial use under CC BY-NC 4.0.
