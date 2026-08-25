"""Dataset and backbone dimensions used by DuSafe experiments.

The HHAR entry is deliberately a protocol description, not a claim that its
hyperparameters have been tuned.  Raw HHAR windows are expected to be kept
unstandardized; the trainer fits the scaler on the source ``train`` split and
reuses those statistics for source ``test`` and every target split.
"""


DATASET_NAMES = ("EEG", "HAR", "FD", "HHAR")


def get_dataset_class(dataset_name):
    dataset_name = str(dataset_name).strip().upper()
    try:
        return globals()[dataset_name]
    except KeyError as exc:
        raise NotImplementedError(f"Dataset not found: {dataset_name}") from exc


class _BaseConfig:
    shuffle = True
    drop_last = False
    normalize = True
    times_dropout = 0.1
    times_ffn_expansion = 2
    # Complete three-axis groups may receive SSAW's physical orientation view.
    # Scalar-sensor datasets retain the existing gain-only behavior.
    ssaw_orientation = False


class EEG(_BaseConfig):
    num_classes = 5
    class_names = ["W", "N1", "N2", "N3", "REM"]
    sequence_len = 3000
    scenarios = [("0", "11"), ("12", "5"), ("7", "18"), ("16", "1"), ("9", "14")]
    input_channels = 1
    kernel_size = 25
    stride = 6
    dropout = 0.2
    mid_channels = 16
    final_out_channels = 8
    features_len = 65
    times_hidden_channels = 128
    times_num_layers = 3
    times_patch_lens = [16, 32, 64]


class HAR(_BaseConfig):
    num_classes = 6
    class_names = ["walk", "upstairs", "downstairs", "sit", "stand", "lie"]
    sequence_len = 128
    scenarios = [("2", "11"), ("6", "23"), ("7", "13"), ("9", "18"), ("12", "16")]
    drop_last = True
    input_channels = 9
    kernel_size = 5
    stride = 1
    dropout = 0.5
    mid_channels = 64
    final_out_channels = 128
    features_len = 1
    times_hidden_channels = 128
    times_num_layers = 2
    times_patch_lens = [4, 8, 16]
    ssaw_orientation = True


class HHAR(_BaseConfig):
    """Notebook-audited HHAR dimensions and transfer flows.

    Domains are users (``0``-``8``), and each sample is a three-axis phone
    accelerometer window of length 128.  ``scenarios`` is kept in the audited
    AdaTime order, including the two flows whose source domain is ``0``.
    Conversion keeps raw windows for the repository's fixed-source runtime
    normalization variant rather than claiming notebook sample equivalence.
    """

    num_classes = 6
    class_names = [
        "bike",
        "sit",
        "stairsdown",
        "stairsup",
        "stand",
        "walk",
    ]
    sequence_len = 128
    scenarios = [
        ("0", "6"),
        ("1", "6"),
        ("2", "7"),
        ("3", "8"),
        ("4", "5"),
        ("5", "0"),
        ("6", "1"),
        ("7", "4"),
        ("8", "3"),
        ("0", "2"),
    ]
    # HHAR source files contain raw [N, 3, 128] windows.  The runtime loader
    # fits source-train statistics once and applies them to source-test and
    # target files; target data must never be used to fit this scaler.
    normalization_reference = "source"
    scaler_fit_split = "train"
    scaler_target_fit_forbidden = True
    drop_last = True
    input_channels = 3
    kernel_size = 5
    stride = 1
    dropout = 0.5
    mid_channels = 64
    final_out_channels = 128
    features_len = 1
    times_hidden_channels = 128
    times_num_layers = 2
    times_patch_lens = [4, 8, 16]
    ssaw_orientation = True


class FD(_BaseConfig):
    num_classes = 3
    class_names = ["Healthy", "D1", "D2"]
    sequence_len = 5120
    scenarios = [("0", "1"), ("1", "2"), ("3", "1"), ("1", "0"), ("2", "3")]
    input_channels = 1
    kernel_size = 32
    stride = 6
    dropout = 0.5
    mid_channels = 64
    final_out_channels = 128
    features_len = 1
    times_hidden_channels = 256
    times_num_layers = 3
    times_patch_lens = [32, 64, 128]


def get_dataset_names():
    """Return the dataset registry in stable CLI/configuration order."""

    return DATASET_NAMES


def scenario_pairs(dataset_name):
    """Return normalized ``(source, target)`` pairs for one dataset."""

    config = get_dataset_class(dataset_name)
    return [(str(source), str(target)) for source, target in config.scenarios]


def validate_scenario(dataset_name, source, target):
    """Validate one transfer flow against the registered protocol.

    Scenario selection is intentionally strict: accepting a syntactically
    valid but unregistered source/target pair would silently change the
    benchmark protocol.
    """

    pair = (str(source), str(target))
    expected = scenario_pairs(dataset_name)
    if pair not in expected:
        formatted = ", ".join(f"{src}->{trg}" for src, trg in expected)
        raise ValueError(
            f"Invalid {str(dataset_name).upper()} scenario {pair[0]}->{pair[1]}; "
            f"expected one of: {formatted}"
        )
    return pair


def supports_ssaw_orientation(dataset_name):
    """Return whether the dataset declares complete three-axis groups."""

    return bool(getattr(get_dataset_class(dataset_name), "ssaw_orientation", False))


__all__ = [
    "DATASET_NAMES",
    "EEG",
    "HAR",
    "HHAR",
    "FD",
    "get_dataset_class",
    "get_dataset_names",
    "scenario_pairs",
    "supports_ssaw_orientation",
    "validate_scenario",
]
