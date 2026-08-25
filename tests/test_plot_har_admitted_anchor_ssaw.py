from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.plot_har_admitted_anchor_ssaw import (
    _caption,
    _first_eligible_index,
    _js_divergence,
    _normalized_multichannel_psd_with_dc,
    _normalized_periodogram,
    _plot,
    _write_trace,
)


def test_first_eligible_index_uses_all_declared_conditions():
    admission = torch.tensor([True, True, True, True])
    consistency = torch.tensor([False, True, True, True])
    active = torch.tensor([True, False, True, True])
    selected_valid = torch.tensor([True, True, True, True])
    raw_margin = torch.tensor([2.0, 2.0, 2.0, 2.0])
    selected_margin = torch.tensor([1.0, 1.0, 2.1, 0.6])
    assert (
        _first_eligible_index(
            admission,
            consistency,
            active,
            selected_valid,
            raw_margin,
            selected_margin,
        )
        == 3
    )


def test_normalized_periodogram_and_js_are_well_formed():
    time = np.arange(128) / 50.0
    first = np.sin(2.0 * np.pi * 3.0 * time)
    second = first * (1.0 + 0.1 * np.sin(2.0 * np.pi * 0.5 * time))
    frequency, first_psd = _normalized_periodogram(first, 50.0)
    second_frequency, second_psd = _normalized_periodogram(second, 50.0)
    assert np.array_equal(frequency, second_frequency)
    assert np.isclose(first_psd.sum(), 1.0)
    assert np.isclose(second_psd.sum(), 1.0)
    assert _js_divergence(first_psd, first_psd) < 1e-12
    assert np.isclose(
        _js_divergence(first_psd, second_psd),
        _js_divergence(second_psd, first_psd),
    )
    multi_frequency, multi_psd = _normalized_multichannel_psd_with_dc(
        np.stack((first, second)), 50.0
    )
    assert np.array_equal(frequency, multi_frequency)
    assert np.isclose(multi_psd.sum(), 1.0)


def test_plot_trace_and_caption_outputs(tmp_path: Path):
    time = np.arange(128) / 50.0
    gain = np.exp(0.1 * np.sin(2.0 * np.pi * time / time[-1]))
    raw_channel = np.sin(2.0 * np.pi * 3.0 * time)
    selected_channel = raw_channel * gain
    frequency, raw_psd = _normalized_periodogram(raw_channel, 50.0)
    _, selected_psd = _normalized_periodogram(selected_channel, 50.0)
    record = {
        "scenario": "12->16",
        "source_seed": 1,
        "stream_seed": 42,
        "channel_index": 0,
        "sampling_hz": 50.0,
        "raw": raw_channel[None, :],
        "selected": selected_channel[None, :],
        "gain": gain,
        "direction_curve": np.sin(2.0 * np.pi * time / time[-1]),
        "frequency_hz": frequency,
        "raw_psd": raw_psd,
        "selected_psd": selected_psd,
        "psd_panel_label": "all-channel mean",
        "selected_direction": 2,
        "selected_sign": 1.0,
        "selected_radius": 0.5,
        "runtime_hparams": {"spline_log_strength": 0.2},
        "normalized_psd_js_nats": _js_divergence(raw_psd, selected_psd),
        "gain_min": float(gain.min()),
        "gain_max": float(gain.max()),
        "raw_margin": 2.0,
        "selected_margin": 1.0,
    }
    png, pdf = _plot(record, tmp_path)
    waveform_csv, psd_csv = _write_trace(record, tmp_path)
    assert png.is_file() and png.stat().st_size > 0
    assert pdf.is_file() and pdf.stat().st_size > 0
    assert waveform_csv.is_file() and psd_csv.is_file()
    caption = _caption(record)
    assert "target labels and target F1 are not used" in caption
    assert "2.000" in caption and "1.000" in caption
    # Keep this serialization check here: the trace metadata is intended to be
    # reusable without custom tensor encoders.
    json.dumps({"caption": caption})
