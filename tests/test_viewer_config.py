# © MNELAB developers
#
# License: BSD (3-clause)

import pytest
import yaml

from mnelab.viewer_config import VIEWER_CONFIG, ViewerConfigError, load_viewer_config


def test_viewer_config_contains_valid_controllable_defaults():
    """The bundled YAML exposes the supported viewer configuration groups."""
    assert VIEWER_CONFIG["psd"]["trace_window_n_fft"] == 16384
    assert VIEWER_CONFIG["psd"]["trace_window_minimum_bins"] <= 8193
    assert VIEWER_CONFIG["psd"]["trace_window_maximum_bins"] >= 8193
    assert (
        VIEWER_CONFIG["trace_amplitude"]["minimum"]
        < VIEWER_CONFIG["trace_amplitude"]["maximum"]
    )
    assert (
        VIEWER_CONFIG["trace_layout"]["minimum_plot_height"]
        <= VIEWER_CONFIG["trace_layout"]["maximum_plot_height"]
    )


def test_viewer_config_rejects_invalid_values(tmp_path):
    """Invalid tunable values fail clearly instead of breaking the viewer later."""
    config = {section: dict(values) for section, values in VIEWER_CONFIG.items()}
    config["psd"]["trace_window_n_fft"] = 1
    path = tmp_path / "viewer_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ViewerConfigError, match="psd.trace_window_n_fft"):
        load_viewer_config(path)
