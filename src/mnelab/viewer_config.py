# © MNELAB developers
#
# License: BSD (3-clause)

"""Validated static configuration for scientific plots and Plot Traces."""

from pathlib import Path

import yaml


class ViewerConfigError(ValueError):
    """The bundled viewer configuration is missing or invalid."""


_RULES = {
    "psd": {
        "main_window_n_fft": (int, lambda value: value >= 2),
        "trace_window_n_fft": (int, lambda value: value >= 2),
        "trace_window_minimum_bins": (int, lambda value: value >= 2),
        "trace_window_maximum_bins": (int, lambda value: value >= 2),
        "lane_step": ((int, float), lambda value: value > 0),
        "lane_half_height": ((int, float), lambda value: 0 < value <= 1),
    },
    "trace_layout": {
        "channel_list_width": (int, lambda value: value > 0),
        "channel_label_width": (int, lambda value: value > 0),
        "panel_body_spacing": (int, lambda value: value >= 0),
        "channel_lane_height": (int, lambda value: value > 0),
        "minimum_plot_height": (int, lambda value: value > 0),
        "maximum_plot_height": (int, lambda value: value > 0),
        "fit_half_lane_fraction": ((int, float), lambda value: 0 < value <= 0.5),
    },
    "trace_amplitude": {
        "step_factor": ((int, float), lambda value: value > 1),
        "minimum": ((int, float), lambda value: value > 0),
        "maximum": ((int, float), lambda value: value > 0),
    },
    "trace_palette": {
        "default_color": (str, lambda value: _valid_color(value)),
        "automatic_hue_start": ((int, float), lambda value: 0 <= value <= 1),
        "automatic_hue_step": ((int, float), lambda value: 0 <= value <= 1),
        "automatic_saturation": ((int, float), lambda value: 0 <= value <= 1),
        "automatic_value": ((int, float), lambda value: 0 <= value <= 1),
    },
    "annotations": {
        "marker_row_limit": (int, lambda value: value > 0),
        "minimum_font_size": (int, lambda value: value > 0),
        "maximum_font_size": (int, lambda value: value > 0),
    },
    "activation_map": {
        "maximum_intermediate_elements": (int, lambda value: value > 0),
        "missing_data_color": (str, lambda value: _valid_color(value)),
        "minimum_axis_width": (int, lambda value: value > 0),
        "maximum_axis_width": (int, lambda value: value > 0),
    },
}


def _valid_color(value):
    """Return whether `value` is a six-digit hexadecimal RGB color."""
    return (
        len(value) == 7
        and value.startswith("#")
        and all(character in "0123456789abcdefABCDEF" for character in value[1:])
    )


def load_viewer_config(path=None):
    """Load and validate the viewer YAML configuration."""
    path = Path(path) if path is not None else Path(__file__).with_suffix(".yaml")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ViewerConfigError(f"Cannot load viewer configuration: {error}") from error
    if not isinstance(config, dict):
        raise ViewerConfigError("Viewer configuration must be a YAML mapping")

    for section, rules in _RULES.items():
        values = config.get(section)
        if not isinstance(values, dict):
            raise ViewerConfigError(f"Missing configuration section: {section}")
        for key, (expected_type, validator) in rules.items():
            value = values.get(key)
            if isinstance(value, bool) or not isinstance(value, expected_type):
                raise ViewerConfigError(f"Invalid configuration value: {section}.{key}")
            if not validator(value):
                raise ViewerConfigError(f"Invalid configuration value: {section}.{key}")

    if (
        config["trace_layout"]["minimum_plot_height"]
        > config["trace_layout"]["maximum_plot_height"]
    ):
        raise ViewerConfigError("Minimum plot height exceeds maximum plot height")
    default_bins = config["psd"]["trace_window_n_fft"] // 2 + 1
    if not (
        config["psd"]["trace_window_minimum_bins"]
        <= default_bins
        <= config["psd"]["trace_window_maximum_bins"]
    ):
        raise ViewerConfigError("Default trace PSD bins fall outside their limits")
    if config["trace_amplitude"]["minimum"] >= config["trace_amplitude"]["maximum"]:
        raise ViewerConfigError("Minimum trace amplitude must be below its maximum")
    if (
        config["annotations"]["minimum_font_size"]
        > config["annotations"]["maximum_font_size"]
    ):
        raise ViewerConfigError("Minimum annotation font size exceeds its maximum")
    if (
        config["activation_map"]["minimum_axis_width"]
        > config["activation_map"]["maximum_axis_width"]
    ):
        raise ViewerConfigError("Minimum activation axis width exceeds its maximum")
    return config


VIEWER_CONFIG = load_viewer_config()
