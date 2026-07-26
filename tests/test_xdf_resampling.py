# © MNELAB developers
#
# License: BSD (3-clause)

import numpy as np

from mnelab.xdf import _resample_xdf_streams


def _stream(values, sampling_rate):
    """Return the subset of a PyXDF stream used by the resampler."""
    timestamps = np.arange(len(values), dtype=float) / sampling_rate
    return {
        "info": {
            "channel_count": [str(values.shape[1])],
            "effective_srate": sampling_rate,
        },
        "time_stamps": timestamps,
        "time_series": values,
    }


def test_xdf_resampling_does_not_spread_nans_across_channels():
    """Explicit missing samples remain local instead of silencing each stream."""
    fast_times = np.arange(400) / 100.0
    fast = np.column_stack(
        (np.sin(2 * np.pi * 4 * fast_times), np.cos(2 * np.pi * 7 * fast_times))
    )
    fast[80:85] = np.nan

    slow_times = np.arange(240) / 60.0
    slow = np.sin(2 * np.pi * 3 * slow_times)[:, np.newaxis]
    slow[120:124] = np.nan

    data, first_time = _resample_xdf_streams(
        {1: _stream(fast, 100.0), 2: _stream(slow, 60.0)},
        [1, 2],
        fs_new=80.0,
    )

    assert first_time == 0.0
    assert data.shape[1] == 3
    assert np.isfinite(data).any(axis=0).all()
    assert np.isnan(data).any(axis=0).all()


def test_xdf_resampling_retains_an_entirely_missing_channel():
    """A genuinely empty channel remains NaN while its valid peer is resampled."""
    times = np.arange(200) / 100.0
    values = np.column_stack((np.sin(2 * np.pi * times), np.full(200, np.nan)))

    data, _ = _resample_xdf_streams(
        {1: _stream(values, 100.0)},
        [1],
        fs_new=80.0,
        use_interpolation=True,
    )

    assert np.isfinite(data[:, 0]).any()
    assert np.isnan(data[:, 1]).all()
