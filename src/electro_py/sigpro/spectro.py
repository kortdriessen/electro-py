from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import spectrogram

bands = {}
bands["delta"] = (0.5, 4.0)
bands["delta1"] = (0.5, 2.0)
bands["delta2"] = (2.0, 3.5)
bands["theta"] = (4.0, 8.0)
bands["alpha"] = (8.0, 13.0)
bands["sigma"] = (11.0, 16.0)
bands["beta"] = (13.0, 30.0)
bands["low_gamma"] = (30.0, 80)
bands["high_gamma"] = (80, 200)


# This function is taken directly from neurodsp.spectral.utils.
# We cannot use the neurodsp package, because a critical IBL library shadows the name.
def all_arrays_equal(iterator):
    """Check if all arrays in the iterator are equal."""
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(np.array_equal(first, rest) for rest in iterator)
    except StopIteration:
        return True


def trim_spectrogram(freqs, times, spg, f_range=None, t_range=None):
    """Extract a frequency or time range of interest from a spectrogram.
    Parameters
    ----------
    freqs : 1d array
        Frequency values for the spectrogram.
    times : 1d array
        Time values for the spectrogram.
    spg : 2d array
        Spectrogram, or time frequency representation of a signal.
        Formatted as [n_freqs, n_time_windows].
    f_range : list of [float, float]
        Frequency range to restrict to, as [f_low, f_high].
    t_range : list of [float, float]
        Time range to restrict to, as [t_low, t_high].
    Returns
    -------
    freqs_ext : 1d array
        Extracted frequency values for the power spectrum.
    times_ext : 1d array
        Extracted segment time values
    spg_ext : 2d array
        Extracted spectrogram values.
    Notes
    -----
    This function extracts frequency ranges >= f_low and <= f_high,
    and time ranges >= t_low and <= t_high. It does not round to below
    or above f_low and f_high, or t_low and t_high, respectively.
    Examples
    --------
    Trim the spectrogram of a simulated time series:
    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.timefrequency import compute_wavelet_transform
    >>> from neurodsp.utils.data import create_times, create_freqs
    >>> fs = 500
    >>> n_seconds = 10
    >>> times = create_times(n_seconds, fs)
    >>> sig = sim_combined(n_seconds, fs,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> freqs = create_freqs(1, 15)
    >>> mwt = compute_wavelet_transform(sig, fs, freqs)
    >>> spg = abs(mwt)**2
    >>> freqs_ext, times_ext, spg_ext = trim_spectrogram(freqs, times, spg,
    ...                                                  f_range=[8, 12], t_range=[0, 5])
    """

    # Initialize spg_ext, to define for case in which neither f_range nor t_range is defined
    spg_ext = spg

    # Restrict frequency range of the spectrogram
    if f_range is not None:
        f_mask = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
        freqs_ext = freqs[f_mask]
        spg_ext = spg_ext[f_mask, :]
    else:
        freqs_ext = freqs

    # Restrict time range of the spectrogram
    if t_range is not None:
        times_mask = np.logical_and(times >= t_range[0], times <= t_range[1])
        times_ext = times[times_mask]
        spg_ext = spg_ext[:, times_mask]
    else:
        times_ext = times

    return freqs_ext, times_ext, spg_ext


def check_spg_settings(fs, window, nperseg, noverlap):
    """Check settings used for calculating spectrogram.
    Parameters
    ----------
    fs : float
        Sampling rate, in Hz.
    window : str or tuple or array_like
        Desired window to use. See scipy.signal.get_window for a list of available windows.
        If array_like, the array will be used as the window and its length must be nperseg.
    nperseg : int or None
        Length of each segment, in number of samples.
    noverlap : int or None
        Number of points to overlap between segments.
    Returns
    -------
    nperseg : int
        Length of each segment, in number of samples.
    noverlap : int
        Number of points to overlap between segments.
    """

    # Set the nperseg, if not provided
    if nperseg is None:
        # If the window is a string or tuple, defaults to 1 second of data
        if isinstance(window, (str, tuple)):
            nperseg = int(fs)
        # If the window is an array, defaults to window length
        else:
            nperseg = len(window)
    else:
        nperseg = int(nperseg)

    if noverlap is not None:
        noverlap = int(noverlap)

    return nperseg, noverlap


def single_spectrogram_welch(
    sig,
    fs,
    window="hann",
    detrend="constant",
    nperseg=None,
    noverlap=None,
    f_range=None,
    t_range=None,
):
    """Compute spectrogram using Welch's method.

    Parameters
    -----------
    sig : (n_samples,)
        Time series.
    fs : float
        Sampling rate, in Hz.
    window : str or tuple or array_like, optional, default: 'hann'
        Desired window to use. See scipy.signal.get_window for a list of available windows.
        If array_like, the array will be used as the window and its length must be nperseg.
    detrend: str or function or False, optional
        Specifies how to detrend each segment. If detrend is a string, it is passed as the
        type argument to the detrend function. If it is a function, it takes a segment and
        returns a detrended segment. If detrend is False, no detrending is done.
        Defaults to ‘constant’, which is mean subtraction.
    nperseg : int, optional
        Length of each segment, in number of samples.
        If None, and window is str or tuple, is set to 1 second of data.
        If None, and window is array_like, is set to the length of the window.
    noverlap : int, optional
        Number of points to overlap between segments.
        If None, noverlap = nperseg // 8.
    f_range: list of [float, float]
        Frequency range to restrict to, as [f_low, f_high].
    t_range: list of [float, float]
        Time range to restrict to, as [t_low, t_high].

    Returns
    -------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    spg_times: 1d array
        Segment times
    spg : (n_freqs, n_spg_times)
        Spectrogram of `sig`.
    """

    # Calculate the short time Fourier transform with signal.spectrogram
    nperseg, noverlap = check_spg_settings(fs, window, nperseg, noverlap)
    freqs, spg_times, spg = spectrogram(
        sig, fs, window, nperseg, noverlap, detrend=detrend
    )
    freqs, spg_times, spg = trim_spectrogram(freqs, spg_times, spg, f_range, t_range)

    return freqs, spg_times, spg


def parallel_spectrogram_welch(sig, fs, **kwargs):
    """Apply `_compute_spectrogram_welch` to each channel in parallel.

    Should also work fine for a single channel, as long as sig is 2D.
    But in that case, maybe you want to save the overhead and use
    single_spectrogram_welch directly...

    Parameters
    ----------
    sig: (n_samples, n_chans)
        The multichannel timeseries.
    fs: float
        The sampling frequency of the data.
    **kwargs: optional
        Keyword arguments passed to `_compute_spectrogram_welch`.

    Returns:
    --------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    spg_times: 1d array
        Segment times
    spg : (n_freqs, n_spg_times, n_chans)
        Spectrogram of `sig`.
    """

    worker = partial(single_spectrogram_welch, fs=fs, **kwargs)
    jobs = [x for x in sig.T]

    n_chans = sig.shape[1]
    with Pool(n_chans) as p:
        freqs, spg_times, spg = zip(*p.map(worker, jobs), strict=False)

    assert all_arrays_equal(
        freqs
    ), "Spectrogram frequecies must match for all channels."
    assert all_arrays_equal(spg_times), "Segment times must match for all channels."

    freqs = freqs[0]
    spg_times = spg_times[0]
    if len(spg) > 1:
        spg = np.dstack(spg)
    else:
        spg = np.expand_dims(spg, axis=-1)

    return freqs, spg_times, spg


"""
SPECTROGRAMS
------------------
"""


def get_spextrogram(
    sig, window_length=4, overlap=2, window="hann", f_range=None, t_range=None, **kwargs
):
    """Calculates a spectrogram and returns as xr.DataArray with dimensions datetime, frequency, channel

    Parameters
    ----------
    Sig --> Should be an xr.DataArray with time or datetime dimension, and a fs attribute
    see ecephys.signal.timefrequency.single_spectrogram_welch for details on kwargs
    """
    try:
        sig = sig.swap_dims({"datetime": "time"})
    except:
        print(
            "Passing Error in get_spextrogram because xarray already has time dimension, no need to swap it in"
        )

    # Add the kwargs
    # window length in number of samples
    kwargs["nperseg"] = int(window_length * sig.fs)

    # overlap in number of samples
    kwargs["noverlap"] = int(overlap * sig.fs)

    # window function
    kwargs["window"] = window

    # frequency range
    kwargs["f_range"] = f_range

    # time range
    kwargs["t_range"] = t_range

    if "channel" in sig.dims:
        freqs, spg_time, spg = parallel_spectrogram_welch(
            sig.transpose("time", "channel").values, sig.fs, **kwargs
        )
    else:
        print("Single channel spectrogram")
        freqs, spg_time, spg = single_spectrogram_welch(sig.values, sig.fs, **kwargs)

    time = sig.time.values.min() + spg_time

    if "timedelta" in list(sig.coords):
        timedelta = sig.timedelta.values.min() + pd.to_timedelta(spg_time, "s")
    if "datetime" in list(sig.coords):
        datetime = sig.datetime.values.min() + pd.to_timedelta(spg_time, "s")

    if "channel" in sig.dims:
        xarray_spg = xr.DataArray(
            spg,
            dims=("frequency", "time", "channel"),
            coords={
                "frequency": freqs,
                "time": time,
                "channel": sig.channel.values,
                "timedelta": ("time", timedelta),
                "datetime": ("time", datetime),
            },
            attrs={"units": f"{sig.units}^2/Hz"},
        )
    else:
        xarray_spg = xr.DataArray(
            spg,
            dims=("frequency", "time"),
            coords={
                "frequency": freqs,
                "time": time,
                "timedelta": ("time", timedelta),
                "datetime": ("time", datetime),
            },
            attrs={"units": f"{sig.units}^2/Hz"},
        )

    # return xarray_spg with default dimension = datetime
    return xarray_spg.swap_dims({"time": "datetime"})


def get_bandpower(spg, f_range):
    """Get band-limited power from a spectrogram.
    Parameters
    ----------
    spg: xr.DataArray (frequency, datetime, [channel])
        Spectrogram data.
    f_range: slice
        Frequency range to restrict to, as slice(f_low, f_high).
    Returns:
    --------
    bandpower: xr.DataArray (datetime, [channel])
        Sum of the power in `f_range` at each point in time.
    """
    bandpower = spg.sel(frequency=slice(*f_range)).sum(dim="frequency")
    bandpower.attrs["f_range"] = f_range

    return bandpower


def get_relative_bp(spg1, spg2, f_range=(0.5, 4), median=False):
    """
    Takes two spectrograms and returns a given bandpower for one of them (spg1) relative to the given bandpower of the other (spg2)

    No state-based filtering is done here, but state-specific spectrograms can be passed in

    The mean of spg2-bandpower is used as the default reference, unless median=True, in which case the median is used
    """

    bp1 = get_bandpower(spg1, f_range)
    bp2 = get_bandpower(spg2, f_range)

    if median:
        bp2 = bp2.median(dim="datetime")
    else:
        bp2 = bp2.mean(dim="datetime")

    return (bp1 / bp2) * 100


def get_bp_set(spg, bands=bands):
    """
    Returns a set of bandpower timeseries for a given spectrogram
    -------------------------------------------------------------
    spg --> xarray.DataArray with datetime dimension
    bands --> dictionary of frequency ranges
    """
    assert type(spg) is xr.core.dataarray.DataArray, "spg must be an xarray.DataArray"

    bp_ds = xr.Dataset({})
    bp_vars = {}
    keys = list(bands.keys())

    for k in keys:
        bp_vars[k] = get_bandpower(spg, bands[k])

    bp_set = bp_ds.assign(**bp_vars)

    return bp_set
