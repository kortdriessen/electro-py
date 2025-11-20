from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except Exception:
    pd = None  # type: ignore
    _HAS_PANDAS = False

try:
    import polars as pl  # type: ignore

    _HAS_POLARS = True
except Exception:
    pl = None  # type: ignore
    _HAS_POLARS = False

try:
    import xarray as xr  # type: ignore

    _HAS_XARRAY = True
except Exception:
    xr = None  # type: ignore
    _HAS_XARRAY = False


import numpy as np

try:
    from scipy import signal as _signal
    from scipy import interpolate as _interp

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    _signal = None
    _interp = None


def _validate_positive_float(value: Any, name: str) -> float:
    try:
        v = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be convertible to float; got {value!r}") from exc
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a positive, finite float; got {v!r}")
    return v


def _gaussian_kernel1d(sigma_samples: float, truncate: float = 4.0) -> np.ndarray:
    # Ensure sigma is at least a small epsilon to avoid zero-length kernels
    sigma = max(float(sigma_samples), 1e-12)
    radius = int(truncate * sigma + 0.5)
    if radius == 0:
        # Degenerate case: essentially no smoothing; use identity kernel
        return np.array([1.0], dtype=float)
    x = np.arange(-radius, radius + 1, dtype=float)
    phi = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = phi / phi.sum()
    return kernel


def _nan_aware_convolve_same(y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Replace NaNs with 0 for signal convolution and track valid weights separately
    finite_mask = np.isfinite(y).astype(float)
    y_safe = np.where(np.isfinite(y), y, 0.0)
    conv_y = np.convolve(y_safe, kernel, mode="same")
    conv_w = np.convolve(finite_mask, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = conv_y / conv_w
    out[conv_w == 0] = np.nan
    return out


def _infer_fs_from_time_values(time_values: np.ndarray) -> float:
    # Convert datetime-like to seconds if needed
    if np.issubdtype(time_values.dtype, np.datetime64):
        # Convert to seconds (float)
        time_seconds = (
            time_values.astype("datetime64[ns]").astype(np.int64) / 1e9
        ).astype(float)
    elif np.issubdtype(time_values.dtype, np.timedelta64):
        time_seconds = (
            time_values.astype("timedelta64[ns]").astype(np.int64) / 1e9
        ).astype(float)
    else:
        time_seconds = time_values.astype(float)

    diffs = np.diff(time_seconds)
    # Remove non-finite or zero diffs
    diffs = diffs[np.isfinite(diffs) & (diffs != 0)]
    if diffs.size == 0:
        raise ValueError(
            "Unable to infer sampling rate from time values (no valid diffs)."
        )
    dt = float(np.median(np.abs(diffs)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Inferred non-positive sampling interval from time values.")
    return 1.0 / dt


def _smooth_1d_array(y: np.ndarray, sigma_samples: float) -> np.ndarray:
    kernel = _gaussian_kernel1d(sigma_samples)
    return _nan_aware_convolve_same(y.astype(float, copy=False), kernel)


def _apply_smoothing_along_axis(
    values: np.ndarray, sigma_samples: float, axis: int
) -> np.ndarray:
    if values.ndim == 1:
        return _smooth_1d_array(values, sigma_samples)
    # Move target axis to the end, reshape to 2D (batch, time), apply row-wise, then restore shape
    v = np.moveaxis(values, axis, -1)
    original_shape = v.shape
    batch = int(np.prod(original_shape[:-1]))
    time_len = original_shape[-1]
    v2d = v.reshape(batch, time_len)
    out2d = np.empty_like(v2d, dtype=float)
    for i in range(batch):
        out2d[i] = _smooth_1d_array(v2d[i], sigma_samples)
    out = out2d.reshape(original_shape)
    return np.moveaxis(out, -1, axis)


def smooth_signal(
    sig: Any,
    smoothing_sigma: float,
    col_to_smooth: Optional[str] = None,
    time_id: Optional[str] = None,
    fs: Optional[float] = None,
):
    """
    Smooth a time-series signal using a Gaussian kernel with NaN-aware handling.

    The function accepts multiple input types and returns the same type:
      - numpy.ndarray (1D): returns a float numpy array of the same shape
      - pandas.DataFrame: returns a new DataFrame with the target column replaced by smoothed values
      - polars.DataFrame: returns a new DataFrame with the target column replaced by smoothed values
      - xarray.DataArray: returns a new DataArray with values smoothed along the time dimension

    Parameters
    ----------
    sig : numpy.ndarray | pandas.DataFrame | polars.DataFrame | xarray.DataArray
        The signal container.
    smoothing_sigma : float
        Gaussian standard deviation in SECONDS. For numpy arrays, you must supply `fs`.
    col_to_smooth : str, optional
        Name of the column to smooth for DataFrame inputs. Ignored for numpy arrays and DataArray inputs.
    time_id : str, optional
        Name of the time column for DataFrame inputs or the time dimension for DataArray inputs.
    fs : float, optional
        Sampling rate in Hz. Required for numpy arrays; inferred from `time_id` for other types.

    Notes
    -----
    - For DataFrame/DataArray inputs, sampling rate is inferred from the median delta of the time column/dimension.
    - Smoothing is applied with a Gaussian kernel. NaN values are handled by normalizing with the convolved valid-sample mask.
    - If the inferred kernel becomes effectively length 1 (very small sigma), the data is returned unchanged (copied) in float dtype.
    """

    sigma_sec = _validate_positive_float(smoothing_sigma, "smoothing_sigma")

    # numpy ndarray
    if isinstance(sig, np.ndarray):
        if sig.ndim != 1:
            raise ValueError("For numpy arrays, only 1D inputs are supported.")
        if fs is None:
            raise ValueError(
                "For numpy arrays, `fs` (sampling rate in Hz) must be provided."
            )
        fs_val = _validate_positive_float(fs, "fs")
        sigma_samples = sigma_sec * fs_val
        return _smooth_1d_array(sig.astype(float, copy=False), sigma_samples)

    # pandas DataFrame
    if _HAS_PANDAS and isinstance(sig, pd.DataFrame):
        if not isinstance(col_to_smooth, str) or not isinstance(time_id, str):
            raise ValueError(
                "For pandas DataFrame, `col_to_smooth` and `time_id` must be provided as strings."
            )
        if col_to_smooth not in sig.columns:
            raise KeyError(f"Column {col_to_smooth!r} not found in DataFrame.")
        if time_id not in sig.columns:
            raise KeyError(f"Time column {time_id!r} not found in DataFrame.")

        df = sig.copy()
        # Ensure sorted by time to avoid negative/zero diffs causing issues
        df = df.sort_values(time_id, kind="stable").reset_index(drop=True)
        time_vals = df[time_id].to_numpy()
        fs_inferred = _infer_fs_from_time_values(time_vals)
        sigma_samples = sigma_sec * fs_inferred
        smoothed = _smooth_1d_array(
            df[col_to_smooth].to_numpy(dtype=float, copy=False), sigma_samples
        )
        df[col_to_smooth] = smoothed
        return df

    # polars DataFrame
    if _HAS_POLARS and isinstance(sig, pl.DataFrame):
        if not isinstance(col_to_smooth, str) or not isinstance(time_id, str):
            raise ValueError(
                "For polars DataFrame, `col_to_smooth` and `time_id` must be provided as strings."
            )
        if col_to_smooth not in sig.columns:
            raise KeyError(f"Column {col_to_smooth!r} not found in DataFrame.")
        if time_id not in sig.columns:
            raise KeyError(f"Time column {time_id!r} not found in DataFrame.")

        df = sig.sort(time_id)
        time_col = df[time_id]
        # Convert time column to numpy seconds
        # if pl.datatypes.is_datetime(time_col.dtype):
        #    # Polars stores as ns by default
        #    time_seconds = (time_col.cast(pl.Int64).to_numpy() / 1e9).astype(float)
        # elif pl.datatypes.is_duration(time_col.dtype):
        #    time_seconds = (time_col.cast(pl.Int64).to_numpy() / 1e9).astype(float)
        # else:
        time_seconds = time_col.to_numpy().astype(float)
        fs_inferred = _infer_fs_from_time_values(np.asarray(time_seconds))
        sigma_samples = sigma_sec * fs_inferred
        target_vals = df[col_to_smooth].to_numpy()
        smoothed = _smooth_1d_array(np.asarray(target_vals, dtype=float), sigma_samples)
        # Replace the column with smoothed values (Float64)
        return df.with_columns(pl.Series(name=col_to_smooth, values=smoothed))

    # xarray DataArray
    if _HAS_XARRAY and isinstance(sig, xr.DataArray):
        if not isinstance(time_id, str):
            raise ValueError(
                "For xarray.DataArray, `time_id` must be provided as the time dimension name."
            )
        if time_id not in sig.dims:
            raise KeyError(
                f"Dimension {time_id!r} not found in DataArray dims {tuple(sig.dims)!r}."
            )
        axis = sig.get_axis_num(time_id)
        # Try to infer fs from coordinate values along time dimension
        time_coord = sig[time_id].values
        fs_inferred = _infer_fs_from_time_values(np.asarray(time_coord))
        sigma_samples = sigma_sec * fs_inferred
        values = sig.values
        smoothed_vals = _apply_smoothing_along_axis(
            values.astype(float, copy=False), sigma_samples, axis=axis
        )
        # Preserve coords and attrs
        out = xr.DataArray(
            smoothed_vals,
            dims=sig.dims,
            coords=sig.coords,
            attrs=sig.attrs,
            name=sig.name,
        )
        return out

    # If we get here, the type is unsupported or the relevant library is missing
    typename = type(sig).__name__
    raise TypeError(
        "Unsupported input type or missing dependency for type "
        f"{typename!r}. Supported types: numpy.ndarray, pandas.DataFrame, "
        "polars.DataFrame, xarray.DataArray."
    )


import numpy as np


def smooth_1d(
    x: np.ndarray,
    fs: float | None = None,
    method: str = "lowpass_fir",
    # ---- common knobs ----
    nan_policy: str = "interp",  # "interp" (default), "drop" (not recommended), or "keep" (will propagate NaNs)
    # ---- median ----
    median_window_s: float = 0.1,  # seconds
    # ---- savgol ----
    sg_window_s: float = 0.2,  # seconds, must be >= (polyorder+2)/fs
    sg_polyorder: int = 3,
    # ---- butterworth ----
    butt_cutoff_hz: float | None = None,  # if None, defaults to 0.25*Nyquist
    butt_order: int = 4,
    # ---- fir ----
    fir_cutoff_hz: float | None = None,  # if None, defaults to 0.25*Nyquist
    fir_transition_hz: float | None = None,  # if None, 0.1*cutoff (soft)
    fir_kaiser_beta: float = 8.6,  # ~60 dB
    # ---- wavelet ----
    wavelet: str = "db4",
    wavelet_levels: int | None = None,  # if None, picked automatically
    wavelet_thresh: str = "universal",  # "universal" or a float (sigma multiplier)
    wavelet_mode: str = "soft",  # "soft" or "hard"
):
    """
    Return a denoised/smoothed version of x (len N) using the selected method.
    Requires SciPy for Butterworth/FIR/Savitzky-Golay and PyWavelets for wavelet.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    N = x.size
    if N == 0:
        return x.copy()

    # Record NaNs and optionally interpolate over them for filtering
    nan_mask = np.isnan(x)
    if nan_policy not in {"interp", "drop", "keep"}:
        raise ValueError("nan_policy must be 'interp', 'drop', or 'keep'")

    def _interp_nans(y: np.ndarray) -> np.ndarray:
        if not np.any(np.isnan(y)):
            return y
        idx = np.arange(y.size)
        good = ~np.isnan(y)
        if good.sum() < 2:
            # Too many NaNs to interpolate meaningfully
            return y.copy()
        y_filled = y.copy()
        y_filled[~good] = np.interp(idx[~good], idx[good], y[good])
        return y_filled

    x_work = x.copy()
    if nan_policy == "interp":
        x_work = _interp_nans(x_work)
    elif nan_policy == "drop":
        # Not used internally (we preserve length), but left for completeness
        pass
    # nan_policy == "keep": do nothing (filters will propagate NaN)

    y = None

    if method == "median":
        if fs is None:
            raise ValueError("median filter needs fs to set window length in seconds.")
        from scipy.signal import medfilt

        win = max(3, int(round(median_window_s * fs)) | 1)  # odd >= 3
        y = medfilt(x_work, kernel_size=win)

    elif method == "savgol":
        if fs is None:
            raise ValueError("Savitzky-Golay needs fs to set window length.")
        from scipy.signal import savgol_filter

        win = int(round(sg_window_s * fs))
        if win % 2 == 0:
            win += 1
        win = max(win, sg_polyorder + 2 | 1)  # ensure enough points and odd
        y = savgol_filter(
            x_work, window_length=win, polyorder=sg_polyorder, mode="interp"
        )

    elif method == "butterworth":
        if fs is None:
            raise ValueError("Butterworth needs fs.")
        from scipy.signal import butter, filtfilt

        nyq = fs / 2.0
        cutoff = butt_cutoff_hz or (0.25 * nyq)
        if not (0 < cutoff < nyq):
            raise ValueError("butt_cutoff_hz must be between 0 and Nyquist.")
        b, a = butter(butt_order, cutoff / nyq, btype="low", analog=False)
        y = filtfilt(b, a, x_work, method="pad")

    elif method == "lowpass_fir":
        if fs is None:
            raise ValueError("FIR low-pass needs fs.")
        from scipy.signal import kaiserord, firwin, filtfilt

        nyq = fs / 2.0
        cutoff = fir_cutoff_hz or (0.25 * nyq)
        if not (0 < cutoff < nyq):
            raise ValueError("fir_cutoff_hz must be between 0 and Nyquist.")
        # Transition width
        tw = fir_transition_hz or max(1e-3, 0.1 * cutoff)
        # Design length from transition and desired ripple (use beta given)
        # Approximate N using kaiserord with ~60 dB ripple
        ripple_db = 60.0
        N_taps, beta = kaiserord(ripple_db, tw / nyq)
        # Ensure odd length for nicer filtfilt symmetry
        if N_taps % 2 == 0:
            N_taps += 1
        # Override beta if user supplied an explicit one
        beta = fir_kaiser_beta if fir_kaiser_beta is not None else beta
        taps = firwin(N_taps, cutoff / nyq, window=("kaiser", beta))
        y = filtfilt(taps, [1.0], x_work, method="pad")

    elif method == "wavelet":
        try:
            import pywt
        except Exception as e:
            raise ImportError("PyWavelets is required for method='wavelet'") from e

        # Decompose
        max_level = pywt.dwt_max_level(N, pywt.Wavelet(wavelet).dec_len)
        level = wavelet_levels or max(1, min(6, max_level))
        coeffs = pywt.wavedec(x_work, wavelet, mode="symmetric", level=level)

        # Estimate noise sigma from the finest detail coeffs
        detail = coeffs[-1]
        sigma = np.median(np.abs(detail)) / 0.6745 if detail.size else 0.0

        if isinstance(wavelet_thresh, str) and wavelet_thresh == "universal":
            thr = sigma * np.sqrt(2.0 * np.log(N)) if sigma > 0 else 0.0
        elif isinstance(wavelet_thresh, (int, float)):
            thr = float(wavelet_thresh) * sigma
        else:
            raise ValueError(
                "wavelet_thresh must be 'universal' or a float multiplier."
            )

        # Threshold detail coeffs
        def _shrink(c):
            if wavelet_mode == "soft":
                return np.sign(c) * np.maximum(np.abs(c) - thr, 0.0)
            elif wavelet_mode == "hard":
                return c * (np.abs(c) >= thr)
            else:
                raise ValueError("wavelet_mode must be 'soft' or 'hard'.")

        coeffs_thr = [coeffs[0]] + [_shrink(c) for c in coeffs[1:]]
        y = pywt.waverec(coeffs_thr, wavelet, mode="symmetric")
        # Match length exactly (waverec can be off by a sample depending on wavelet)
        if y.size != N:
            y = y[:N] if y.size > N else np.pad(y, (0, N - y.size), mode="edge")

    else:
        raise ValueError(f"Unknown method: {method}")

    # Restore original NaNs if requested
    if nan_policy in {"interp", "keep"} and np.any(nan_mask):
        y = y.astype(float, copy=False)
        y[nan_mask] = np.nan

    return y


import numpy as np


def derivative_1d(
    x: np.ndarray,
    fs: float,
    method: str = "smooth_diff",
    *,
    # NaN handling
    nan_policy: str = "interp",  # "interp", "keep"
    # Savitzky–Golay params
    sg_window_s: float = 0.2,  # seconds; choose so window >= (polyorder+2)/fs
    sg_polyorder: int = 3,
    # Smooth-then-diff params (FIR low-pass)
    lp_cutoff_hz: float | None = None,  # if None, 0.25*Nyquist
    lp_transition_hz: float | None = None,  # if None, 0.1*cutoff
    lp_ripple_db: float = 60.0,  # Kaiser design target attenuation
    # FFT params
    fft_taper: bool = True,  # apply a light Tukey window to reduce edge ringing
    fft_alpha: float = 0.1,  # Tukey window shape (0=Hann, 1=rect)
):
    """
    Compute instantaneous derivative dx/dt of a 1D signal x sampled at fs [Hz].
    Returns an array of the same length as x. Units: x-units per second.

    Methods:
      - "gradient": central finite differences (no smoothing).
      - "savgol": Savitzky–Golay derivative (denoise+deriv together).
      - "smooth_diff": zero-phase FIR low-pass (filtfilt) then central diff.
      - "fft": frequency-domain derivative (assumes periodicity; may ring at edges).
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    N = x.size
    if N == 0:
        return x.copy()

    if fs <= 0:
        raise ValueError("fs must be positive")

    # NaN handling
    nan_mask = np.isnan(x)
    if nan_policy not in {"interp", "keep"}:
        raise ValueError("nan_policy must be 'interp' or 'keep'")

    def _interp_nans(y: np.ndarray) -> np.ndarray:
        if not np.any(np.isnan(y)):
            return y
        idx = np.arange(y.size)
        good = ~np.isnan(y)
        if good.sum() < 2:
            # Not enough points to interpolate meaningfully
            return y.copy()
        y2 = y.copy()
        y2[~good] = np.interp(idx[~good], idx[good], y[good])
        return y2

    x_work = _interp_nans(x) if nan_policy == "interp" else x.copy()

    if method == "gradient":
        # Central differences with physical scaling
        y = np.gradient(x_work) * fs

    elif method == "savgol":
        from scipy.signal import savgol_filter

        win = int(round(sg_window_s * fs))
        if win < sg_polyorder + 2:
            win = sg_polyorder + 2
        if win % 2 == 0:
            win += 1
        y = savgol_filter(
            x_work,
            window_length=win,
            polyorder=sg_polyorder,
            deriv=1,
            delta=1.0 / fs,
            mode="interp",
        )

    elif method == "smooth_diff":
        # Zero-phase FIR low-pass via filtfilt, then central difference
        from scipy.signal import kaiserord, firwin, filtfilt

        nyq = fs / 2.0
        cutoff = lp_cutoff_hz or (0.25 * nyq)
        if not (0 < cutoff < nyq):
            raise ValueError("lp_cutoff_hz must be in (0, Nyquist)")
        tw = lp_transition_hz or max(1e-3, 0.1 * cutoff)
        N_taps, beta = kaiserord(lp_ripple_db, tw / nyq)
        if N_taps % 2 == 0:
            N_taps += 1
        taps = firwin(N_taps, cutoff / nyq, window=("kaiser", beta))
        xs = filtfilt(taps, [1.0], x_work, method="pad")
        y = np.gradient(xs) * fs

    elif method == "fft":
        # Frequency-domain derivative: d/dt <-> i*2*pi*f
        # Optionally taper to mitigate edge effects
        xw = x_work
        if fft_taper:
            # Tukey window
            n = np.arange(N)
            L = N - 1
            alpha = float(fft_alpha)
            w = np.ones(N)
            if 0 < alpha < 1:
                # build Tukey
                per = alpha * L
                m1 = int(np.floor(per / 2))
                m2 = int(np.floor(L - per / 2))
                # cosine taper up
                if m1 > 0:
                    t = np.linspace(0, np.pi, m1 + 1)
                    w[: m1 + 1] = 0.5 * (1 - np.cos(t))
                # cosine taper down
                if m2 < L:
                    t = np.linspace(0, np.pi, L - m2 + 1)
                    w[m2:] = 0.5 * (1 - np.cos(t[::-1]))
            xw = x_work * w

        # rfft frequencies in Hz
        X = np.fft.rfft(xw)
        freqs = np.fft.rfftfreq(N, d=1.0 / fs)
        dX = 1j * 2.0 * np.pi * freqs * X
        y = np.fft.irfft(dX, n=N)

        # If tapered, optionally undo average gain (not critical for derivative)
        # We leave it as-is; derivative units remain correct.

    else:
        raise ValueError(f"Unknown method: {method}")

    # Restore original NaNs (keep gaps as NaN in the result)
    if np.any(nan_mask):
        y = y.astype(float, copy=False)
        y[nan_mask] = np.nan

    return y


import numpy as np


def match_data_to_times(data, current_times, new_times):
    """
    Resample `data` from `current_times` onto `new_times`.

    - `data`: np.ndarray where exactly one axis has length == len(current_times)
    - `current_times`: 1D array-like of monotonic time stamps (not necessarily perfectly uniform)
    - `new_times`: 1D array-like of desired time stamps (resampled output grid)

    Returns
    -------
    new_data : np.ndarray
        `data` resampled so that the chosen time axis has length len(new_times).
        The shape matches `data` except along the time axis.
    """
    # -- Normalize inputs
    data = np.asarray(data)
    t_old = np.asarray(current_times, dtype=float).ravel()
    t_new = np.asarray(new_times, dtype=float).ravel()
    if t_old.ndim != 1 or t_new.ndim != 1:
        raise ValueError("current_times and new_times must be 1D.")

    # -- Identify the time axis
    matches = [ax for ax, n in enumerate(data.shape) if n == t_old.size]
    if len(matches) != 1:
        raise ValueError(
            "Exactly one axis of `data` must match len(current_times). "
            f"Found matches at axes {matches}."
        )
    time_axis = matches[0]

    # -- Ensure ascending times; sort data and times if needed
    order_old = np.argsort(t_old)
    if not np.all(order_old == np.arange(t_old.size)):
        t_old = t_old[order_old]
        data = np.take(data, order_old, axis=time_axis)

    order_new = np.argsort(t_new)
    inv_order_new = np.empty_like(order_new)
    inv_order_new[order_new] = np.arange(order_new.size)
    t_new_sorted = t_new[order_new]

    # -- Estimate sampling rates (robust to tiny jitter)
    if t_old.size > 1:
        dt_old = np.diff(t_old)
        fs_old = 1.0 / np.median(dt_old)
    else:
        dt_old = np.array([np.inf])
        fs_old = np.inf

    if t_new_sorted.size > 1:
        dt_new = np.diff(t_new_sorted)
        fs_new = 1.0 / np.median(dt_new)
    else:
        fs_new = 0.0

    downsampling = np.isfinite(fs_old) and (fs_new < fs_old)

    # -- Move time axis to the end for convenience: (..., T)
    x = np.moveaxis(data, time_axis, -1)
    T = x.shape[-1]
    x2d = x.reshape(-1, T)  # (M, T) where M = product(other dims)

    # -- If downsampling, apply an anti-aliasing low-pass FIR to each 1D trace
    if downsampling and T > 1:
        # Cutoff below the new Nyquist to leave headroom for transition band
        cutoff_hz = 0.45 * fs_new  # Hz
        # A reasonable transition width (Hz); wider -> shorter filter
        transition_hz = max(1e-12, 0.1 * cutoff_hz)

        # Kaiser window design (~60 dB stopband). No SciPy required.
        A = 60.0
        if A > 50:
            beta = 0.1102 * (A - 8.7)
        elif A >= 21:
            beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
        else:
            beta = 0.0

        # Kaiser order estimate: N ≈ (A-8)/(2.285*Δω), with Δω = 4π*transition/fs
        N = int(np.ceil((A - 8.0) * fs_old / (2.285 * 4.0 * np.pi * transition_hz)))
        N = max(N, 31)
        if N % 2 == 0:
            N += 1  # make it odd for symmetric linear-phase
        M = N // 2

        n = np.arange(N) - M
        # Ideal low-pass impulse response (continuous-time sinc sampled on old grid)
        h_ideal = 2.0 * cutoff_hz / fs_old * np.sinc(2.0 * cutoff_hz * n / fs_old)
        window = np.kaiser(N, beta)
        taps = h_ideal * window
        taps /= taps.sum()  # unity DC gain

        # Convolve each row (zero-padded at the ends, like np.convolve(..., 'same'))
        y2d = np.empty_like(x2d, dtype=float)
        for i in range(x2d.shape[0]):
            y2d[i] = np.convolve(x2d[i], taps, mode="same")
    else:
        # No anti-aliasing needed (upsampling or degenerate cases)
        y2d = x2d.astype(float, copy=False)

    # -- Interpolate each trace to t_new_sorted (piecewise-linear, stable, fast)
    out2d = np.empty((y2d.shape[0], t_new_sorted.size), dtype=float)
    for i in range(y2d.shape[0]):
        out2d[i] = np.interp(
            t_new_sorted, t_old, y2d[i], left=y2d[i, 0], right=y2d[i, -1]
        )

    # -- Restore original dimensionality and axis placement
    new_shape = (*x.shape[:-1], t_new_sorted.size)
    out = out2d.reshape(new_shape)
    out = np.moveaxis(out, -1, time_axis)

    # -- Undo sorting of new_times to match user's order
    out = np.take(out, inv_order_new, axis=time_axis)
    return out


def downsample_to_new_fs(data, old_fs, new_fs, new_n_samples):
    """
    Downsample 1D data from old_fs (Hz) to new_fs and return exactly
    new_n_samples points. Uses linear interpolation over the original
    time extent. If new_fs and new_n_samples are slightly inconsistent,
    new_n_samples takes precedence and we span the original duration.
    """
    import numpy as np

    arr = np.asarray(data)
    if arr.ndim != 1:
        arr = arr.ravel()

    n_old = arr.shape[0]
    if new_n_samples <= 0:
        return np.array([], dtype=arr.dtype)
    if n_old == 0:
        return np.zeros((int(new_n_samples),), dtype=arr.dtype)
    if n_old == 1:
        return np.full((int(new_n_samples),), arr[0], dtype=arr.dtype)

    # Original sample times (inclusive of both ends)
    t_old = np.arange(n_old, dtype=np.float64) / float(old_fs)
    t_end = t_old[-1]

    # If new_fs matches new_n_samples for the same duration within 1 sample,
    # adjust t_end to align exactly with new_fs spacing.
    if new_fs is not None and new_fs > 0 and new_n_samples >= 2:
        expected_n = int(round(t_end * float(new_fs))) + 1
        if abs(expected_n - int(new_n_samples)) <= 1:
            t_end = (int(new_n_samples) - 1) / float(new_fs)

    # Target sample times across the chosen duration
    t_new = np.linspace(0.0, t_end, int(new_n_samples), dtype=np.float64)

    # Linear interpolation
    out = np.interp(t_new, t_old, arr.astype(np.float64, copy=False))

    # Preserve float dtype if input was float; else return float64 (interp default)
    if np.issubdtype(arr.dtype, np.floating):
        return out.astype(arr.dtype, copy=False)
    return out


def _validate_rates(fs_in: float, fs_out: float) -> None:
    if not (np.isfinite(fs_in) and np.isfinite(fs_out)):
        raise ValueError("Sampling rates must be finite numbers.")
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError("Sampling rates must be positive.")


def _target_length(n_in: int, fs_in: float, fs_out: float) -> int:
    # Round to the nearest integer number of samples at the new rate
    return int(np.round(n_in * fs_out / fs_in))


def _fraction(
    fs_out: float, fs_in: float, max_denominator: int = 1000
) -> tuple[int, int]:
    # Rational approximation of fs_out/fs_in for polyphase resampling
    from fractions import Fraction

    # Fraction(numerator, denominator) requires Rational inputs; use single-arg with float ratio.
    ratio = float(fs_out) / float(fs_in)
    frac = Fraction(ratio).limit_denominator(max_denominator)
    return int(frac.numerator), int(frac.denominator)


def _match_length(y: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    # Ensure output length along axis matches n_out (tolerate +/- 1 and adjust)
    cur_len = y.shape[axis]
    if cur_len == n_out:
        return y
    if cur_len > n_out:
        slicer = [slice(None)] * y.ndim
        slicer[axis] = slice(0, n_out)
        return y[tuple(slicer)]
    # pad by repeating the last available sample
    pad_count = n_out - cur_len
    last = np.take(y, indices=[cur_len - 1], axis=axis)
    last_rep = np.repeat(last, repeats=pad_count, axis=axis)
    return np.concatenate([y, last_rep], axis=axis)


def _gather_along_last_axis(x2: np.ndarray, idx: np.ndarray) -> np.ndarray:
    # x2 shape: (batch, n_in), idx shape: (n_out,)
    # Broadcast idx to (batch, n_out) and gather
    idx2 = np.broadcast_to(idx[None, :], (x2.shape[0], idx.shape[0]))
    return np.take_along_axis(x2, idx2, axis=1)


def _linear_family_resample(
    x: np.ndarray, fs_in: float, fs_out: float, n_out: int, axis: int, mode: str
) -> np.ndarray:
    """
    Vectorized resampling using index-space interpolation on uniformly-sampled data.

    Supported modes:
      - 'linear': linear interpolation
      - 'nearest': nearest-neighbor (round)
      - 'zoh': zero-order hold (floor, a.k.a. previous-sample hold)
    """
    x = np.asarray(x)
    # Move target axis to the end and flatten the rest to one batch dimension
    if axis < 0:
        axis += x.ndim
    x_moved = np.moveaxis(x, axis, -1)
    batch_shape = x_moved.shape[:-1]
    n_in = x_moved.shape[-1]
    if n_in == 0:
        return np.moveaxis(x_moved, -1, axis)  # empty
    x2 = x_moved.reshape(-1, n_in)

    # Fractional indices in the input sample index domain
    # idx = t_out * fs_in, where t_out = arange(n_out) / fs_out
    idx_f = (np.arange(n_out, dtype=np.float64) * fs_in) / fs_out

    if mode == "nearest":
        idx = np.clip(np.rint(idx_f).astype(np.int64), 0, n_in - 1)
        y2 = _gather_along_last_axis(x2, idx)
    elif mode == "zoh":
        idx = np.clip(np.floor(idx_f).astype(np.int64), 0, n_in - 1)
        y2 = _gather_along_last_axis(x2, idx)
    elif mode == "linear":
        if n_in == 1:
            y2 = np.broadcast_to(x2[:, :1], (x2.shape[0], n_out))
        else:
            i0 = np.clip(np.floor(idx_f).astype(np.int64), 0, n_in - 2)
            i1 = i0 + 1
            alpha = (idx_f - i0).astype(np.float64)  # in [0,1)
            v0 = _gather_along_last_axis(x2, i0)
            v1 = _gather_along_last_axis(x2, i1)
            y2 = (1.0 - alpha[None, :]) * v0 + alpha[None, :] * v1
    else:
        raise ValueError(f"Unsupported interpolation mode '{mode}'.")

    y = y2.reshape(*batch_shape, n_out)
    y = np.moveaxis(y, -1, axis)
    return y


def _block_reduce_downsample(
    x: np.ndarray, q: int, axis: int, aggregate: str = "mean"
) -> np.ndarray:
    """
    Integer-factor downsampling by aggregating non-overlapping blocks (boxcar lowpass + decimate).
    aggregate: 'mean', 'median', 'first', 'last', 'max', 'min'
    """
    if q <= 0:
        raise ValueError("Block size (q) must be a positive integer.")
    x = np.asarray(x)
    if axis < 0:
        axis += x.ndim
    # Move target axis to the end and flatten the rest to one batch dimension
    x_moved = np.moveaxis(x, axis, -1)
    batch_shape = x_moved.shape[:-1]
    n_in = x_moved.shape[-1]
    if n_in == 0:
        return np.moveaxis(x_moved, -1, axis)
    x2 = x_moved.reshape(-1, n_in)

    n_blocks = n_in // q
    if n_blocks == 0:
        # Not enough samples for a single full block: return first sample
        y2 = x2[:, :1]
    else:
        cut = n_blocks * q
        x2c = x2[:, :cut]
        x3 = x2c.reshape(x2.shape[0], n_blocks, q)
        if aggregate == "mean":
            y2 = x3.mean(axis=2)
        elif aggregate == "median":
            y2 = np.median(x3, axis=2)
        elif aggregate == "first":
            y2 = x3[:, :, 0]
        elif aggregate == "last":
            y2 = x3[:, :, -1]
        elif aggregate == "max":
            y2 = x3.max(axis=2)
        elif aggregate == "min":
            y2 = x3.min(axis=2)
        else:
            raise ValueError(f"Unsupported aggregate '{aggregate}'.")
    y = y2.reshape(*batch_shape, y2.shape[-1])
    y = np.moveaxis(y, -1, axis)
    return y


def downsample_signal(
    x: np.ndarray,
    fs_in: float,
    fs_out: float,
    *,
    t: np.ndarray | None = None,
    method: str = "auto",
    axis: int = -1,
    max_denominator: int = 1000,
    window=("kaiser", 5.0),
    block_aggregate: str = "mean",
    zero_phase: bool = True,
    dtype: np.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample uniformly sampled data from fs_in to fs_out.

    This function supports several methods and will use SciPy when available for
    the highest quality anti-aliased resampling. If SciPy is not available, it
    falls back to high-quality vectorized NumPy implementations.

    Parameters
    ----------
    x : np.ndarray
        Input array. The target resampling axis is controlled by `axis`.
        Supports any shape; data are assumed uniformly sampled along `axis`.
    fs_in : float
        Original sampling rate (Hz).
    fs_out : float
        Desired (lower) sampling rate (Hz).
    t : np.ndarray, optional
        Optional 1D time array aligned with `x` along `axis`. If provided, it must have
        length equal to `x.shape[axis]`. The returned `t_out` will correspond to the
        resampled signal. If not provided, `t_out` is constructed assuming start time
        `t0 = 0` and uniform spacing `1/fs_in` for the input and `1/fs_out` for the output.
    method : {'auto','polyphase','fft','linear','nearest','zoh','block','decimate_iir','decimate_fir'}, optional
        - 'auto' (default): uses 'polyphase' if SciPy is available; otherwise 'linear'.
        - 'polyphase': high-quality polyphase FIR rational resampling (scipy.signal.resample_poly).
           Anti-alias filtering is built-in. Works for arbitrary rational ratios.
        - 'fft': Fourier-domain resampling (scipy.signal.resample). Assumes periodic/band-limited signals.
        - 'linear': time-domain linear interpolation in index space (NumPy-only). Simple and robust.
        - 'nearest': nearest-neighbor (may alias; fastest).
        - 'zoh': zero-order hold (previous-sample hold; may alias).
        - 'block': integer-factor block aggregation (boxcar lowpass followed by decimation).
           Requires fs_in/fs_out to be near an integer. Use `block_aggregate` to choose 'mean' (default),
           'median', 'first', 'last', 'max', or 'min'.
        - 'decimate_iir' / 'decimate_fir': integer-factor decimation with IIR or FIR anti-alias filtering
           (scipy.signal.decimate). Requires SciPy and near-integer factor.
    axis : int, optional
        Axis to resample along (default: -1).
    max_denominator : int, optional
        Maximum denominator when approximating fs_out/fs_in as a rational for 'polyphase'.
        Larger values improve accuracy but may increase computation.
    window : tuple | str, optional
        FIR window for 'polyphase' (scipy.signal.resample_poly). Default ('kaiser', 5.0).
    block_aggregate : str, optional
        Aggregation for 'block' method: 'mean'|'median'|'first'|'last'|'max'|'min'.
    zero_phase : bool, optional
        For 'decimate_*' methods (SciPy), whether to apply zero-phase filtering (default True).
    dtype : np.dtype | None, optional
        If provided, cast the result to this dtype. By default, float precision is preserved for
        float and complex inputs; integer inputs will be promoted to float.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Tuple `(y, t_out)` where:
        - `y` is the resampled array with the same shape as `x` except length along
          `axis` is round(x.shape[axis] * fs_out / fs_in).
        - `t_out` is a 1D time array of length `y.shape[axis]` corresponding to the
          resampled signal.

    Method guidance
    ---------------
    - Prefer 'polyphase' for general-purpose, high-quality, anti-aliased resampling (arbitrary ratios).
    - 'fft' can be excellent for band-limited signals but assumes periodicity; may introduce ringing at edges.
    - 'linear' is robust and dependency-free but may alias under heavy downsampling.
    - 'block' is a simple integer-factor approach with a boxcar lowpass; not as selective as FIR/IIR.
    - 'decimate_iir'/'decimate_fir' are good integer-factor choices when SciPy is available.

    Notes
    -----
    - Anti-alias filtering is essential when fs_out < fs_in. 'polyphase' and 'decimate_*' include it by design.
    - For integer-factor downsampling (e.g., fs_in=1000 -> fs_out=200), 'block' and 'decimate_*' are appropriate.
    - This function assumes uniform sampling and does not accept irregular time stamps.
    - Time handling:
        * For 'polyphase', 'fft', and interpolation modes ('linear', 'nearest', 'zoh'), `t_out`
          is constructed as `t0 + arange(n_out)/fs_out`, where `t0 = t[0]` if `t` is given, else 0.
        * For 'block', `t_out` represents each block:
              - 'first'/'last' use the first/last time in each block (if `t` given), or the corresponding
                index time based on `fs_in` if `t` is not provided.
              - 'mean'/'median' (and also 'max'/'min' for lack of a unique representative time) use the
                block's mean (or median) time when `t` is given; otherwise the block center time.
          Length is matched to the signal via trimming/padding if needed.
        * For 'decimate_*', if `t` is provided, `t_out` is `t[::q]` (matched to length). Otherwise
          `t0 + arange(n_out)/fs_out`.

    Examples
    --------
    >>> y = downsample_signal(x, fs_in=1000, fs_out=200, method='polyphase')
    >>> y2 = downsample_signal(x, 2000, 250, method='block', block_aggregate='mean')
    >>> y3 = downsample_signal(x, 1000, 333, method='linear')  # no SciPy required
    """
    _validate_rates(fs_in, fs_out)
    if fs_out >= fs_in:
        raise ValueError(
            "Downsampling requires fs_out < fs_in. Use upsample_signal for fs_out >= fs_in."
        )

    n_in = np.asarray(x).shape[axis]
    n_out = _target_length(n_in, fs_in, fs_out)

    # Validate/prepare input time vector if provided
    t_arr: np.ndarray | None
    if t is not None:
        t_arr = np.asarray(t, dtype=float).ravel()
        if t_arr.ndim != 1:
            raise ValueError("t must be a 1D array when provided.")
        if t_arr.size != n_in:
            raise ValueError(
                f"t must have length equal to x.shape[axis] ({n_in}); got {t_arr.size}."
            )
        t0 = float(t_arr[0])
    else:
        t_arr = None
        t0 = 0.0

    # Choose default method
    if method == "auto":
        method = "polyphase" if HAS_SCIPY else "linear"

    # Dtype strategy
    x_arr = np.asarray(x)
    in_kind = x_arr.dtype.kind
    working = x_arr.astype(
        np.float64 if in_kind not in ("f", "c") else x_arr.dtype, copy=False
    )

    if method == "polyphase":
        if not HAS_SCIPY:
            # graceful fallback
            y = _linear_family_resample(
                working, fs_in, fs_out, n_out, axis, mode="linear"
            )
        else:
            L, M = _fraction(fs_out, fs_in, max_denominator)
            y = _signal.resample_poly(working, up=L, down=M, axis=axis, window=window)
            y = _match_length(y, n_out, axis)

    elif method == "fft":
        if not HAS_SCIPY:
            y = _linear_family_resample(
                working, fs_in, fs_out, n_out, axis, mode="linear"
            )
        else:
            y = _signal.resample(working, num=n_out, axis=axis)

    elif method in ("linear", "nearest", "zoh"):
        y = _linear_family_resample(working, fs_in, fs_out, n_out, axis, mode=method)

    elif method == "block":
        # Require near-integer factor
        ratio = fs_in / fs_out
        q = int(np.round(ratio))
        if not np.isclose(ratio, q, rtol=0, atol=1e-8):
            raise ValueError(
                f"'block' requires integer decimation factor. fs_in/fs_out={ratio:.6f} not close to integer."
            )
        y = _block_reduce_downsample(working, q=q, axis=axis, aggregate=block_aggregate)
        # Adjust length if needed
        y = _match_length(y, n_out, axis)
        # Construct block-based times
        n_blocks = n_in // q
        if n_blocks == 0:
            t_blocks = np.array([t0], dtype=float)
        else:
            if t_arr is not None:
                cut = n_blocks * q
                t_cut = t_arr[:cut]
                t_grid = t_cut.reshape(n_blocks, q)
                if block_aggregate == "first":
                    t_blocks = t_grid[:, 0]
                elif block_aggregate == "last":
                    t_blocks = t_grid[:, -1]
                elif block_aggregate == "median":
                    t_blocks = np.median(t_grid, axis=1)
                else:
                    # 'mean', 'max', 'min' (no unique time for max/min -> use mean time as representative)
                    t_blocks = t_grid.mean(axis=1)
            else:
                # No explicit t: construct from fs_in
                dt_in = 1.0 / fs_in
                if block_aggregate == "first":
                    offset = 0.0
                elif block_aggregate == "last":
                    offset = (q - 1) * dt_in
                elif block_aggregate == "median":
                    offset = 0.5 * (q - 1) * dt_in
                else:
                    # 'mean', 'max', 'min' -> use block center
                    offset = 0.5 * (q - 1) * dt_in
                starts = np.arange(n_blocks, dtype=float) * (q * dt_in)
                t_blocks = t0 + starts + offset
        # Match t length to y (pad/truncate if off by 1)
        t_out = _match_length(t_blocks.reshape(-1), y.shape[axis], axis=0)

    elif method in ("decimate_iir", "decimate_fir"):
        if not HAS_SCIPY:
            raise ValueError(
                f"'{method}' requires SciPy. Consider 'block' or 'polyphase'."
            )
        ratio = fs_in / fs_out
        q = int(np.round(ratio))
        if not np.isclose(ratio, q, rtol=0, atol=1e-8):
            raise ValueError(
                f"'{method}' requires integer decimation factor. fs_in/fs_out={ratio:.6f} not close to integer."
            )
        ftype = "iir" if method.endswith("iir") else "fir"
        y = _signal.decimate(working, q, ftype=ftype, axis=axis, zero_phase=zero_phase)
        y = _match_length(y, n_out, axis)
        # Time grid: pick every q-th input time if provided; else uniform from fs_out
        if t_arr is not None:
            t_dec = t_arr[::q]
            t_out = _match_length(t_dec.reshape(-1), y.shape[axis], axis=0)
        else:
            t_out = t0 + (np.arange(y.shape[axis], dtype=float) / fs_out)

    else:
        raise ValueError(f"Unknown method '{method}'.")

    # For non-block/decimate methods, build uniform output time grid if not already constructed
    if method not in {"block", "decimate_iir", "decimate_fir"}:
        t_out = t0 + (np.arange(n_out, dtype=float) / fs_out)

    # Output dtype policy for signal (time remains float64)
    if dtype is not None:
        y = y.astype(dtype, copy=False)
    elif in_kind in ("f", "c"):
        y = y.astype(x_arr.dtype, copy=False)
    # else: integer inputs promoted to float by design

    return y, t_out
