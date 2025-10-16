

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
        time_seconds = (time_values.astype("datetime64[ns]").astype(np.int64) / 1e9).astype(float)
    elif np.issubdtype(time_values.dtype, np.timedelta64):
        time_seconds = (time_values.astype("timedelta64[ns]").astype(np.int64) / 1e9).astype(float)
    else:
        time_seconds = time_values.astype(float)

    diffs = np.diff(time_seconds)
    # Remove non-finite or zero diffs
    diffs = diffs[np.isfinite(diffs) & (diffs != 0)]
    if diffs.size == 0:
        raise ValueError("Unable to infer sampling rate from time values (no valid diffs).")
    dt = float(np.median(np.abs(diffs)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Inferred non-positive sampling interval from time values.")
    return 1.0 / dt


def _smooth_1d_array(y: np.ndarray, sigma_samples: float) -> np.ndarray:
    kernel = _gaussian_kernel1d(sigma_samples)
    return _nan_aware_convolve_same(y.astype(float, copy=False), kernel)


def _apply_smoothing_along_axis(values: np.ndarray, sigma_samples: float, axis: int) -> np.ndarray:
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


def smooth_signal(sig: Any, smoothing_sigma: float, col_to_smooth: Optional[str] = None, time_id: Optional[str] = None, fs: Optional[float] = None):
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
            raise ValueError("For numpy arrays, `fs` (sampling rate in Hz) must be provided.")
        fs_val = _validate_positive_float(fs, "fs")
        sigma_samples = sigma_sec * fs_val
        return _smooth_1d_array(sig.astype(float, copy=False), sigma_samples)

    # pandas DataFrame
    if _HAS_PANDAS and isinstance(sig, pd.DataFrame):
        if not isinstance(col_to_smooth, str) or not isinstance(time_id, str):
            raise ValueError("For pandas DataFrame, `col_to_smooth` and `time_id` must be provided as strings.")
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
        smoothed = _smooth_1d_array(df[col_to_smooth].to_numpy(dtype=float, copy=False), sigma_samples)
        df[col_to_smooth] = smoothed
        return df

    # polars DataFrame
    if _HAS_POLARS and isinstance(sig, pl.DataFrame):
        if not isinstance(col_to_smooth, str) or not isinstance(time_id, str):
            raise ValueError("For polars DataFrame, `col_to_smooth` and `time_id` must be provided as strings.")
        if col_to_smooth not in sig.columns:
            raise KeyError(f"Column {col_to_smooth!r} not found in DataFrame.")
        if time_id not in sig.columns:
            raise KeyError(f"Time column {time_id!r} not found in DataFrame.")

        df = sig.sort(time_id)
        time_col = df[time_id]
        # Convert time column to numpy seconds
        if pl.datatypes.is_datetime(time_col.dtype):
            # Polars stores as ns by default
            time_seconds = (time_col.cast(pl.Int64).to_numpy() / 1e9).astype(float)
        elif pl.datatypes.is_duration(time_col.dtype):
            time_seconds = (time_col.cast(pl.Int64).to_numpy() / 1e9).astype(float)
        else:
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
            raise ValueError("For xarray.DataArray, `time_id` must be provided as the time dimension name.")
        if time_id not in sig.dims:
            raise KeyError(f"Dimension {time_id!r} not found in DataArray dims {tuple(sig.dims)!r}.")
        axis = sig.get_axis_num(time_id)
        # Try to infer fs from coordinate values along time dimension
        time_coord = sig[time_id].values
        fs_inferred = _infer_fs_from_time_values(np.asarray(time_coord))
        sigma_samples = sigma_sec * fs_inferred
        values = sig.values
        smoothed_vals = _apply_smoothing_along_axis(values.astype(float, copy=False), sigma_samples, axis=axis)
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