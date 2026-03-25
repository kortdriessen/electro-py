"""Utility functions for working with hypnograms and external data.

All functions are updated to work with the unified ``Hypnogram`` class.
Functions that bridge hypnograms with xarray or operate on raw DataFrames
remain standalone.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from electro_py.hypno.hypno import Hypnogram

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _infer_bout_start(df, bout):
    """Infer a bout's start time from the previous bout's end time.

    Parameters
    ----------
    df : DataFrame
        Hypnogram in Visbrain format with ``end_time``.
    bout
        A row from *df* representing the bout.

    Returns
    -------
    float
        Start time of the bout.
    """
    if bout.name == 0:
        return 0.0
    return df.loc[bout.name - 1].end_time


def load_hypno_file(path: str, st, dt: bool = True) -> Hypnogram:
    """Load a Visbrain-formatted hypnogram.

    Parameters
    ----------
    path : str
        Path to the Visbrain file.
    st : datetime-like
        Start datetime (used when *dt=True*).
    dt : bool
        If True, convert float times to datetimes.

    Returns
    -------
    Hypnogram
    """
    hyp = Hypnogram.from_visbrain(path)
    if dt:
        return hyp.as_datetime(st)
    return hyp


def to_datetime(df, start_datetime) -> Hypnogram:
    """Convert a float-time DataFrame to a datetime ``Hypnogram``.

    Parameters
    ----------
    df : pandas or polars DataFrame
        Must have ``state``, ``start_time``, ``end_time`` columns.
    start_datetime : datetime-like
        Reference datetime for *t = 0*.

    Returns
    -------
    Hypnogram
    """
    hyp = Hypnogram(df)
    return hyp.as_datetime(start_datetime)


# ---------------------------------------------------------------------------
# State labelling
# ---------------------------------------------------------------------------


def get_states_fast(hyp, times, code: bool = False) -> pd.Series:
    """Label each time with its hypnogram state using binary search.

    Thin wrapper around :meth:`Hypnogram.get_states`.

    Parameters
    ----------
    hyp : Hypnogram or DataFrame
        The hypnogram (converted to ``Hypnogram`` if needed).
    times : array-like
        Times to label.
    code : bool
        If True, use the ``state_code`` column instead of ``state``.

    Returns
    -------
    pandas.Series
    """
    if not isinstance(hyp, Hypnogram):
        hyp = Hypnogram(hyp)
    col = "state_code" if code else "state"
    default = 0 if code else "no_state"
    labels = hyp.get_states(times, column=col, default=default)
    return pd.Series(labels, index=getattr(times, "index", None))


def get_states(hyp, times, code: bool = False) -> pd.Series:
    """Label each time with its hypnogram state.

    Same as :func:`get_states_fast` (kept for backwards compatibility).

    Parameters
    ----------
    hyp : Hypnogram or DataFrame
    times : array-like
    code : bool

    Returns
    -------
    pandas.Series
    """
    return get_states_fast(hyp, times, code=code)


def no_states_array(times) -> pd.Series:
    """Return an array of ``"no_state"`` matching the length of *times*.

    Parameters
    ----------
    times : array-like
    """
    return pd.Series(["no_state"] * len(times))


# ---------------------------------------------------------------------------
# xarray integration
# ---------------------------------------------------------------------------


def add_states(dat, hypnogram):
    """Annotate each timepoint in *dat* with the corresponding state label.

    Parameters
    ----------
    dat : xarray.Dataset or xarray.DataArray
        Must have dimension ``datetime``.
    hypnogram : Hypnogram or pandas.DataFrame

    Returns
    -------
    xarray object with new coordinate ``state`` on dimension ``datetime``.
    """
    assert "datetime" in dat.dims, "Data must contain datetime dimension."

    if not isinstance(hypnogram, Hypnogram):
        hypnogram = Hypnogram(hypnogram)

    states = hypnogram.get_states(dat.datetime)
    return dat.assign_coords(state=("datetime", states))


def keep_states(dat, hypnogram, states: list[str]):
    """Select only timepoints corresponding to desired states.

    Parameters
    ----------
    dat : xarray.Dataset or xarray.DataArray
        Must have dimension ``datetime``.
    hypnogram : Hypnogram
    states : list[str]
    """
    assert isinstance(hypnogram, Hypnogram)
    try:
        assert "datetime" in dat.dims, "Data must contain datetime dimension."
    except AssertionError:
        dat = dat.swap_dims({"time": "datetime"})
    keep = hypnogram.keep_states(states).covers_time(dat.datetime)
    return dat.sel(datetime=keep)


def keep_hypnogram_contents(dat, hypnogram):
    """Select only timepoints covered by the hypnogram.

    Parameters
    ----------
    dat : xarray.Dataset or xarray.DataArray
        Must have dimension ``datetime``.
    hypnogram : Hypnogram
    """
    assert isinstance(hypnogram, Hypnogram)
    assert "datetime" in dat.dims, "Data must contain datetime dimension."
    keep = hypnogram.covers_time(dat.datetime)
    return dat.sel(datetime=keep)


# ---------------------------------------------------------------------------
# DataFrame utilities
# ---------------------------------------------------------------------------


def trim_hypnogram(
    df: pl.DataFrame | pd.DataFrame,
    start,
    end,
) -> pl.DataFrame | pd.DataFrame:
    """Trim a hypnogram DataFrame to *[start, end]*, clamping bouts.

    Accepts and returns the same type (polars or pandas).

    Parameters
    ----------
    df : polars.DataFrame or pandas.DataFrame
        Must have ``state``, ``start_time``, ``end_time``.
    start, end
        Time boundaries.

    Returns
    -------
    DataFrame of the same type as *df*, with bouts clamped and trimmed.
    """
    is_pandas = isinstance(df, pd.DataFrame)
    pl_df: pl.DataFrame = pl.from_pandas(df) if is_pandas else df  # type: ignore[assignment]

    required = {"state", "start_time", "end_time"}
    if not required.issubset(pl_df.columns):
        raise AttributeError(
            "Required columns `state`, `start_time`, `end_time` are not present."
        )
    if start > end:
        raise ValueError("Invalid value: expected `start` <= `end`")

    trimmed = (
        pl_df.with_columns(
            pl.when(pl.col("start_time") < start)
            .then(pl.lit(start))
            .otherwise(pl.col("start_time"))
            .alias("start_time"),
            pl.when(pl.col("end_time") > end)
            .then(pl.lit(end))
            .otherwise(pl.col("end_time"))
            .alias("end_time"),
        )
        .filter(pl.col("start_time") < pl.col("end_time"))
        .with_columns(
            (pl.col("end_time") - pl.col("start_time")).alias("duration")
        )
    )

    if is_pandas:
        return trimmed.to_pandas().reset_index(drop=True)
    return trimmed


def merge_consecutive_labels(
    df: pl.DataFrame | pd.DataFrame,
    label_col: str = "label",
    start_col: str = "start_s",
    end_col: str = "end_s",
) -> pl.DataFrame | pd.DataFrame:
    """Merge rows with consecutive identical labels.

    Accepts and returns the same type (polars or pandas).

    Parameters
    ----------
    df : polars.DataFrame or pandas.DataFrame
        Must have *start_col*, *end_col*, and *label_col*.
    label_col : str
        Column containing labels.
    start_col : str
        Column containing start times.
    end_col : str
        Column containing end times.

    Returns
    -------
    DataFrame of the same type as *df*.
    """
    is_pandas = isinstance(df, pd.DataFrame)
    pl_df: pl.DataFrame = pl.from_pandas(df) if is_pandas else df  # type: ignore[assignment]

    if pl_df.is_empty():
        return df

    pl_df = pl_df.sort(start_col)
    pl_df = pl_df.with_columns(
        (pl.col(label_col) != pl.col(label_col).shift(1))
        .fill_null(True)
        .cum_sum()
        .alias("_group")
    )

    merged = (
        pl_df.group_by([label_col, "_group"], maintain_order=True)
        .agg(
            pl.col(start_col).first(),
            pl.col(end_col).last(),
        )
        .drop("_group")
        .sort(start_col)
    )

    if is_pandas:
        return merged.to_pandas().reset_index(drop=True)
    return merged
