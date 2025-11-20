import numpy as np
import polars as pl
from typing import Sequence, Union, Optional
import pandas as pd


def cube_to_df(
    a: np.ndarray,
    axis_labels: Sequence[str],
    *,
    order: str = "C",
    value_axis: Union[int, str, None] = None,
    value_name: Optional[str] = None,
    index_dtype: Optional[pl.DataType] = None,
    pandas: bool = False,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Convert an n-D numpy array into a long Polars DataFrame with one column per
    axis (optionally dropping one axis and using its label as the values column),
    plus a values column.

    Parameters
    ----------
    a : np.ndarray
        Input array of any shape.
    axis_labels : Sequence[str]
        One label per axis in the same order as a.shape.
        Example: a.shape == (C, D, S) and axis_labels == ("channel", "data", "source").
    order : {"C","F"}, default "C"
        Linearization order used to flatten `a` (also used for index math).
    value_axis : int | str | None, default None
        If given, this axis will NOT be output as an index column; instead, its label
        (or `value_name` if provided) will be used as the values column name.
        - If int, it is the axis index (supports negative).
        - If str, it must match one of `axis_labels`.
        - If None, all axes are emitted as index columns and the values column will be
          named `value_name or "value"`.
    value_name : str | None, default None
        Name for the values column. If `value_axis` is provided and `value_name` is None,
        the values column name defaults to the label of `value_axis`. If `value_axis`
        is None and `value_name` is None, defaults to "value".
    index_dtype : pl.DataType | None, default None
        Polars integer dtype for index columns. If None, chooses UInt32 if all axis sizes
        fit, else UInt64.

    Returns
    -------
    pl.DataFrame
        Columns:
          - One index column per axis (except `value_axis`, if provided), named from
            `axis_labels`, with integer indices starting at 0.
          - One values column containing the array values (name as described above).

    Notes
    -----
    • No Python loops over elements; just a single flatten and index arithmetic in Polars.
    • Uses integer arithmetic inside Polars for speed and low memory.
    • Column order preserves the axis order; if `value_axis` is set, the values column
      is placed where that axis would have been.
    """
    if not isinstance(a, np.ndarray):
        raise TypeError("`a` must be a numpy.ndarray")
    if a.ndim != len(axis_labels):
        raise ValueError(
            f"`axis_labels` length ({len(axis_labels)}) must match a.ndim ({a.ndim})"
        )
    if order not in ("C", "F"):
        raise ValueError("`order` must be 'C' or 'F'")

    ndim = a.ndim
    shape = list(a.shape)

    # Resolve value_axis to an integer index or None
    if isinstance(value_axis, str):
        try:
            value_axis = axis_labels.index(value_axis)
        except ValueError as e:
            raise ValueError(
                f"value_axis '{value_axis}' not found in axis_labels"
            ) from e
    if isinstance(value_axis, int):
        if value_axis < 0:
            value_axis += ndim
        if not (0 <= value_axis < ndim):
            raise ValueError(f"value_axis out of range for ndim={ndim}")

    # Decide value column name
    if value_axis is not None:
        val_col = value_name if value_name is not None else axis_labels[value_axis]
    else:
        val_col = value_name if value_name is not None else "value"

    # Choose index dtype
    if index_dtype is None:
        max_dim = max(shape, default=0)
        index_dtype = pl.UInt32 if max_dim <= np.iinfo(np.uint32).max else pl.UInt64

    # Flatten once; view if possible under requested order
    flat = a.reshape(-1, order=order)

    # Start DF with the values column
    df = pl.DataFrame({val_col: flat})

    # Row index expression for arithmetic inside Polars
    idx = pl.arange(0, pl.len(), dtype=pl.UInt64)

    # Precompute strides depending on order
    # For "C": coord[j] = (i // prod(shape[j+1:])) % shape[j]
    # For "F": coord[j] = (i // prod(shape[:j]))   % shape[j]
    if order == "C":
        suffix_prod = []
        running = 1
        for size in reversed(
            shape[1:] + [1]
        ):  # compute for j in 0..ndim-1; last stride=1
            suffix_prod.append(running)
            running *= size
        # suffix_prod currently holds [prod(shape[k+1:]) for k from end], reverse:
        suffix_prod = list(reversed(suffix_prod))
        strides = suffix_prod
    else:  # "F"
        strides = []
        running = 1
        for j in range(ndim):
            strides.append(running)  # prod(shape[:j])
            running *= shape[j]

    # Build index columns (skip value_axis if provided)
    idx_cols = []
    for j, label in enumerate(axis_labels):
        if value_axis is not None and j == value_axis:
            continue
        stride_j = strides[j]
        size_j = shape[j]
        expr = ((idx // stride_j) % size_j).cast(index_dtype).alias(label)
        idx_cols.append(expr)

    if idx_cols:
        df = df.with_columns(idx_cols)

    # Column order: keep the axis order, placing the value column where value_axis sits (if set)
    if value_axis is None:
        final_cols = list(axis_labels) + [val_col]
    else:
        final_cols = [lab for k, lab in enumerate(axis_labels) if k != value_axis]
        final_cols.insert(value_axis, val_col)

    if pandas:
        return df.to_pandas().reset_index(drop=True)
    else:
        return df.select(final_cols)
