"""Unified hypnogram module backed by polars.

A hypnogram represents sleep/wake state annotations as a series of bouts,
each with a state label, start time, end time, and duration.
"""

from __future__ import annotations

import re
import warnings
from collections import namedtuple
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from datetime import time as dt_time
from enum import Enum, auto
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# TimeKind enum
# ---------------------------------------------------------------------------


class TimeKind(Enum):
    """Whether hypnogram times are float seconds or datetimes."""

    FLOAT = auto()
    DATETIME = auto()


# ---------------------------------------------------------------------------
# Duration parsing helper
# ---------------------------------------------------------------------------

_COMPONENT_RE = re.compile(
    r"\s*(?:(\d+)\s*[hH])?\s*(?:(\d+)\s*[mM])?\s*(?:(\d+(?:\.\d+)?)\s*[sS])?\s*"
)


def _parse_timedelta(value) -> timedelta:
    """Parse various duration representations into a `datetime.timedelta`.

    Accepted inputs:
      - ``timedelta`` objects
      - ``int`` / ``float`` (seconds)
      - ``numpy.timedelta64``
      - Strings: ``"HH:MM:SS"``, ``"MM:SS"``, ``"2h30m10s"``, ``"0S"``, or
        a plain number (seconds).
    """
    if isinstance(value, timedelta):
        return value
    if isinstance(value, (int, float)):
        return timedelta(seconds=float(value))
    if isinstance(value, np.timedelta64):
        return timedelta(seconds=float(value / np.timedelta64(1, "s")))
    if isinstance(value, str):
        # HH:MM:SS or MM:SS
        parts = value.split(":")
        if len(parts) == 3:
            return timedelta(
                hours=int(parts[0]), minutes=int(parts[1]), seconds=float(parts[2])
            )
        if len(parts) == 2:
            return timedelta(minutes=int(parts[0]), seconds=float(parts[1]))

        # Component format: 2h30m10s, 30m, 0S, etc.
        m = _COMPONENT_RE.fullmatch(value)
        if m and any(m.groups()):
            return timedelta(
                hours=int(m.group(1) or 0),
                minutes=int(m.group(2) or 0),
                seconds=float(m.group(3) or 0),
            )

        # Plain number (seconds)
        try:
            return timedelta(seconds=float(value))
        except ValueError:
            pass

        raise ValueError(f"Cannot parse '{value}' as a duration")
    raise TypeError(f"Expected timedelta, number, or string, got {type(value)}")


# ---------------------------------------------------------------------------
# Hypnogram class
# ---------------------------------------------------------------------------


class Hypnogram:
    """Unified hypnogram backed by a polars DataFrame.

    Auto-detects whether times are float (seconds) or ``datetime``.

    Required columns: ``state``, ``start_time``, ``end_time``.
    ``duration`` is always auto-computed from ``end_time - start_time``.
    Extra columns (e.g. ``note``) are preserved.

    Parameters
    ----------
    data : polars.DataFrame | pandas.DataFrame | dict
        The hypnogram data.  Pandas DataFrames are converted to polars
        automatically.
    """

    # ------------------------------------------------------------------ init

    def __init__(self, data):
        if isinstance(data, pl.DataFrame):
            df = data
        elif isinstance(data, dict):
            df = pl.DataFrame(data)
        else:
            # Try pandas conversion
            try:
                df = pl.from_pandas(data)
            except Exception as exc:
                raise TypeError(f"Cannot create Hypnogram from {type(data)}") from exc

        # Validate required columns
        required = {"state", "start_time", "end_time"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # If time columns are strings, try to parse
        if df.schema["start_time"] == pl.String or df.schema["start_time"] == pl.Utf8:
            try:
                df = df.with_columns(
                    pl.col("start_time").str.to_datetime(),
                    pl.col("end_time").str.to_datetime(),
                )
            except Exception:
                try:
                    df = df.with_columns(
                        pl.col("start_time").cast(pl.Float64),
                        pl.col("end_time").cast(pl.Float64),
                    )
                except Exception:
                    pass  # let _detect_time_kind raise

        self._time_kind = self._detect_time_kind(df)

        # Compute / recompute duration (single source of truth)
        df = df.with_columns(
            (pl.col("end_time") - pl.col("start_time")).alias("duration")
        )

        # Sort by start_time
        df = df.sort("start_time")
        self._df = df

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _detect_time_kind(df: pl.DataFrame) -> TimeKind:
        dtype = df.schema["start_time"]
        if dtype.is_numeric():
            return TimeKind.FLOAT
        if "Datetime" in str(dtype):
            return TimeKind.DATETIME
        raise TypeError(
            f"Cannot determine time kind from start_time dtype: {dtype}. "
            "Expected numeric (float/int) or Datetime."
        )

    @property
    def _zero(self):
        """A zero-duration value matching the hypnogram's time kind."""
        return 0.0 if self._time_kind == TimeKind.FLOAT else timedelta(0)

    def _duration_scalar(self, value):
        """Convert a user-supplied duration to the right scalar type."""
        if value is None:
            return None
        td = _parse_timedelta(value)
        return td.total_seconds() if self._time_kind == TimeKind.FLOAT else td

    # ----------------------------------------------------------- properties

    @property
    def time_kind(self) -> TimeKind:
        """Whether times are float seconds or datetimes."""
        return self._time_kind

    @property
    def df(self) -> pl.DataFrame:
        """Read-only access to the internal polars DataFrame."""
        return self._df

    @property
    def states(self) -> list[str]:
        """Unique states present, sorted alphabetically."""
        return self._df["state"].unique().sort().to_list()

    @property
    def total_duration(self):
        """Sum of all bout durations."""
        return self._df["duration"].sum()

    @property
    def start(self):
        """Earliest ``start_time``."""
        return self._df["start_time"].min()

    @property
    def end(self):
        """Latest ``end_time``."""
        return self._df["end_time"].max()

    # -------------------------------------------------------- dunder / iter

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f"Hypnogram({self._time_kind.name}, {len(self)} bouts)\n{self._df}"

    def __iter__(self):
        return self.itertuples()

    def itertuples(self):
        """Yield each bout as a named tuple."""
        fields = self._df.columns
        BoutTuple = namedtuple("Bout", fields)  # noqa: PYI024
        for row in self._df.iter_rows():
            yield BoutTuple(*row)

    # ------------------------------------------------------------- filtering

    def keep_states(self, states: list[str]) -> Hypnogram:
        """Return only bouts whose state is in *states*."""
        return Hypnogram(self._df.filter(pl.col("state").is_in(states)))

    def drop_states(self, states: list[str]) -> Hypnogram:
        """Return only bouts whose state is NOT in *states*."""
        return Hypnogram(self._df.filter(~pl.col("state").is_in(states)))

    def keep_longer(self, duration) -> Hypnogram:
        """Keep bouts longer than *duration*."""
        dur = self._duration_scalar(duration)
        return Hypnogram(self._df.filter(pl.col("duration") > dur))

    def keep_first(self, cumulative_duration, trim: bool = True) -> Hypnogram:
        """Keep bouts from the start until *cumulative_duration* is reached.

        Parameters
        ----------
        cumulative_duration
            Any value accepted by ``_parse_timedelta`` (e.g. ``"02:30:10"``).
        trim : bool
            If True (default), the boundary bout is trimmed so the total
            duration is exact.  If False, whole bouts are kept.
        """
        dur_value = self._duration_scalar(cumulative_duration)
        zero = self._zero

        df = self._df.with_columns(pl.col("duration").cum_sum().alias("_cumsum"))

        if trim:
            df = df.with_columns((pl.col("_cumsum") - dur_value).alias("_excess"))
            excess_df = df.filter(pl.col("_excess") > zero)
            if excess_df.is_empty():
                return self
            cutoff = excess_df.row(0, named=True)
            trim_until = cutoff["end_time"] - cutoff["_excess"]
            return self.trim(self.start, trim_until)
        else:
            return Hypnogram(df.filter(pl.col("_cumsum") <= dur_value).drop("_cumsum"))

    def keep_last(self, cumulative_duration, trim: bool = True) -> Hypnogram:
        """Keep bouts from the end until *cumulative_duration* is reached.

        Parameters
        ----------
        cumulative_duration
            Any value accepted by ``_parse_timedelta``.
        trim : bool
            If True (default), the boundary bout is trimmed exactly.
        """
        dur_value = self._duration_scalar(cumulative_duration)
        zero = self._zero

        rcumsum = self._df["duration"].reverse().cum_sum().reverse()
        df = self._df.with_columns(rcumsum.alias("_rcumsum"))

        if trim:
            df = df.with_columns((pl.col("_rcumsum") - dur_value).alias("_excess"))
            excess_df = df.filter(pl.col("_excess") > zero)
            if excess_df.is_empty():
                return self
            # Last row with excess (latest start_time, smallest excess)
            cutoff = excess_df.sort("start_time", descending=True).row(0, named=True)
            trim_from = cutoff["start_time"] + cutoff["_excess"]
            return self.trim(trim_from, self.end)
        else:
            return Hypnogram(
                df.filter(pl.col("_rcumsum") <= dur_value).drop("_rcumsum")
            )

    def trim(self, start, end) -> Hypnogram:
        """Trim the hypnogram to *[start, end]*, clamping bouts at boundaries.

        Bouts that fall entirely outside the range are dropped.
        """
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")

        df = self._df.with_columns(
            pl
            .when(pl.col("start_time") < start)
            .then(pl.lit(start))
            .otherwise(pl.col("start_time"))
            .alias("start_time"),
            pl
            .when(pl.col("end_time") > end)
            .then(pl.lit(end))
            .otherwise(pl.col("end_time"))
            .alias("end_time"),
        ).filter(pl.col("start_time") < pl.col("end_time"))
        return Hypnogram(df)

    def keep_between(self, start, end) -> Hypnogram:
        """Keep only bouts wholly within *[start, end]* (no clamping)."""
        return Hypnogram(
            self._df.filter(
                (pl.col("start_time") >= start) & (pl.col("end_time") <= end)
            )
        )

    def keep_between_time(self, start_time, end_time) -> Hypnogram:
        """Keep bouts where both start and end fall between two times of day.

        Only meaningful for datetime hypnograms.

        Parameters
        ----------
        start_time, end_time : str or datetime.time
            Times of day, e.g. ``"13:00:00"`` / ``"14:00:00"``.
        """
        if self._time_kind != TimeKind.DATETIME:
            raise ValueError("keep_between_time requires a datetime hypnogram")

        if isinstance(start_time, str):
            start_time = dt_time.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = dt_time.fromisoformat(end_time)

        s_ok = pl.col("start_time").dt.time()
        e_ok = pl.col("end_time").dt.time()

        if start_time <= end_time:
            mask = s_ok.is_between(start_time, end_time) & e_ok.is_between(
                start_time, end_time
            )
        else:
            # Wraps around midnight
            mask = ((s_ok >= start_time) | (s_ok <= end_time)) & (
                (e_ok >= start_time) | (e_ok <= end_time)
            )

        return Hypnogram(self._df.filter(mask))

    def merge_consecutive(self) -> Hypnogram:
        """Merge adjacent bouts that share the same state."""
        df = self._df.with_columns(
            (pl.col("state") != pl.col("state").shift(1))
            .fill_null(True)
            .cum_sum()
            .alias("_group")
        )

        agg_exprs = [
            pl.col("state").first(),
            pl.col("start_time").min(),
            pl.col("end_time").max(),
        ]
        skip = {"state", "start_time", "end_time", "duration", "_group"}
        for col in df.columns:
            if col not in skip:
                agg_exprs.append(pl.col(col).first())

        merged = (
            df.group_by("_group", maintain_order=True).agg(agg_exprs).drop("_group")
        )
        return Hypnogram(merged)

    def states_by_duration(self, states: list[str], duration=None) -> Hypnogram:
        """Keep bouts of *states*, optionally filtering by minimum *duration*."""
        h = self.keep_states(states)
        return h.keep_longer(duration) if duration is not None else h

    # -------------------------------------------------------------- queries

    def get_states(
        self,
        times,
        column: str = "state",
        default="no_state",
    ) -> np.ndarray:
        """Label each time with its hypnogram value from *column*.

        Uses binary search — O(n log m) where *n* = ``len(times)`` and
        *m* = number of bouts.

        Parameters
        ----------
        times : array-like
            The times to label (must match the hypnogram's time kind).
        column : str
            Column to pull labels from (default ``"state"``).
        default
            Value for times not covered by any bout.

        Returns
        -------
        numpy.ndarray
        """
        if column not in self._df.columns:
            raise ValueError(
                f"Column '{column}' not in hypnogram. Available: {self._df.columns}"
            )

        times_array = np.asarray(times)
        start_times = self._df["start_time"].to_numpy()
        end_times = self._df["end_time"].to_numpy()
        labels_col = self._df[column].to_numpy()

        labels = np.full(len(times_array), default, dtype=object)

        start_idx = np.searchsorted(start_times, times_array, side="right") - 1
        end_idx = np.searchsorted(end_times, times_array, side="left")

        valid = (
            (start_idx == end_idx) & (start_idx >= 0) & (start_idx < len(labels_col))
        )
        labels[valid] = labels_col[start_idx[valid]]
        return labels

    def covers_time(self, times) -> np.ndarray:
        """Boolean mask — ``True`` where *times* fall within any bout."""
        times_array = np.asarray(times)
        start_times = self._df["start_time"].to_numpy()
        end_times = self._df["end_time"].to_numpy()

        start_idx = np.searchsorted(start_times, times_array, side="right") - 1
        end_idx = np.searchsorted(end_times, times_array, side="left")

        return (
            (start_idx == end_idx) & (start_idx >= 0) & (start_idx < len(start_times))
        )

    def mask_times_by_state(self, times, states: list[str]) -> np.ndarray:
        """Boolean mask — ``True`` where *times* fall within *states*."""
        return self.keep_states(states).covers_time(times)

    def fractional_occupancy(self, ignore_gaps: bool = True) -> pl.DataFrame:
        """Fraction of total time spent in each state.

        Parameters
        ----------
        ignore_gaps : bool
            If True, unscored gaps are excluded from the total.
        """
        if self._time_kind == TimeKind.FLOAT:
            dur_expr = pl.col("duration")
            if ignore_gaps:
                total = self._df["duration"].sum()
            else:
                total = self._df["end_time"].max() - self._df["start_time"].min()
        else:
            dur_expr = pl.col("duration").dt.total_seconds()
            if ignore_gaps:
                total = self._df["duration"].dt.total_seconds().sum()
            else:
                span = self._df["end_time"].max() - self._df["start_time"].min()
                total = span.total_seconds()

        return (
            self._df
            .group_by("state")
            .agg(dur_expr.sum().alias("seconds"))
            .with_columns((pl.col("seconds") / total).alias("fraction"))
            .sort("fraction", descending=True)
        )

    # ----------------------------------------------------------- structural

    def get_gaps(self, tolerance=0) -> pl.DataFrame:
        """Return a DataFrame of unscored gaps between consecutive bouts.

        Parameters
        ----------
        tolerance
            Ignore gaps shorter than this.
        """
        tol = self._duration_scalar(tolerance)

        if len(self._df) < 2:
            return pl.DataFrame({
                "start_time": pl.Series([], dtype=self._df.schema["start_time"]),
                "end_time": pl.Series([], dtype=self._df.schema["end_time"]),
                "duration": pl.Series([], dtype=self._df.schema["duration"]),
            })

        return (
            self._df
            .with_columns(pl.col("start_time").shift(-1).alias("_next_start"))
            .filter(pl.col("_next_start").is_not_null())
            .select(
                pl.col("end_time").alias("start_time"),
                pl.col("_next_start").alias("end_time"),
            )
            .with_columns((pl.col("end_time") - pl.col("start_time")).alias("duration"))
            .filter(pl.col("duration") > tol)
        )

    def fill_gaps(self, tolerance=0, fill_state: str = "None") -> Hypnogram:
        """Fill unscored gaps with *fill_state*.

        Parameters
        ----------
        tolerance
            Ignore gaps shorter than this.
        fill_state : str
            State label used for filled gaps.
        """
        gaps = self.get_gaps(tolerance)
        if gaps.is_empty():
            return self

        gap_df = gaps.select(["start_time", "end_time"]).with_columns(
            pl.lit(fill_state).alias("state")
        )

        # Add extra columns as null
        extra = [
            c
            for c in self._df.columns
            if c not in ("state", "start_time", "end_time", "duration")
        ]
        for col in extra:
            gap_df = gap_df.with_columns(pl.lit(None).alias(col))

        select_cols = ["state", "start_time", "end_time"] + extra
        combined = pl.concat([self._df.select(select_cols), gap_df.select(select_cols)])
        return Hypnogram(combined)

    def get_consolidated(
        self,
        states: list[str],
        frac: float = 0.8,
        minimum_time=0,
        minimum_endpoint_bout_duration=0,
        maximum_antistate_bout_duration=None,
    ) -> list[Hypnogram]:
        """Find maximal periods of consolidated state occupancy.

        A period is *consolidated* if ``>= frac`` of its duration is spent
        in the target *states*, cumulative target-state time exceeds
        *minimum_time*, and no anti-state bout exceeds
        *maximum_antistate_bout_duration*.

        Parameters
        ----------
        states : list[str]
        frac : float
        minimum_time
            Minimum cumulative time in *states*.
        minimum_endpoint_bout_duration
            Endpoint bouts must be at least this long.
        maximum_antistate_bout_duration
            Reject periods containing any anti-state bout longer than this.

        Returns
        -------
        list[Hypnogram]
            Each element is a slice of this hypnogram covering one
            consolidated period.
        """
        min_time = self._duration_scalar(minimum_time)
        max_anti = self._duration_scalar(maximum_antistate_bout_duration)
        zero = self._zero

        # Identify endpoint bouts (target-state, long enough)
        ep_mask = pl.col("state").is_in(states)
        if minimum_endpoint_bout_duration:
            min_epb = self._duration_scalar(minimum_endpoint_bout_duration)
            ep_mask = ep_mask & (pl.col("duration") > min_epb)

        df_idx = self._df.with_row_index("_idx")
        ep_indices = df_idx.filter(ep_mask)["_idx"].to_list()

        if not ep_indices:
            return []

        k = ep_indices[0] - 1
        matches: list[Hypnogram] = []

        for i in ep_indices:
            if i <= k:
                continue
            for j in reversed(ep_indices):
                if j < max(i, k):
                    break

                sub = self._df.slice(i, j - i + 1)

                # Time in target states
                iso = sub.filter(pl.col("state").is_in(states))["duration"]
                time_in = iso.sum() if len(iso) > 0 else zero
                if self._time_kind == TimeKind.FLOAT:
                    time_in = max(time_in, 0.0)
                else:
                    time_in = max(time_in, timedelta(0))

                if time_in < min_time:
                    break

                # Anti-state bout duration check
                if max_anti is not None:
                    anti = sub.filter(~pl.col("state").is_in(states))
                    if len(anti) > 0 and anti["duration"].max() > max_anti:
                        continue

                # Fractional occupancy
                total = sub["end_time"].max() - sub["start_time"].min()
                if self._time_kind == TimeKind.FLOAT:
                    frac_val = time_in / total if total > 0 else 0
                else:
                    t_s = total.total_seconds()
                    i_s = time_in.total_seconds()
                    frac_val = i_s / t_s if t_s > 0 else 0

                if frac_val >= frac:
                    matches.append(Hypnogram(sub))
                    k = j
                    break

        return matches

    def reconcile(self, other: Hypnogram, how: str = "self") -> Hypnogram:
        """Reconcile this hypnogram with *other*.

        Parameters
        ----------
        other : Hypnogram
        how : ``"self"`` or ``"other"``
            Which hypnogram wins on conflicts.
        """
        if how == "self":
            return reconcile_hypnograms(self, other)
        elif how == "other":
            return reconcile_hypnograms(other, self)
        else:
            raise ValueError(f"Argument `how` should be 'self' or 'other'. Got {how}.")

    # ----------------------------------------------------------- conversion

    def as_float(self, reference_time=None) -> Hypnogram:
        """Convert a datetime hypnogram to float seconds.

        Parameters
        ----------
        reference_time : datetime, optional
            The zero point.  Defaults to the earliest ``start_time``.
        """
        if self._time_kind == TimeKind.FLOAT:
            return self

        if reference_time is None:
            reference_time = self._df["start_time"].min()
        elif isinstance(reference_time, str):
            reference_time = datetime.fromisoformat(reference_time)

        df = self._df.with_columns(
            ((pl.col("start_time") - reference_time).dt.total_seconds()).alias(
                "start_time"
            ),
            ((pl.col("end_time") - reference_time).dt.total_seconds()).alias(
                "end_time"
            ),
        ).drop("duration")

        return Hypnogram(df)

    def as_datetime(self, start_datetime) -> Hypnogram:
        """Convert a float-seconds hypnogram to datetime.

        Parameters
        ----------
        start_datetime : datetime or str
            The reference datetime corresponding to ``t = 0``.
        """
        if self._time_kind == TimeKind.DATETIME:
            return self

        if isinstance(start_datetime, str):
            start_datetime = datetime.fromisoformat(start_datetime)

        df = self._df.with_columns(
            (
                pl.lit(start_datetime)
                + pl.duration(
                    microseconds=(pl.col("start_time") * 1_000_000).cast(pl.Int64)
                )
            ).alias("start_time"),
            (
                pl.lit(start_datetime)
                + pl.duration(
                    microseconds=(pl.col("end_time") * 1_000_000).cast(pl.Int64)
                )
            ).alias("end_time"),
        ).drop("duration")

        return Hypnogram(df)

    def to_seconds(self, reference_time=None) -> Hypnogram:
        """Alias for :meth:`as_float`."""
        return self.as_float(reference_time)

    def to_polars(self) -> pl.DataFrame:
        """Return a copy of the internal DataFrame."""
        return self._df.clone()

    def to_pandas(self):
        """Return a pandas DataFrame."""
        return self._df.to_pandas()

    # -------------------------------------------------------- I/O — loaders

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        state_col: str = "state",
        start_col: str = "start_time",
        end_col: str = "end_time",
        note_col: str | None = None,
        separator: str = ",",
    ) -> Hypnogram:
        """Load from a CSV with configurable column names."""
        df = pl.read_csv(str(path), separator=separator, try_parse_dates=True)
        rename: dict[str, str] = {}
        if state_col != "state" and state_col in df.columns:
            rename[state_col] = "state"
        if start_col != "start_time" and start_col in df.columns:
            rename[start_col] = "start_time"
        if end_col != "end_time" and end_col in df.columns:
            rename[end_col] = "end_time"
        if note_col and note_col != "note" and note_col in df.columns:
            rename[note_col] = "note"
        if rename:
            df = df.rename(rename)
        return cls(df)

    @classmethod
    def from_htsv(cls, path: str | Path) -> Hypnogram | None:
        """Load a tab-separated ``.htsv`` file.

        Returns ``None`` if the file is empty.
        """
        path = Path(path)
        assert path.suffix == ".htsv", "File must use extension .htsv"
        try:
            df = pl.read_csv(
                str(path),
                separator="\t",
                has_header=True,
                try_parse_dates=True,
            )
        except Exception:
            return None
        return cls(df)

    @classmethod
    def from_visbrain(cls, path: str | Path) -> Hypnogram:
        """Load a Visbrain-formatted hypnogram."""
        df = pl.read_csv(
            str(path),
            separator="\t",
            has_header=False,
            new_columns=["state", "end_time"],
            comment_prefix="*",
        )
        df = df.with_columns(
            pl.col("end_time").shift(1).fill_null(0.0).alias("start_time")
        )
        return cls(df)

    @classmethod
    def from_spike2(cls, path: str | Path) -> Hypnogram:
        """Load a Spike2-formatted hypnogram."""
        df = pl.read_csv(
            str(path),
            separator="\t",
            has_header=False,
            skip_rows=22,
            new_columns=[
                "epoch",
                "start_time",
                "end_time",
                "state",
                "comment",
                "blank",
            ],
        ).select(["start_time", "end_time", "state"])
        return cls(df)

    @classmethod
    def from_sleepsign(cls, path: str | Path) -> Hypnogram:
        """Load a SleepSign hypnogram exported via the *trend* function."""
        df = pl.read_csv(
            str(path),
            separator="\t",
            has_header=False,
            skip_rows=19,
            try_parse_dates=True,
        )
        # First 3 columns: start_time (datetime), epoch (int), state (str)
        cols = df.columns[:3]
        df = df.select(cols).rename({
            cols[0]: "start_time",
            cols[1]: "epoch",
            cols[2]: "state",
        })

        assert df["epoch"][0] == 0, (
            "First epoch is not #0. Unexpected number of header lines?"
        )

        # Convert datetime column to float seconds from start
        t0 = df["start_time"][0]
        if "Datetime" in str(df.schema["start_time"]):
            df = df.with_columns(
                ((pl.col("start_time") - t0).dt.total_seconds()).alias("start_time")
            )

        # Verify uniform epoch length
        epoch_diffs = df["start_time"].diff().drop_nulls()
        assert epoch_diffs.n_unique() == 1, "Epochs are not all the same length"
        epoch_length = epoch_diffs[0]

        df = df.with_columns(
            (pl.col("start_time") + epoch_length).alias("end_time")
        ).select(["state", "start_time", "end_time"])

        return cls(df)

    @classmethod
    def from_loupe(cls, path: str | Path) -> Hypnogram:
        """Load a loupe-exported CSV (``start_s``, ``end_s``, ``label``, ``note``)."""
        return cls.from_csv(
            path,
            state_col="label",
            start_col="start_s",
            end_col="end_s",
            note_col="note",
        )

    @classmethod
    def dummy(
        cls,
        start_time: float = 0.0,
        end_time: float = float("inf"),
    ) -> Hypnogram:
        """Create a single-bout placeholder hypnogram."""
        return cls(
            pl.DataFrame({
                "state": ["None"],
                "start_time": [float(start_time)],
                "end_time": [float(end_time)],
            })
        )

    # -------------------------------------------------------- I/O — writers

    def write_csv(self, path: str | Path, separator: str = ",") -> None:
        """Write to CSV."""
        self._df.write_csv(str(path), separator=separator)

    def write_htsv(self, path: str | Path) -> None:
        """Write as a tab-separated ``.htsv`` file."""
        path = Path(path)
        assert path.suffix == ".htsv", "File must use extension .htsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._df.write_csv(str(path), separator="\t")

    def write_visbrain(self, path: str | Path) -> None:
        """Write in Visbrain format (float times only)."""
        if self._time_kind != TimeKind.FLOAT:
            raise ValueError(
                "Visbrain format requires float times. Use .as_float() first."
            )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._df.select(["state", "end_time"]).write_csv(
            str(path), separator="\t", include_header=False
        )

    def write_loupe(self, path: str | Path) -> None:
        """Write in loupe format (float times only)."""
        if self._time_kind != TimeKind.FLOAT:
            raise ValueError(
                "Loupe format requires float times. Use .as_float() first."
            )
        df = self._df.select(
            [
                pl.col("start_time").alias("start_s"),
                pl.col("end_time").alias("end_s"),
                pl.col("state").alias("label"),
            ]
            + [
                c
                for c in self._df.columns
                if c not in ("start_time", "end_time", "state")
            ]
        )
        df.write_csv(str(path), separator=",", include_header=True)


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


def reconcile_hypnograms(h1: Hypnogram, h2: Hypnogram) -> Hypnogram:
    """Combine two hypnograms, resolving conflicts in favour of *h1*.

    Where *h1* and *h2* overlap, *h1* wins: overlapping *h2* intervals are
    truncated, split, or dropped as needed.
    """
    core = ["state", "start_time", "end_time"]
    h1_df = h1.df.select(core)
    h2_df = h2.df.select(core).clone()

    for row in h1_df.iter_rows(named=True):
        r_start, r_end = row["start_time"], row["end_time"]

        # Drop identical intervals
        h2_df = h2_df.filter(
            ~((pl.col("start_time") == r_start) & (pl.col("end_time") == r_end))
        )

        # Drop sub-intervals wholly contained by this h1 row
        h2_df = h2_df.filter(
            ~((pl.col("start_time") >= r_start) & (pl.col("end_time") <= r_end))
        )

        # Split super-intervals that wholly contain this h1 row
        super_mask = (pl.col("start_time") <= r_start) & (pl.col("end_time") >= r_end)
        super_df = h2_df.filter(super_mask)
        if len(super_df) > 0:
            assert len(super_df) == 1, (
                "Multiple h2 intervals wholly contain the same h1 interval"
            )
            sup = super_df.row(0, named=True)
            h2_df = h2_df.filter(~super_mask)
            fragments: list[pl.DataFrame] = []
            if sup["start_time"] < r_start:
                fragments.append(
                    pl.DataFrame({
                        "state": [sup["state"]],
                        "start_time": [sup["start_time"]],
                        "end_time": [r_start],
                    })
                )
            if sup["end_time"] > r_end:
                fragments.append(
                    pl.DataFrame({
                        "state": [sup["state"]],
                        "start_time": [r_end],
                        "end_time": [sup["end_time"]],
                    })
                )
            if fragments:
                h2_df = pl.concat([h2_df.select(core)] + fragments)

        # Truncate left overlaps
        left_mask = (
            (pl.col("start_time") < r_start)
            & (pl.col("end_time") > r_start)
            & (pl.col("end_time") < r_end)
        )
        if h2_df.filter(left_mask).height > 0:
            h2_df = h2_df.with_columns(
                pl
                .when(left_mask)
                .then(pl.lit(r_start))
                .otherwise(pl.col("end_time"))
                .alias("end_time")
            )

        # Truncate right overlaps
        right_mask = (
            (pl.col("start_time") > r_start)
            & (pl.col("start_time") < r_end)
            & (pl.col("end_time") > r_end)
        )
        if h2_df.filter(right_mask).height > 0:
            h2_df = h2_df.with_columns(
                pl
                .when(right_mask)
                .then(pl.lit(r_end))
                .otherwise(pl.col("start_time"))
                .alias("start_time")
            )

    combined = pl.concat([h1_df, h2_df.select(core)]).sort("start_time")
    return Hypnogram(combined)


# ---------------------------------------------------------------------------
# Multi-hypnogram unification
# ---------------------------------------------------------------------------

ConflictRecord = namedtuple(
    "ConflictRecord",
    [
        "winner_label",
        "loser_label",
        "start_time",
        "end_time",
        "winner_state",
        "loser_state",
    ],
)


def _find_overlap_conflicts(
    winner: Hypnogram,
    loser: Hypnogram,
    winner_label: str,
    loser_label: str,
) -> list[ConflictRecord]:
    """Find overlapping regions where *winner* and *loser* disagree on state."""
    w_rows = winner.df.select("state", "start_time", "end_time").to_dicts()
    l_rows = loser.df.select("state", "start_time", "end_time").to_dicts()

    conflicts: list[ConflictRecord] = []
    j = 0
    for wr in w_rows:
        # advance past loser bouts that end before this winner bout starts
        while j < len(l_rows) and l_rows[j]["end_time"] <= wr["start_time"]:
            j += 1
        for k in range(j, len(l_rows)):
            lr = l_rows[k]
            if lr["start_time"] >= wr["end_time"]:
                break
            # overlap exists
            if wr["state"] != lr["state"]:
                conflicts.append(
                    ConflictRecord(
                        winner_label=winner_label,
                        loser_label=loser_label,
                        start_time=max(wr["start_time"], lr["start_time"]),
                        end_time=min(wr["end_time"], lr["end_time"]),
                        winner_state=wr["state"],
                        loser_state=lr["state"],
                    )
                )
    return conflicts


def _resolve_sort_key(
    sort_key: Callable[[Path], object] | str,
) -> Callable[[Path], object]:
    """Turn a sort_key name or callable into a key function for sorted()."""
    if sort_key == "name":
        return lambda p: p.name
    if sort_key == "mtime":
        return lambda p: p.stat().st_mtime
    if not callable(sort_key):
        raise ValueError(
            f"sort_key must be 'name', 'mtime', or a callable, got {sort_key!r}"
        )
    return sort_key  # type: ignore[return-value]


def unify_hypnograms(
    hypnograms: Sequence[tuple[str, Hypnogram]],
    *,
    priority: str = "last",
    warn_conflicts: bool = True,
) -> tuple[Hypnogram, list[ConflictRecord]]:
    """Sequentially reconcile multiple labelled hypnograms into one.

    Parameters
    ----------
    hypnograms : sequence of (label, Hypnogram) pairs
        The hypnograms to merge, **in the desired merge order**.
        The order determines conflict resolution: with ``priority="last"``,
        later entries override earlier ones in overlapping regions.
    priority : ``"first"`` or ``"last"``
        ``"last"`` (default): later hypnograms win conflicts (natural for
        versioned files where later versions are corrections).
        ``"first"``: earlier hypnograms win conflicts.
    warn_conflicts : bool
        If ``True``, emit a ``UserWarning`` for each pair that has
        overlapping regions with different state labels.

    Returns
    -------
    unified : Hypnogram
        The merged result (consecutive identical states are merged).
    conflicts : list[ConflictRecord]
        A record of every conflict that was resolved, including which
        hypnogram won and what the competing states were.
    """
    if not hypnograms:
        raise ValueError("No hypnograms to unify.")
    if len(hypnograms) == 1:
        return hypnograms[0][1], []

    if priority not in ("first", "last"):
        raise ValueError(f"priority must be 'first' or 'last', got {priority!r}")

    all_conflicts: list[ConflictRecord] = []
    _label_acc, h_acc = hypnograms[0]

    for label_new, h_new in hypnograms[1:]:
        if priority == "last":
            winner_h, loser_h = h_new, h_acc
            winner_label, loser_label = label_new, _label_acc
        else:
            winner_h, loser_h = h_acc, h_new
            winner_label, loser_label = _label_acc, label_new

        pair_conflicts = _find_overlap_conflicts(
            winner_h,
            loser_h,
            winner_label,
            loser_label,
        )
        all_conflicts.extend(pair_conflicts)

        if warn_conflicts and pair_conflicts:
            warnings.warn(
                f"Resolved {len(pair_conflicts)} conflict(s) — "
                f"'{winner_label}' overrides '{loser_label}'.",
                stacklevel=2,
            )

        h_acc = reconcile_hypnograms(winner_h, loser_h)
        _label_acc = f"{_label_acc}+{label_new}"

    return h_acc.merge_consecutive(), all_conflicts


def unify_hypno_directory(
    directory: str | Path,
    *,
    loader: Callable[[Path], Hypnogram] | None = None,
    glob_pattern: str = "*.csv",
    sort_key: Callable[[Path], object] | str = "name",
    priority: str = "last",
    warn_conflicts: bool = True,
) -> tuple[Hypnogram, list[ConflictRecord]]:
    """Load and unify all hypnogram files in a directory.

    Parameters
    ----------
    directory : str or Path
        Directory containing hypnogram files.
    loader : callable, optional
        ``Path -> Hypnogram``.  Defaults to ``Hypnogram.from_loupe``.
    glob_pattern : str
        Glob pattern for file discovery.  Default ``"*.csv"``.
    sort_key : callable or ``"name"`` or ``"mtime"``
        How to order the discovered files before merging.
        ``"name"`` (default): lexicographic by filename.
        ``"mtime"``: by file modification time.
        A callable: ``sort_key(path) -> comparable``.
    priority : ``"first"`` or ``"last"``
        Which end of the sorted list wins conflicts.
        Default ``"last"`` (later files override earlier ones).
    warn_conflicts : bool
        If ``True``, emit warnings about resolved conflicts.

    Returns
    -------
    unified : Hypnogram
    conflicts : list[ConflictRecord]
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if loader is None:
        loader = Hypnogram.from_loupe

    key_fn = _resolve_sort_key(sort_key)
    files = sorted(directory.glob(glob_pattern), key=key_fn)
    if not files:
        raise ValueError(f"No files matching '{glob_pattern}' in {directory}")

    hypnograms = [(f.name, loader(f)) for f in files]
    return unify_hypnograms(
        hypnograms,
        priority=priority,
        warn_conflicts=warn_conflicts,
    )


def get_separated_wake_hypnogram(
    qwk_intervals: list | np.ndarray,
    awk_intervals: list | np.ndarray,
) -> Hypnogram:
    """Create a hypnogram from quiet-wake and active-wake interval lists.

    Parameters
    ----------
    qwk_intervals, awk_intervals : array-like of shape (n, 2)
        Each row is ``[start_time, end_time]``.
    """
    qwk = np.asarray(qwk_intervals)
    awk = np.asarray(awk_intervals)

    qwk_df = pl.DataFrame({
        "state": ["qWk"] * len(qwk),
        "start_time": qwk[:, 0].tolist(),
        "end_time": qwk[:, 1].tolist(),
    })
    awk_df = pl.DataFrame({
        "state": ["aWk"] * len(awk),
        "start_time": awk[:, 0].tolist(),
        "end_time": awk[:, 1].tolist(),
    })

    return Hypnogram(pl.concat([qwk_df, awk_df]))
