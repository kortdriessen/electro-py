import numpy as np
import pandas as pd
import tdt
import xarray as xr


def load_tdt_block(path: str, t1: int = 0, t2: int = 0):
    """Loads everything in a TDT block between t1 and t2

    Parameters
    ----------
    path : str
        path to TDT block
    t1 : int, optional
        start time in seconds, by default 0
    t2 : int, optional
        end time in seconds, by default 0
    """
    return tdt.read_block(path, t1=t1, t2=t2)


def load_sev_store(
    path: str,
    t1: int = 0,
    t2: int = 0,
    channel: int = 0,
    store: str = None,
    start_date: pd.Timestamp = None,
):
    data = tdt.read_block(path, channel=channel, store=store, t1=t1, t2=t2)
    store = data.streams[store]
    info = data.info
    datax = sev_to_xarray(info, store, start_date=start_date)
    return datax


# Functions for loading TDT SEV-stores


def sev_to_xarray(info, store, start_date=None):
    """Convert a single stream store to xarray format.

    Paramters:
    ----------
    info: tdt.StructType
        The `info` field of a tdt `blk` struct, as returned by `_load_stream_store`.
    store: tdt.StructType
        The store field of a tdt `blk.streams` struct, as returned by `_load_stream_store`.

    Returns:
    --------
    data: xr.DataArray (n_samples, n_channels)
        Values: The data, in microvolts.
        Attrs: units, fs
        Name: The store name
    """
    try:
        n_channels, n_samples = store.data.shape
    except ValueError:
        n_samples = store.data.shape[0]

    time = np.arange(0, n_samples) / store.fs + store.start_time
    timedelta = pd.to_timedelta(time, "s")
    datetime = pd.to_datetime(info.start_date) + timedelta

    if start_date is not None:
        datetime = start_date + timedelta

    volts_to_microvolts = 1e6
    # had to add this try-except because stupid TDT defines 'channels' for EEG/LFP,
    # but 'channel' for EMG.
    try:
        data = xr.DataArray(
            store.data.T * volts_to_microvolts,
            dims=("time", "channel"),
            coords={
                "time": time,
                "channel": store.channels,
                "timedelta": ("time", timedelta),
                "datetime": ("time", datetime),
            },
            name=store.name,
        )
    except:
        data = xr.DataArray(
            store.data.T * volts_to_microvolts,
            dims=("time"),
            coords={
                "time": time,
                "timedelta": ("time", timedelta),
                "datetime": ("time", datetime),
            },
            name=store.name,
        )
    data.attrs["units"] = "uV"
    data.attrs["fs"] = store.fs

    return data


def get_data(
    block_path,
    store="",
    t1=0,
    t2=0,
    channel=0,
    pandas=False,
    sel_chan=False,
    start_date=None,
    dt=True,
):

    data = load_sev_store(
        block_path, t1=t1, t2=t2, channel=channel, store=store, start_date=start_date
    )
    if dt:
        try:
            data = data.swap_dims({"time": "datetime"})
        except ValueError:
            print("Passing ValueError on dimension swap in get_data")
    if pandas:
        data = data.to_dataframe().drop(labels=["time", "timedelta"], axis=1)
    if sel_chan:
        data = data.sel(channel=sel_chan)
    return data
