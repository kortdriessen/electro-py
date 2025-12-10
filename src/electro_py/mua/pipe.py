import numpy as np
import os
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.core.recording_tools as rt
import electro_py as ep
import pandas as pd


def prepro_np_array_via_si(np_array: np.ndarray, fs: float) -> si.NumpyRecording:
    nchans = np_array.shape[0]
    if nchans != 16:
        raise ValueError("This function is only for 16 channel probes")
    chan_ids = np.arange(1, nchans + 1)
    si_raw = se.NumpyRecording(np_array.T, sampling_frequency=fs, channel_ids=chan_ids)
    si_filt = sp.bandpass_filter(si_raw, freq_min=300, freq_max=12000)
    si_filt_cmr = sp.common_reference(si_filt, reference="global", operator="median")

    # now we should set the offset and gain (to get to uV) to 0 and 1 respectively, since we've already converted the incoming array.
    num_chans = si_filt_cmr.get_num_channels()
    si_filt_cmr.set_property(
        key="gain_to_uV", values=np.ones(num_chans, dtype="float32")
    )
    si_filt_cmr.set_property(
        key="offset_to_uV", values=np.zeros(num_chans, dtype="float32")
    )

    # we will also set the noise levels here which will be used later in detection
    rand_kwargs = dict(
        chunk_size=int(si_filt_cmr.sampling_frequency * 2), num_chunks_per_segment=100
    )
    new_nlvs = rt.get_noise_levels(
        si_filt_cmr, method="mad", force_recompute=True, **rand_kwargs
    )
    if "noise_level_mad_scaled" not in si_filt_cmr.get_property_keys():
        si_filt_cmr.set_property(key="noise_level_mad_scaled", values=new_nlvs)
        si_filt_cmr.set_property(key="noise_level_scaled", values=new_nlvs)

    si_filt_cmr_with_probe = ep.tdt.utils.set_NNX_probe_info_on_SIrec(
        si_filt_cmr, probe_style="acr_style"
    )

    return si_filt_cmr_with_probe


def basic_peak_detection(
    rec, chunk_duration=1.0, n_jobs=64, progress_bar=True, threshold=4.0
):

    job_kwargs = dict(
        chunk_duration=chunk_duration, n_jobs=n_jobs, progress_bar=progress_bar
    )
    noise_levs = rec.get_property("noise_level_mad_scaled")
    peaks = detect_peaks(
        rec,
        method="by_channel",
        detect_threshold=threshold,
        noise_levels=noise_levs,
        peak_sign="neg",
        **job_kwargs,
    )
    dtype = [
        ("sample", "int64"),
        ("channel", "int64"),
        ("amplitude", "float64"),
        ("segment", "int64"),
    ]
    peaks_array = peaks.view(dtype).reshape(-1)
    peaks_df = pd.DataFrame(peaks_array)
    peaks_df.drop(columns="segment", inplace=True)
    peaks_df["time"] = peaks_df["sample"] / rec.get_sampling_frequency()
    return peaks_df
