import numpy as np


def spectro_plotter(
    spg,
    chan=None,
    f_range=None,
    t_range=None,
    yscale="linear",
    figsize=(35, 10),
    vmin=None,
    vmax=None,
    title="Title",
    ax=None,
    cmap="nipy_spectral",
):
    if f_range is not None:
        spg = spg.sel(frequency=f_range)

    try:
        # spg = spg.swap_dims({'datetime': 'time'})
        spg = spg.sel(channel=chan)
    except:
        print("Passing error - no channel dimension")

    freqs = spg.frequency
    spg_times = spg.datetime.values if "datetime" in spg else spg.time.values
    # freqs, spg_times, spg = dsps.trim_spectrogram(freqs, spg_times, spg, f_range, t_range)

    ax.pcolormesh(
        spg_times,
        freqs,
        np.log10(spg),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.5,
        shading="gouraud",
    )
    # ax.figure.colorbar(im)
    ax.set_yscale(yscale)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time")
    ax.set_title(title)

    if yscale == "log":
        ax.set_ylim(np.min(freqs[freqs > 0]), np.max(freqs))
    return ax
