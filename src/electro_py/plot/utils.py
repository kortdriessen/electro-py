hypno_colors = {
    "Wake": "#62e273",
    "Brief-Arousal": "chartreuse",
    "Transition-to-NREM": "lightskyblue",
    "Transition-to-Wake": "palegreen",
    "NREM": "cornflowerblue",
    "Transition-to-REM": "plum",
    "REM": "magenta",
    "Transition": "grey",
    "Art": "crimson",
    "Wake-art": "crimson",
    "Unsure": "white",
    "Good": "lime",
    # "Wake-Good": "turquoise",
    "Wake-Good": "#62e273",
    "Sort-Exclude": "coral",
    "unclear": "yellow",
}


def shade_hypno_for_me(hypnogram, ax=None, xlim=None, ymin=0, ymax=1, alpha=0.15):
    """Shade plot background using hypnogram state.

    Parameters
    ----------
    hypnogram: pandas.DataFrame
        Hypnogram with with state, start_time, end_time columns.
    ax: matplotlib.Axes, optional
        An axes upon which to plot.
    """
    xlim = ax.get_xlim() if (ax and not xlim) else xlim

    for bout in hypnogram.itertuples():
        ax.axvspan(
            bout.start_time,
            bout.end_time,
            ymin=ymin,
            ymax=ymax,
            alpha=alpha,
            color=hypno_colors[bout.state],
            zorder=1000,
            ec="none",
        )

    ax.set_xlim(xlim)
    return ax
