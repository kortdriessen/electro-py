import matplotlib.pyplot as plt

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
    hypnogram : Hypnogram or any iterable of bouts with
        ``state``, ``start_time``, ``end_time`` attributes.
    ax : matplotlib.Axes, optional
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


def plot_basic_hypnogram(
    h,
    f,
    ax,
    size=(20, 1),
    xlim=None,
    style_path=None,
    single_tone=False,
    state_colors=None,
):
    state_colors = state_colors if state_colors is not None else {}
    if state_colors == {}:
        if single_tone:
            # GRAY SCALE
            state_colors["Wake"] = (1, "#333333")
            state_colors["NREM"] = (2, "#4f4f4f")
            state_colors["REM"] = (3, "#797979")

            # REDS
            state_colors["Wake"] = (1, "#6E2032")
            state_colors["NREM"] = (2, "#983F3F")
            state_colors["REM"] = (3, "#C88E87")

        else:
            state_colors["NREM"] = (2, "#4b71e3")
            state_colors["REM"] = (3, "#e34bde")
            state_colors["Wake"] = (1, "#4be350")

    # plt.rcdefaults()
    if style_path is not None:
        plt.style.use(style_path)
    plt.rcParams["axes.spines.left"] = False
    plt.rcParams["axes.spines.bottom"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["xtick.bottom"] = False
    # f, ax = plt.subplots(figsize=size)
    ax.set_ylim(0, 3)
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(h.start, h.end)

    epsilon = 0.001

    prev_end = None
    for bout in h.itertuples():
        if bout.state in state_colors:
            value, color = state_colors[bout.state]
        else:
            value, color = state_colors["Wake"]

        y_range = (value - 1, value)
        y_range = (y_range[0] / 3, y_range[1] / 3)

        start_time = bout.start_time
        end_time = bout.end_time

        # Ensure no overlap with previous bout
        if prev_end is not None and start_time <= prev_end:
            start_time = prev_end + epsilon

        ax.axvspan(
            start_time,
            end_time,
            ymin=y_range[0],
            ymax=y_range[1],
            color=color,
            alpha=1,
            linewidth=0,
        )
        prev_end = end_time

    ax.set_yticks([])

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return f, ax
