# Functions for plotting raw data
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def atomic_lfp(
    lfp_data,
    times=None,
    sems=None,
    hspace=-0.6,
    figsize=(24, 8),
    color="blue",
    line_alpha=1,
    linewidth=2,
):
    """Quick plot of raw LFP lfp_data traces

    Parameters
    ----------
    lfp_data : np.ndarray
        Raw lfp_data to plot, of shape (n_channels, n_samples)
    color : str, optional
        Color of the traces, by default blue
    hspace : float, optional
        Space between traces, by default -0.6
    figsize : tuple, optional
        Size of the figure, by default (24, 8)
    sems : np.ndarray, optional
        Semitransparent error bars, of shape (n_channels, n_samples)
    """
    plt.rcParams["axes.spines.bottom"] = False
    plt.rcParams["axes.spines.left"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.grid"] = False
    plt.rcParams["xtick.major.size"] = 0
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "None"
    plt.rcParams["xtick.labelbottom"] = False
    plt.rcParams["ytick.labelleft"] = False
    plt.rcParams["ytick.left"] = False
    plt.rcParams["xtick.bottom"] = False

    if times is None:
        times = np.arange(lfp_data.shape[1])
    f, ax = plt.subplots(lfp_data.shape[0], 1, figsize=figsize)
    for i in range(lfp_data.shape[0]):
        ax[i].plot(
            times, lfp_data[i, :], color=color, alpha=line_alpha, linewidth=linewidth
        )
        ax[i].set_xlim(times[0], times[-1])
        if sems is not None:
            ax[i].fill_between(
                times,
                lfp_data[i, :] - sems[i, :],
                lfp_data[i, :] + sems[i, :],
                color=color,
                alpha=0.2,
            )
    plt.subplots_adjust(hspace=hspace)
    return f, ax


def atomic_raster(
    mua_df,
    xname="datetime",
    yname="negchan",
    color="blue",
    alpha=0.7,
    s=60,
    figsize=(24, 8),
):
    """Plot a raster plot of MUA data

    Parameters
    ----------
    mua_df : pl.DataFrame or pd.DataFrame
        the MUA data to plot, should have columns according to xname and yname
    xname : str, optional
        Column name for the x-axis, by default 'datetime'
    yname : str, optional
        Column name for the y-axis, by default 'negchan'
    color : str, optional
        Color of the raster plot, by default 'blue'
    figsize : tuple, optional
        Size of the figure, by default (24, 8)

    Returns
    -------
    f, ax : tuple
        Figure and axes objects
    """

    plt.rcParams["axes.spines.bottom"] = False
    plt.rcParams["axes.spines.left"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.grid"] = False
    plt.rcParams["xtick.major.size"] = 0
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "None"

    assert xname in mua_df.columns, f"xname {xname} not in mua_df"
    assert yname in mua_df.columns, f"yname {yname} not in mua_df"

    plt.rcdefaults()
    f, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(
        mua_df, x=xname, y=yname, linewidth=0, alpha=alpha, s=s, ax=ax, color=color
    )
    # ax.set_xlim()
    # ax.set_yticks([])
    # ax.set_xticks([])
    plt.tight_layout()
    return f, ax


def plot_lfp_mua_combined(
    lfp_data,
    mua_df,
    times,
    lfp_color="blue",
    mua_color="blue",
    mua_xname="datetime",
    mua_yname="negchan",
    figsize=(36, 14),
    lfp_subplot_hspace=-0.4,
    mua_height_multiplier=0.2,
    lw=1.5,
    rsz=30,
    lfp_alpha=1,
    spike_alpha=0.7,
):
    """Plots LFP traces (each channel in its own subplot, stacked)
    above an MUA raster plot, in a single figure.

    Parameters
    ----------
    lfp_data : np.ndarray
        Raw LFP data to plot, of shape (n_channels, n_samples).
    mua_df : pl.DataFrame or pd.DataFrame
        MUA data to plot. Needs columns specified by mua_xname and mua_yname.
    times : np.ndarray
        Time vector for the LFP data, same length as lfp_data.shape[1].
    lfp_color : str, optional
        Color of the LFP traces.
    mua_color : str, optional
        Color of the MUA raster plot.
    mua_xname : str, optional
        Column name in mua_df for MUA event times.
    mua_yname : str, optional
        Column name in mua_df for MUA event channels/depths.
    figsize : tuple, optional
        Size of the figure.
    lfp_subplot_hspace : float, optional
        Vertical spacing between all subplots (LFP channels and MUA plot).
        Negative values (e.g., -0.5 or -0.6) can make LFP channel plots overlap,
        similar to the original base_trace's hspace effect. Default is 0.0.

    Returns
    -------
    f : matplotlib.figure.Figure
        The created Matplotlib figure.
    axs : np.ndarray of matplotlib.axes.Axes
        Array of axes. axs[0:-1] are LFP channel axes, axs[-1] is MUA axis.
    """

    num_lfp_channels = lfp_data.shape[0]
    if num_lfp_channels == 0:
        print("No LFP data to plot. Plotting MUA only.")
        f, mua_ax = plt.subplots(
            figsize=(
                figsize[0],
                figsize[1] / 3.0 if figsize[1] and figsize[1] > 0 else 5,
            )
        )
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "None"
        sns.scatterplot(
            data=mua_df,
            x=mua_xname,
            y=mua_yname,
            linewidth=0,
            alpha=0.7,
            s=30,
            ax=mua_ax,
            color=mua_color,
        )
        mua_ax.set_yticks([])
        mua_ax.set_yticklabels([])
        mua_ax.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            labelbottom=True,
            labelsize=10,
        )
        mua_ax.set_xlabel(str(mua_xname).capitalize(), fontsize=12)
        mua_ax.spines["top"].set_visible(False)
        mua_ax.spines["right"].set_visible(False)
        mua_ax.spines["left"].set_visible(False)
        plt.tight_layout()
        return f, np.array([mua_ax])

    plt.rcParams["axes.spines.left"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.grid"] = False
    # plt.rcParams['xtick.major.size'] = 0
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "None"
    plt.rcParams["ytick.left"] = False

    total_rows = num_lfp_channels + 1
    f = plt.figure(figsize=figsize)

    mua_height_ratio = max(1, num_lfp_channels * mua_height_multiplier)
    height_ratios = [1] * num_lfp_channels + [mua_height_ratio]

    gs = f.add_gridspec(
        total_rows, 1, height_ratios=height_ratios, hspace=lfp_subplot_hspace
    )

    axs = np.empty(total_rows, dtype=object)

    for i in range(num_lfp_channels):
        axs[i] = f.add_subplot(gs[i, 0], sharex=axs[0] if i > 0 else None)
        axs[i].plot(
            times, lfp_data[i, :], color=lfp_color, alpha=lfp_alpha, linewidth=lw
        )
        if i == 0:  # Set xlim only for the first plot, others will share
            axs[i].set_xlim(times[0], times[-1])
        axs[i].set_yticks([])
        axs[i].set_yticklabels([])
        axs[i].spines["bottom"].set_visible(False)
        axs[i].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

    axs[-1] = f.add_subplot(gs[-1, 0], sharex=axs[0])
    mua_ax = axs[-1]
    sns.scatterplot(
        data=mua_df,
        x=mua_xname,
        y=mua_yname,
        linewidth=0,
        alpha=spike_alpha,
        s=rsz,
        ax=mua_ax,
        color=mua_color,
    )
    mua_ax.set_yticks([])
    mua_ax.set_yticklabels([])
    mua_ax.tick_params(
        axis="x", which="both", bottom=True, top=False, labelbottom=True, labelsize=14
    )
    mua_ax.set_xlabel(str(mua_xname).capitalize(), fontsize=12)
    mua_ax.spines["bottom"].set_visible(True)

    # Adjust the position of the last axes to move it further down
    pos = mua_ax.get_position()  # Get the current position
    mua_ax.set_position([pos.x0, pos.y0 - 0.04, pos.width, pos.height])

    return f, axs
