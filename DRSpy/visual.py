from DRSpy.analysis import np, plt
from DRSpy.analysis import matplotlib

from abc import ABC, abstractmethod


class Figure(ABC):
    """
    The one to rule then all. Basic representation of Figure classes.

    :param row: Number of rows in figure, defaults to 1
    :type row: 1, optional
    :param col: Number of columns in figure, defaults to 1
    :type col: 1, optional
    :param figsize: Figure size, defaults to (12,7)
    :type figsize: (int, int), optional
    """

    def __init__(self, row=1, col=1, figsize=(12, 3)):
        """
        Creating figure and axis object (matplotlib)
        """
        self.fig, self.ax = plt.subplots(row, col, figsize=figsize)
        # self.

    def savefig(self, filename, fig_extension="png"):
        """
        Save figure to file and clear axis.

        :param filename: Filename of saved figure
        :type filename: str
        :param fig_extension: File extension  of saved figure, defaults to ".png"
        :type fig_extension: str, optional
        """
        self.fig.savefig(f"{filename}.{fig_extension}")
        plt.close()

    @abstractmethod
    def add_plot(self, axis, **kwargs):
        """
        Add new plot to figure.

        :param axis: Axis to draw plot.
        :type: numpy.ndarray
        :param kwargs: Configuration parameters specific for child classes.
        """
        pass


class Regular(Figure):
    """
    Create regular
    """

    def __init__(self):
        pass


def create_figure(figsize=(12, 7), row=1, col=1):
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots(row, col, figsize=figsize)
    return fig, ax


def add_plot(
    axis,
    X,
    Y,
    label,
    xerr=None,
    yerr=None,
    fmt=".",
    marker="o",
    errorline=1,
    errordash=2,
    markersize=6,
    linewidth=1,
    grid=False,
    legend=False,
    xlim=False,
    ylim=False,
    title=False,
    xlabel=False,
    ylabel=False,
    **kwargs,
):
    axis.errorbar(
        X,
        Y,
        xerr=xerr,
        yerr=yerr,
        fmt=fmt,
        markersize=markersize,
        linewidth=linewidth,
        elinewidth=errorline,
        capsize=errordash,
        label=label,
        **kwargs,
    )
    if grid:
        axis.grid(True)
    if legend:
        axis.legend()
    if xlim:
        axis.set_xlim(xlim)
    if ylim:
        axis.set_ylim(ylim)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)
    if title:
        axis.set_title(title)


def add_hist(
    axis,
    N,
    alpha=0.9,
    label=None,
    xlim=None,
    ylim=None,
    xlabel=None,
    ylabel=None,
    legend=False,
    title=False,
    **kwargs,
):
    axis.hist(N, alpha=alpha, label=label, **kwargs)
    if grid:
        axis.grid(True)
    if legend:
        axis.legend()
    if xlim:
        axis.set_xlim(xlim)
    if ylim:
        axis.set_ylim(ylim)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)
    if title:
        axis.set_title(title)


def save(fig, filename):
    fig.savefig(f"{filename}")
