from DRSpy.analysis import np, plt
from DRSpy.analysis import matplotlib

def create_figure(figsize=(12,7),row=1, col=1):
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots(row,col, figsize=figsize)
    return fig, ax

def add_plot(axis, X, Y, label, xerr=None, yerr=None, fmt=".", marker="o", errorline=1, errordash=2, markersize=6, linewidth=1, grid=False, legend=False,xlim=False, ylim=False, title=False, xlabel=False, ylabel=False,  **kwargs):
    axis.errorbar(X, Y, xerr=xerr, yerr=yerr, fmt=fmt, markersize=markersize, linewidth=linewidth, elinewidth=errorline, capsize=errordash, label=label, **kwargs)
    if grid: axis.grid(True)
    if legend: axis.legend()
    if xlim: axis.set_xlim(xlim)
    if ylim: axis.set_ylim(ylim)
    if xlabel: axis.set_xlabel(xlabel)
    if ylabel: axis.set_ylabel(ylabel)
    if title: axis.set_title(title)

def add_hist(axis, N, alpha=0.9, label=None, xlim=None, ylim=None, xlabel=None, ylabel=None, legend=False, title=False, **kwargs):
    axis.hist(N, alpha=alpha, label=label, **kwargs)
    if grid: axis.grid(True)
    if legend: axis.legend()
    if xlim: axis.set_xlim(xlim)
    if ylim: axis.set_ylim(ylim)
    if xlabel: axis.set_xlabel(xlabel)
    if ylabel: axis.set_ylabel(ylabel)
    if title: axis.set_title(title)
     

def save(fig, filename):
    fig.savefig(f"{filename}")
    
