import os
import numpy as np

# import matplotlib.pyplot as plt
import pandas as pd
from sys import argv
from DRSpy.visual import *
from scipy.optimize import curve_fit
from DRSpy.optimize import inspector_calib, fit
from DRSpy.data_struct import log


class InSpector:
    """
    Spectrum analyzer for InSpector1000.

    :param plot_ext: extension for generated plots, defaults to png
    :type plot_ext: str
    """

    def __init__(self, plot_ext="png"):
        log("--> Initialize InSpector-100 module")
        self.plot_ext = plot_ext
        self.calibration = 1

    def calibrate(self, calibfile):
        """
        Calibration method. Create linear corelation between peaks in .tka and real decay energy.
        Example of input calibration file example.txt:
        Channel,E,Source
        102,511,Na22
        251,1274,Na22

        :param calibfile: input calib .csv
        :type calibfile: str
        """
        log("---> Starting calibration: ", wait=True)
        fig, ax = create_figure()
        try:
            calib_data = pd.read_csv(calibfile)
            calib_group = calib_data.groupby("Source")
            # fig, ax = plt.subplots()
            for source, data in calib_group:
                add_plot(
                    ax, data["Channel"], data["E"], source, markersize=12, legend=True
                )
            x, y, p, pc = fit(inspector_calib, calib_data["Channel"], calib_data["E"])
            add_plot(
                ax,
                x,
                y,
                "Calib_fit",
                fmt="--",
                legend=True,
                xlabel="Channel",
                ylabel="Energy [keV]",
                grid=True,
            )
            save(fig, f"InSpector_calibration.{self.plot_ext}")
        except Exception as e:
            log("Failed", "red")
            log(f"\n{e}")
        else:
            log("Done", "green")
            log("--> Calculated a param: ", wait=True)
            log(f"a = {round(p[0],3)} ± {round(pc[0,0], 3)} ", "green")
            self.calibration = p[0]

    def get_spectrum(self, paths, normalize=False, calibration=True):
        """
        Get spectrum from TKA files.

        :param paths: List of paths to tka files
        :type paths: list(str)
        :param normalize: Normalize generated spectrum, avalible methods: minmax, defaults to False
        :type normalize: str
        :param calibration: enable or disable calibration, defaults to True
        :type calibration: bool
        """
        _ylabel = "Counts"
        if normalize:
            _ylabel += " [MinMax Norm]"
        if self.calibration == 1 or calibration:
            log("---> Missing calibration", "yellow")
            _xlabel = "Channel"
            self.calib_scaler = 1
        else:
            _xlabel = "Energy [MeV]"
            self.calib_scaler = self.calibration * 1e-3

        try:
            filenames = [os.path.basename(i).split(".")[0] for i in paths]
            log(f"--> Drawing spectra of {len(paths)} files: ", wait=True)
            for nm in filenames:
                log(f"{nm}, ", "green", wait=True)
            fig, ax = create_figure()
            for n, file in enumerate(paths):
                with open(file, "r") as ff:
                    _dat = [float(i) for i in ff.readlines()][
                        2::
                    ]  # unidentified 2 rows
                    Y = np.array(_dat)
                    if normalize:
                        Y = self.normalize(Y)
                    X = np.arange(0, len(Y)) * self.calib_scaler
                    add_plot(
                        ax,
                        X,
                        Y,
                        filenames[n],
                        legend=True,
                        fmt="-",
                        xlabel=_xlabel,
                        ylabel=_ylabel,
                        grid=True,
                    )
            save(fig, f"InSpect.{self.plot_ext}")

        except Exception as e:
            log("---> Failed to draw spectra", "red")
            log(f"\n{e}")

    def normalize(self, data, method="minmax"):
        """
        Normalize dataset.

        :param data: 1D list to normalize
        :type data: numpy.ndarray
        :param method: method of normalization, defaults to minmax, available: minmax, propability
        :type method: str

        :return: normalized list of data
        :rtype: numpy.ndarray
        """
        data = np.array(data)
        normalized_data = np.zeros_like(data)
        for n, nw in enumerate(data):
            if method == "minmax":
                normalized_data[n] = (nw - data.min()) / (data.max() - data.min())
        return normalized_data
