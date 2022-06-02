import os
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sys import argv
from DRSpy.visual import *
from scipy.optimize import curve_fit
from DRSpy.optimize import inspector_calib, fit
from DRSpy.data_struct import log


class InSpector():
    """
    """
    def __init__(self, plot_ext="png"):
        log("--> Initialize InSpector-100 module")
        self.plot_ext = plot_ext
        self.calibration = 1
    
    def calibrate(self, calibfile):
        """"""
        log("---> Starting calibration: ", wait=True)
        fig, ax = create_figure()
        try:
            calib_data = pd.read_csv(calibfile)
            calib_group = calib_data.groupby("Source")
            #fig, ax = plt.subplots()
            for source, data in calib_group:
                add_plot(ax, data["Channel"], data["E"], source, markersize=12, legend=True)
            x, y, p, pc = fit(inspector_calib, calib_data["Channel"], calib_data["E"])
            add_plot(ax, x, y, "Calib_fit", fmt="--", legend=True, xlabel="Channel", ylabel="Energy [keV]", grid=True)
            save(fig, f"InSpector_calibration.{self.plot_ext}")
        except Exception as e:
            log("Failed", "red")
            log(f"\n{e}")
        else:
            log("Done", "green")
            log("--> Calculated a param: ", wait=True); log(f"a = {round(p[0],3)} ± {round(pc[0,0], 3)} ", "green")
            self.calibration = p[0]
    
    def get_spectrum(self, paths):
        """ """
        if self.calibration == 1:
            log("---> Missing calibration", "yellow")
            _xlabel = "Channel"
        else:
            _xlabel = "Energy [MeV]"
            self.calibration *= 1e-3
            
        try:
            filenames = [ os.path.basename(i).split(".")[0] for i in paths ]
            log(f"--> Drawing spectra of {len(paths)} files: ", wait=True)
            for nm in filenames: log(f"{nm}, ", "green", wait=True)
            fig, ax = create_figure()
            for n, file in enumerate(paths):
                with open(file, "r") as ff:
                    _dat = [float(i) for i in ff.readlines()][2::] #unidentified 2 rows
                    Y = np.array(_dat)
                    X = np.arange(0, len(Y))*self.calibration
                    add_plot(ax, X, Y, filenames[n], legend=True,fmt="-", xlabel=_xlabel, ylabel="Counts", grid=True)
            save(fig, f"InSpect.{self.plot_ext}")
        
        except Exception as e:
            log("---> Failed to draw spectra", "red")
            log(f"\n{e}")
        

if __name__  == "__main__":
    files = argv[1::]
    filenames = [i.split(".")[0] for i in files]
    
    for n, file in enumerate(files):
        with open(file, "r") as ff:
            _dat = [float(i) for i in ff.readlines()][2::]
            Y = np.array(_dat)
            X = np.arange(0, len(Y))
            plt.plot(X,Y, label=filenames[n])
    plt.legend()
    plt.grid(True)
    plt.show()
    

