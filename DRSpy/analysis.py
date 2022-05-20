import numpy as np

from DRSpy.optimize import *
from DRSpy.ruct import click, log

class Analysis():
    def __init__(self, datastruct, fit=False, fverbose=False):
        log(f"-> Analysis initialization")
        self.fverbose = fverbose
        self.fit = fit
        self._data = datastruct

    def get_metainfo(self):
        self.positions = set(self._data["Position"].to_list())
        self.distances = set(self._data["Distance [cm]"].to_list())
    
    def getdata(self, dist, pos, dtype):
        """
        """
        if dtype == "CH":
            pos
        elif dtype == "T":

    def get_avg_delay(self):
        pass
    
    def get_ch_hist(self):
        pass
    
    def get_hist(self, htype, fit=False):
        pass


