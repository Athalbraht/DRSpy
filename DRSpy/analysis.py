import numpy as np
import  matplotlib
import numpy as np
import  matplotlib.pyplot as plt
import DRSpy.optimize as optim

from DRSpy.data_struct import click, log

matplotlib.use("Agg")

class Analysis():
    """
    """
    def __init__(self, datastruct, dtype="txt", fit=False, fverbose=False):
        log(f"-> Analysis initialization")
        self.fverbose = fverbose
        self.fit = fit
        self._data = datastruct
        self.get_metainfo()

    def get_metainfo(self):
        """ """
        self.positions = set(self._data["Position"].to_list())
        self.distances = set(self._data["Distance [cm]"].to_list())
    
    def getdata(self, dist, pos, dtype):
        """
        """
        if pos in self.positions and dist in self.distances:
            _data = self._data[self._data["Position"] == pos]
            _data = _data[_data["Distance [cm]"] == dist]
        else:
            log(f"---> Cannot find {pos} or {dist} in database. Rejected", "red")
            exit()
        if dtype == "CH":
            _data = _data.loc[:, ["Channel [V]", "Counts CH0", "Counts CH1"]]
        elif dtype == "T":
            _data = _data.loc[:, ["Delay [ns]", "Counts delay",]]
        return _data.dropna().astype(float)

    def get_avg_delay(self, edges=["C", "D", "U"]):
        for edge in edges:
            wavg = [[], []] # POS, WAVG
            with click.progressbar(self.distances, label="---> Calculating weighted averages for edge {edge} ") as distances:
                for distance in distances: 
                    try:
                        _cut = np.array(self.getdata(distance, edge, "T")).T
                        _wa = optim.w_avg(_cut[0], _cut[1])
                    except Exception as e:
                        log(f"---> Missing column for pos: {edge} and dist: {distance}: ", wait=True); log(f"skip", "yellow")
                    else:
                        wavg[0].append(distance)
                        wavg[1].append(_wa)
        
        
    
    def get_ch_hist(self):
        pass
    
    def get_hist(self, htype, fit=False):
        pass


