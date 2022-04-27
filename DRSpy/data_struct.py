import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import check_output
from DRSpy.main import click

import  matplotlib
import linecache
#matplotlib.use("Agg")
import  matplotlib.pyplot as plt

from scipy.optimize import curve_fit

def log(msg, color="white", wait=False):
    if wait:    print(click.style(msg, fg=color), end="")
    else:       print(click.style(msg, fg=color))
    return None

class DataStruct():
    def __init__(self, config_file="drspy.config", drsframe="data.csv", fverbose=False):
        log(f"-> Creating DRSpy instance")
        self.drsframe = drsframe
        self._config_file = config_file
        self._fverbose = fverbose

        self.check_csv(drsframe)
        self.cuts = {
                        "PtP-CH0"   : [3, 153],
                        "PtP-CH1"   : [158, 308],
                        "Delay"     : [3, 154]}
        self.desc()

    def desc(self):
        log(f"---> Verbose mode: ", wait=True); log("Enabled", "green") if self._fverbose else log("Disabled", "red")
        log(f"---> Peak-to-Peak files to load: ", wait=True); log(f"{self.nPtP}","green")
        log(f"---> Time delay files to load: ", wait=True); log(f"{self.ndelay}","green")
        log(f"---> Waveform files to load: ", wait=True); log(f"{self.nxml}","green")
        log(f"---> Cuts: ", wait=True); log(f"{self.cuts}","green")
    
    def conf_parser(self):
        if os.path.isfile(self._config):
            pass
        else:
            log(f"---> Creating config file {self._config_file}")
            with open(self._config_file, "r") as file: file.write("#DRSpy config file")
    
    def check_csv(self, new_csv):
        if os.path.isfile(new_csv):
            if self._fverbose: log("-> Configuration file exists: ", wait=True); log(new_csv, "green")
            self._data = pd.read_csv(f"{new_csv}", index_col=False)
            self.nxml = 0
            self.nPtP = 0
            self.ndelay = 0
            for col in self.data.columns:
                if "CH1" in col: self.nPtP += 1
                if "_t" in col: self.ndelay += 1
        else:
            if self._fverbose: log("---> New DataFrame created")
            self.nxml = 0
            self.nPtP = 0
            self.ndelay = 0
            self._data = pd.DataFrame({"Channel":[], "CHX":[], "Delay [ns]":[], "DX":[]})
            self._data.to_csv(f"{self.drsframe}", index=False)
    
           
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, ns):
        self._data = self._data.astype(float)
        try:
            if self._fverbose: log("---> Updating DataFrame: ", wait=True)
            self._data.to_csv(f"{self.drsframe}_bck", index=False)
            self._data = pd.merge(self._data, ns, on=list(ns.columns[0:2]),  how="outer")
            os.remove(f"{self.drsframe}")
            self._data.to_csv(f"{self.drsframe}", index=False)
            self.check_csv(self.drsframe)
        except Exception as e:
            log(f"\n---> Updating DF Failed {e}")
        else:
            if self._fverbose: log("Done", "green")
    
    def auto_recognize(self, path, ftag=""):
        try:
            log("---> Trying to decode files.", wait=True)
            filenames = os.listdir(path)
            txt_files = [file for file in filenames if ".txt" in file]
            xml_files = [file for file in filenames if ".xml" in file]
            xml_files = [ os.path.join(path, f)  for f in xml_files]
            txt_files = [ os.path.join(path, f)  for f in txt_files]
            txt_filesPtP = []
            txt_filesDelay = []
            header = ""
            for file in txt_files:
                try:
                    header = linecache.getline(file, 1)
                    if self._fverbose: log(f"---> Autorec: found head: {header}, ", "yellow", wait=True)
                except Exception as e:
                    log(f"\n---> Failed to decode {file} file:  {e}", "red")
                else:
                    if "Pk-Pk" in header:
                        txt_filesPtP.append(file)
                        if self._fverbose: log(f"------> P2P, ", "yellow")
                    elif "delay" in header:
                        txt_filesDelay.append(file)
                        if self._fverbose: log(f"------> Del, ", "yellow")
                    else:
                        log(f"---> Unknow file content {file}. Line 1: {header}", "yellow")
        except Exception as e:
            log(f"\n---> Failed to decode files {e}", "red")
        else:
            #qself._fPtP.extend(txt_filesPtP)
            #self._fdelay.extend(txt_filesDelay)
            #self._fxml.extend(xml_files)
            log(f"\n---> Detected: {len(xml_files)} .xml, {len(txt_files)} .txt (p2p-{len(txt_filesPtP)}, delay-{len(txt_filesDelay)})")
            with click.progressbar(txt_filesPtP, label="---> Loading Peak2Peak files: ") as P2Pbar:
                for P2Pfile in P2Pbar: self.load_file(P2Pfile, "PtP", ftag)
            with click.progressbar(txt_filesDelay, label="---> Loading Delay files: ") as Dbar:
                for Dfile in Dbar: self.load_file(Dfile, "delay", ftag)

    def get_distance(self, name, spl="_"):
        return float(name.split(spl)[0])

    def plot(self, xreg="", yreg="", figsize=(12,4), fkind="line",ext="pdf", flive=False, filename="output", regx=False):
        #if not flive: matplotlib.use("Agg")
        x = []; y = []
        rx = re.compile(xreg)
        ry = re.compile(yreg)
        columns = list(self.data.columns)
        #for col in self.data.columns:
            #if xreg in col: x.append(col)
            #if yreg in col: y.append(col)
        x = list(filter(rx.match, columns))
        y = list(filter(ry.match, columns))
        if regx:
            return x,y
        else:
            print(x)
            print(y)
            for xs in x:
                fig, ax = plt.subplots(figsize=figsize)
                for ys in y:
                    try:
                        self.data.plot(xs, ys, kind=fkind, ax=ax)
                    except:
                        print("x")
                
                plt.show() if flive else plt.savefig(f"{filename}.{ext}") 
                plt.clf()

    def load_file(self, filename, ftype, ftag=""):
        header = filename.split("/")[-1]
        if ftype == "PtP":
            try:
                if self._fverbose: log(f"---> Converting (CH0, CH1) to DataFrame: file: {filename}, head: {header}, tag: {ftag}", wait=True)
                ch0 = pd.read_table(    filename,
                                        skiprows=lambda x: x not in range(self.cuts["PtP-CH0"][0], self.cuts["PtP-CH0"][1]+1),
                                        names=["Channel", "CHX", f"{ftag}{header[:-4]}-CH0"])
                if self._fverbose: log("(Done, ","green", wait=True)
                ch1 = pd.read_table(    filename,
                                        skiprows=lambda x: x not in range(self.cuts["PtP-CH1"][0], self.cuts["PtP-CH1"][1]+1),
                                        names=["Channel", "CHX", f"{ftag}{header[:-4]}-CH1"])
                if self._fverbose: log("Done), ","green", wait=True)
                log("  ", wait=True)
                ch01 = pd.merge(ch0,ch1, on=["Channel", "CHX"], how="outer")
                if self._fverbose: log("MERGED","green")

            except Exception as e:
                log(f"\n---> Converting to DataFrame failed. File {filename}: {e}","red")
                return False
            else:
                self.data = ch01
                return True
        elif ftype == "delay":
            try:
                if self._fverbose: log(f"---> Converting delay to DataFrame: {filename}, head: {header}, tag: {ftag}", wait=True)
                _delay = pd.read_table( filename,
                                        skiprows=lambda x: x not in range(self.cuts["Delay"][0], self.cuts["Delay"][1]+1),
                                        names=["Delay [ns]", "DX", f"{ftag}{header[:-4]}"])
                if self._fverbose: log("Done","green")
            except Exception as e:
                log(f"\n---> Converting to DataFrame failed. File {filename}: {e}","red")
            else:
                self.data = _delay
                return True
        else:
            raise TypeError(f"Unsupported file extension: {ftype}")
            return False
        
