import os
import linecache
import pandas as pd

from DRSpy.main import click


def log(msg, color="white", wait=False):
    if wait:
        print(click.style(msg, fg=color), end="")
    else:
        print(click.style(msg, fg=color))
    return None


class DataStruct:
    """ """

    def __init__(self, data_file="drs_data.csv", fverbose=False):
        log("-> Initialization...")
        self.data_file = data_file
        self.check_csv()
        self.fverbose = fverbose
        # if fverbose: self.describe()
        # Data columns in txt files
        self.cuts = {"PtP-CH0": [2, 153], "PtP-CH1": [156, 308], "Delay": [3, 154]}

    @property
    def data(self):
        """
        :return: return database
        :rtype: pandas.DataFrame
        """
        return self._data

    @data.setter
    def data(self, new_data):
        """
        Extend the database with new data and update csv file

        :param new_data: new dataframe
        :type new_data: pandas.DataFrame
        """
        try:
            if self.fverbose:
                log("---> Creating backup file", "yellow")
            self._data.to_csv(f"{self.data_file}_bck", index=False)
            if self.fverbose:
                log("---> Merging data files", "yellow")
            self._data = pd.merge(
                self._data, new_data, on=list(new_data.columns), how="outer"
            )
            if self.fverbose:
                log("---> Saving new csv file", "yellow")
            self._data.to_csv(f"{self.data_file}", index=False)
        except Exception as e:
            log(
                f"---> Failed to update dataframe. Backup file saved as {self.data_file}_bck:\n\n{e}",
                color="red",
            )

    def check_csv(self):
        """
        Check for existance of database or create new one.
        """
        self.nP2P = 0
        self.nDelay = 0
        self.nXML = 0
        if os.path.isfile(self.data_file):
            log(f"--> Datafile {self.data_file} exists")
            self._data = pd.read_csv(f"{self.data_file}", index_col=False)
        else:
            log(f"--> Creating empty datafile: {self.data_file}")
            self._data = pd.DataFrame(
                {
                    "Channel [V]": [],
                    "Counts CH0": [],
                    "Counts CH1": [],
                    "Delay [ns]": [],
                    "Counts delay": [],
                    "Distance [cm]": [],
                    "Position": [],
                }
            )
            self._data.to_csv(f"{self.data_file}", index=False)
        return None

    def read_xml(self):
        pass

    def read_txt(self, filename, ftype):
        """
        Read txt file and append to database.
        !TMP: data rows ranges hardcoded in self.cuts!

        :param filename: full path to data file
        :type filename: str
        :param ftype: type of file "P2P" or "Delay"
        :type ftype: str
        """
        try:
            meta = self.name_decode(filename.split("/")[-1].split(".")[0])
            if ftype == "P2P":
                ch0 = pd.read_table(
                    filename,
                    skiprows=lambda x: x
                    not in range(self.cuts["PtP-CH0"][0], self.cuts["PtP-CH0"][1] + 1),
                    names=["Channel [V]", "XX", f"Counts CH0"],
                )
                ch1 = pd.read_table(
                    filename,
                    skiprows=lambda x: x
                    not in range(self.cuts["PtP-CH1"][0], self.cuts["PtP-CH1"][1] + 1),
                    names=["Channel [V]", "XX", f"Counts CH1"],
                )
                # merge frames without XX col
                _data = pd.merge(
                    ch0.loc[:, ["Channel [V]", "Counts CH0"]],
                    ch1.loc[:, ["Channel [V]", "Counts CH1"]],
                    on="Channel [V]",
                    how="outer",
                )
            elif ftype == "Delay":
                delay = pd.read_table(
                    filename,
                    skiprows=lambda x: x
                    not in range(self.cuts["Delay"][0], self.cuts["Delay"][1] + 1),
                    names=["Delay [ns]", "DX", "Counts delay"],
                )
                _data = delay.loc[:, ["Delay [ns]", "Counts delay"]]
        except Exception as e:
            log(f"---> Failed to load {filename} Exception:\n{e}", "red")
        else:
            _data["Distance [cm]"] = meta["Distance"]
            _data["Position"] = meta["Position"]
            self.data = _data  # send to data.setter
        finally:
            return None

    def auto_decode(self, path):
        """
        Recognize file types automatically in directory and redirect to self.read_txt() or self.read_xml()

        :param path: Path to folder with data files
        :type path: str
        """
        try:
            log("---> Trying to decode files.", wait=True)
            filenames = os.listdir(path)
            txt_files = [file for file in filenames if ".txt" in file]
            xml_files = [file for file in filenames if ".xml" in file]
            xml_files = [os.path.join(path, f) for f in xml_files]
            txt_files = [os.path.join(path, f) for f in txt_files]
            header = ""
            txt_filesPtP = []
            txt_filesDelay = []
            for file in txt_files:
                try:
                    header = linecache.getline(file, 1)
                    if self.fverbose:
                        log(
                            f"---> Autorec: found head: {header}, ", "yellow", wait=True
                        )
                except Exception as e:
                    log(f"\n---> Failed to decode {file} file:  {e}", "red")
                else:
                    if "Pk-Pk" in header:
                        txt_filesPtP.append(file)
                        if self.fverbose:
                            log(f"------> P2P, ", "yellow")
                    elif "delay" in header:
                        txt_filesDelay.append(file)
                        if self.fverbose:
                            log(f"------> Del, ", "yellow")
                    else:
                        log(
                            f"---> Unknow file content {file}. Line 1: {header}",
                            "yellow",
                        )
        except Exception as e:
            log(f"\n---> Failed to decode files {e}", "red")
        else:
            log(
                f"\n---> Detected: {len(xml_files)} .xml, {len(txt_files)} .txt (p2p-{len(txt_filesPtP)}, delay-{len(txt_filesDelay)})"
            )
            with click.progressbar(
                txt_filesPtP, label="---> Loading Peak2Peak files: "
            ) as P2Pbar:
                for P2Pfile in P2Pbar:
                    self.read_txt(P2Pfile, "P2P")
            with click.progressbar(
                txt_filesDelay, label="---> Loading Delay files: "
            ) as Dbar:
                for Dfile in Dbar:
                    self.read_txt(Dfile, "Delay")
            # with click.progressbar(xml_files, label="---> Loading Delay files: ") as XMLbar:
            # for XMLfile in XMLbar: self.read_xml(XMLfile)

    def name_decode(self, name, style="std"):
        """
        Extract metainfo from filename e.g. 40_U_t to dict{"Distance":40, "Position":"U", "Delay":True}

        :param name: filename to decode
        :type name: str
        :param style: filename format, default to "std"
        :return: decoded metainfo
        :rtype: dict{"Distance":float, "Position":str, "Delay":bool}
        """
        decoder = {}
        try:
            if style == "std":
                name = name.split("_")
                decoder["Distance"] = float(name[0])
                decoder["Position"] = str(name[1])
                if len(name) >= 3:
                    decoder["Delay"] = True
                else:
                    decoder["Delay"] = False
            else:
                pass
        except Exception as e:
            log(f"--> Failed to decode name {name}, error:\n{e}", "red")
        else:
            return decoder
