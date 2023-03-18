from __future__ import annotations

__version__ = "v0.1"

import inspect

# import os
from pathlib import Path
from sys import argv
from typing import Any, Callable

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import toml
from DRSpy.plugins.digitizer_reader import DigitizerEventData

# from scipy import stats
# from digitizer_reader import *
from scipy.optimize import curve_fit
from scipy.special import erf

sns.set_theme()


class Analysis:
    """Analysis class."""

    def __init__(
        self,
        data_path: str,
        limit_file: str | None = None,
        charts_path: str | None = None,
    ) -> None:
        """Init.

        data_path:     path to folder with .root files
        limit_file:    path to .root file. Reading events from other files will be limited to limit_file size.
        charts_path:   path to plots output folder
        """
        # init params
        self.files = []
        self.df = pd.DataFrame()
        self.ddf = pd.DataFrame()
        self.data_path = Path(data_path)
        # Check charts_path
        if charts_path:
            self.charts_path = Path(charts_path)
            for folder in ["waveforms", "time", "asymmetry", "stats"]:
                self.charts_path.joinpath(folder).mkdir(exist_ok=True, parents=True)
        else:
            self.charts_path = False
        # Check limit_file
        self.read_limit = (
            len(DigitizerEventData.create_from_file(limit_file)[0]) if limit_file else 0
        )
        # Read self.data_path
        choice = 0
        for file in self.data_path.iterdir():
            if file.suffix == ".df" and not choice:
                print("---------------------------------")
                df_files = {
                    f"{i}": df_file
                    for i, df_file in enumerate(self.data_path.iterdir())
                    if df_file.suffix == ".df"
                }
                for df_file in df_files:
                    print(f"\tID:{df_file}\t{df_files[df_file].name}")
                print("---------------------------------")
                print(
                    f"->\tFound DataFrame files in {self.data_path}.\n\tLoad file? [<ID>/N]"
                )
                choice = input("Choice: ")
                if df_files.get(choice, False):
                    print("->\tLoading DataFrames...")
                    self.df = pd.read_csv(df_files[choice].absolute())
                    self.ddf = pd.read_csv(
                        df_files[choice].with_suffix(".ddf").absolute()
                    )
            if file.suffix == ".root":
                self.files.append(self.data_path.joinpath(file))
        # Get waveform properties
        evt = DigitizerEventData.create_from_file(self.files[0])
        self.channels = len(evt)
        self.T_samples = len(evt[0][0].waveform)
        self.t = 0.2 * np.arange(0, self.T_samples)
        # Define graphs styling
        self.chart_ext = ".pdf"
        sns.set_theme()
        self.hist_density = 350
        self.line_plot_style = {"ls": "-", "lw": 1}
        self.fitline_plot_style = {"ls": "--", "lw": 1}
        self.scatter_plot_style = {"s": 10}
        # ADD translator func
        # Define DataFrame column names
        self.df_cols = [
            "event",
            "timestamp",
            "L",
            "CH",
            "A",
            "t_0",
            "t_r",
            "t_f",
            "Q",
            "dV",
            "V_0",
        ]
        self.df_cols_sigma = [
            "sig_t_0",
            "sig_t_r",
            "sig_t_f",
            "sig_Q",
            "sig_dV",
            "sig_V_0",
        ]
        self.ddf_cols = [
            "event",
            "t_0_ch0",
            "t_0_ch1",
            "L",
            "Q_ch0",
            "Q_ch1",
            "A_ch0",
            "A_ch1",
            "dt",
            "lnQ",
            "sqrtQ",
            "asymQ",
            "lnA",
            "sqrtA",
            "asymA",
        ]
        self.df_cols_latex = [
            "evt_ID",
            "timestamp",
            "L [cm]",
            "Channel",
            "Amp. [V]",
            "$t_0$ [ns]",
            "$t_r$ [ns]",
            "$t_f$ [ns]",
            "Charge",
            "$dV$ [V]",
            "$V_0$ [V]",
        ]
        self.df_cols_sigma_latex = [
            "$\sigma t_0$",
            "$\sigma t_r",
            "$\sigma t_f$",
            "$\sigma Q$",
            "$\sigma dV$",
            "$\sigma V_0$",
        ]
        self.ddf_cols_latex = [
            "evt_ID",
            "$t_0^{CH0}$ [ns]",
            "$t_0^{CH1}$ [ns]",
            "L [cm]",
            "$Q^{CH0}$",
            "$Q^{CH1}$",
            "Amp. CH0 [V]",
            "Amp. CH1 [V]",
            "$t_1^{CH1} - t_0^{CH0}$",
            "$\ln{\\frac{Q_1}{Q_0}}$",
            "$\sqrt{Q_0Q_1}$",
            "$\\frac{Q_0-Q_1}{Q_1+Q_1}$",
            "$\ln{\\frac{A_1}{A_0}}$",
            "$\sqrt{A_0A_1}$",
            "$\\frac{A_0-A_1}{A_1+A_1}$",
        ]
        # change default df column names to LaTEX
        self.lx = dict(
            zip(
                self.df_cols + self.df_cols_sigma + self.ddf_cols,
                self.df_cols_latex + self.df_cols_sigma_latex + self.ddf_cols_latex,
            )
        )

        self.toml_manager()
        print(
            f"\n->\tFound:\n\t\t* {len(self.files)} .root files\n\t\t* {self.T_samples=}\n\t\t* {self.channels=}\n"
        )

    def toml_manager(self) -> None:
        """Init and read config.toml."""
        print("->\tChecking config.toml...")
        self.config_path = Path(self.data_path.joinpath("config.toml"))
        if self.config_path.exists():
            self.data: dict[str, Any] = toml.load(self.config_path)
        else:
            print("\t->File not found. Initialization...")
            self.data = {
                "metadata": {},
                "filters": {},
                "style": {},
                "history": {},
                "progress": {},
            }
            self.config_path.touch()
            self.data["mxveto_version"] = __version__
            self.config_path = toml.dump(self.config_path)

    def decode_filename(
        self, filename: str, separator: str = "_", pos: int = 1
    ) -> float | bool:
        """Extract position data from filename."""
        try:
            position = float(filename.split(separator)[pos])
            return position
        except Exception as e:
            print(f"Can't decode filename. Skip\n{e}")
            return False

    def load_waveform(self, root_filename: Path) -> list[Any]:
        """TMP."""
        events = DigitizerEventData.create_from_file(root_filename)
        waveforms = []  # waveform[channel] = [waveforms list, amplitudes list]
        for channel in range(self.channels):
            waveforms.append([])
            waveforms[channel].append([event.waveform for event in events[channel]])
            waveforms[channel].append([event.amplitude for event in events[channel]])
        return waveforms

    def load_waveforms(
        self,
        filename_decoder: Callable[[str], float | bool],
        fit_func: Callable[[Any], float],
        wf_num: int = -1,
    ) -> pd.DataFrame:
        """Load waveforms from self.data_path."""
        evt_counter = 0
        p_list = [[] for _ in range(len(self.df_cols))]  # params & cov list
        q_list = [[] for _ in range(len(self.df_cols))]
        for nfile, filename in enumerate(self.files):
            source_position = filename_decoder(filename.name)
            if not isinstance(source_position, float):
                print(f"\n\t->\tcant decode {filename=}. Skip")
                continue
            waveforms = DigitizerEventData.create_from_file(filename)
            plot_counter = 10
            with click.progressbar(
                range(len(waveforms[0][:wf_num])),
                label=f"Analyzing waveforms {nfile}/{len(self.files)} ({source_position}cm)\t",
            ) as wbar:
                for nevt in wbar:
                    if self.read_limit and nevt > self.read_limit:
                        print("\n\t->\tFile is to big. SKIP")
                        break
                    evt_counter += 1
                    for channel in range(self.channels):
                        # FIX
                        p_list[0].append(evt_counter)
                        p_list[1].append(waveforms[channel][nevt].timestamp)
                        p_list[2].append(source_position)
                        p_list[3].append(channel)
                        p_list[4].append(waveforms[channel][nevt].amplitude)
                        p, q = self.get_waveform_fit(
                            fit_func, waveforms[channel][nevt].waveform
                        )
                        for i in range(len(p)):
                            p_list[
                                i + len(self.df_cols) - len(self.df_cols_sigma)
                            ].append(p[i])
                            q_list[i].append(np.sqrt(np.diag(q))[i])
                    if plot_counter != 0:
                        plot_counter -= 1
                        plot_df = pd.DataFrame(
                            dict(zip(self.df_cols_latex, [i[-2:] for i in p_list]))
                        )
                        try:
                            self.plot_waveforms(
                                waveforms[0][nevt].waveform,
                                waveforms[1][nevt].waveform,
                                fit_func,
                                f"Waveform $L={source_position}$cm",
                                plot_df,
                            )
                        except Exception as e:
                            print(f"Can't create plot. Clearing figure...\n{e}")
                            plt.clf()
                            plt.cla()
        params_dict = dict(zip(self.df_cols + self.df_cols_sigma, p_list + q_list))
        return pd.DataFrame(params_dict)

    def prepare(self, raw_df: pd.DataFrame | str) -> tuple[pd.DataFrame]:
        """Filter raw DataFrame and extract only useful entries.

        The method generates filtered df with relative errors, groups by
        event and source position (ddf) and
        """
        print("->\tCreating DataFrame")
        if isinstance(raw_df, str):
            self.df = pd.read_csv(raw_df)
        else:
            self.df = pd.DataFrame(raw_df)
        dcol = ["event", "t_0", "L", "Q", "A"]
        print("->\tClearing nan and inf values")
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
        self.df.reset_index(inplace=True, drop=True)

        print("->\tCalculating rel error")
        tmp = [[] for i in range(len(self.df_cols_sigma) - 1)]
        with click.progressbar(range(len(self.df))) as bar:
            for i in bar:
                for j in range(len(tmp)):
                    diff = round(
                        self.df.loc[i, self.df.columns[len(self.df_cols) + 1 + j]]
                        / self.df.loc[
                            i,
                            self.df.columns[
                                (len(self.df_cols) - len(self.df_cols_sigma) + 1) + j
                            ],
                        ],
                        4,
                    )
                    if np.abs(diff) > 1 or diff == np.nan or np.abs(diff) == np.inf:
                        tmp[j].append(1)
                    else:
                        tmp[j].append(diff)
        tmp = dict(
            zip(
                "rel_"
                + self.df.columns[
                    (len(self.df_cols) - len(self.df_cols_sigma) + 1) : (
                        len(self.df_cols) + 1
                    )
                ],
                tmp,
            )
        )
        for i in tmp.keys():
            self.df[i] = tmp[i]

        print("->\tFiltering df data...")
        self.df = self.df[
            (self.df["t_r"] > 0)
            & (self.df["t_f"] > 0)
            & (self.df["t_r"] < 10)
            & (self.df["t_f"] < 10)
            & (self.df["t_0"] > 20)
            & (self.df["t_0"] < 80)
            & (self.df["Q"] <= 0)
            & (self.df["Q"] > -5)
        ]

        print("->\tCalculating asymmeties")
        self.ddf = self.df[self.df["CH"] == 0][dcol].merge(
            self.df[self.df["CH"] == 1][dcol],
            how="inner",
            on=["event", "L"],
            suffixes=["_ch0", "_ch1"],
        )
        self.ddf[self.ddf_cols[8]] = self.ddf.apply(
            lambda x: x.t_0_ch1 - x.t_0_ch0, axis=1
        )
        self.ddf[self.ddf_cols[9]] = self.ddf.apply(
            lambda x: np.log(x.Q_ch1 / x.Q_ch0), axis=1
        )
        self.ddf[self.ddf_cols[10]] = self.ddf.apply(
            lambda x: np.sqrt(x.Q_ch1 * x.Q_ch0), axis=1
        )
        self.ddf[self.ddf_cols[11]] = self.ddf.apply(
            lambda x: (x.Q_ch0 - x.Q_ch1) / (x.Q_ch0 + x.Q_ch1), axis=1
        )
        self.ddf[self.ddf_cols[12]] = self.ddf.apply(
            lambda x: np.log(x.A_ch1 / x.A_ch0), axis=1
        )
        self.ddf[self.ddf_cols[13]] = self.ddf.apply(
            lambda x: np.sqrt(x.A_ch1 * x.A_ch0), axis=1
        )
        self.ddf[self.ddf_cols[14]] = self.ddf.apply(
            lambda x: (x.A_ch0 - x.A_ch1) / (x.A_ch0 + x.A_ch1), axis=1
        )
        print("->\tFiltering ddf data...")
        self.ddf = self.ddf[
            (np.abs(self.ddf["asymQ"]) < 4) & (np.abs(self.ddf["lnQ"])) < 4
        ]

        # col = ["t_0", "t_r", "t_f", "Q", "A", "V_0", "dV"]
        # hcol = ["t_r", "t_f", "Q", "A", "V_0", "dV"]
        # tt = [[] for i in range(len(col))]
        # dh = 200

    def plot_waveforms(self, w0, w1, fit_func, note, df) -> None:
        f, ax = plt.subplots(
            2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [1, 4]}
        )
        sns.lineplot(
            x=self.t,
            y=w0,
            color="green",
            label="CH0",
            **self.line_plot_style,
            alpha=0.6,
            ax=ax[1],
        )
        sns.lineplot(
            x=self.t,
            y=w1,
            color="red",
            label="CH1",
            **self.line_plot_style,
            alpha=0.6,
            ax=ax[1],
        )
        sns.lineplot(
            x=self.t,
            y=fit_func(
                self.t,
                *df.iloc[
                    0,
                    range(
                        len(self.df_cols) - len(self.df_cols_sigma), len(self.df_cols)
                    ),
                ],
            ),
            color="green",
            **self.line_plot_style,
            alpha=0.6,
            ax=ax[1],
        )
        sns.lineplot(
            x=self.t,
            y=fit_func(
                self.t,
                *df.iloc[
                    1,
                    range(
                        len(self.df_cols) - len(self.df_cols_sigma), len(self.df_cols)
                    ),
                ],
            ),
            color="red",
            **self.line_plot_style,
            alpha=0.6,
            ax=ax[1],
        )
        ax[0].axis("off")
        ax[0].axis("tight")
        ax[0].set_title(note)
        # ax[1].axis('tight')
        table = ax[0].table(
            cellText=df.round(4).to_numpy(), colLabels=df.columns, loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        # g.annotate(note, size=10, xy={0, np.min(np.append(w0, w1)/2)} )
        # g.axvline(30, ls='--', lw=0.9, c='red')
        # plt.show()
        filename = f"waveform_{df.iloc[0, 0]}"
        f.savefig(
            self.charts_path.joinpath("waveforms")
            .joinpath(filename)
            .with_suffix(self.chart_ext)
        )
        plt.clf()

    def plot_lin(
        self, fit_func, y="dt", fit_param="c", folder="asymmetry", lim=(-20, 20)
    ):
        df = self.ddf.groupby("L").mean().reset_index()
        p, q = curve_fit(fit_func, df["L"], df[y])
        c = round(-2 / p[0], 3)
        err = round(np.abs(np.sqrt(q[0][0]) / p[0]), 3)
        print(f"->\tFound {fit_param}={c} +- {err}")
        sns.scatterplot(
            data=df.rename(columns=self.lx),
            x=self.lx["L"],
            y=self.lx[y],
            **self.scatter_plot_style,
        )
        sns.lineplot(
            x=df["L"], y=fit_func(df["L"], *p), **self.fitline_plot_style, color="black"
        )
        filename = f"fit_{y}"
        plt.savefig(
            self.charts_path.joinpath(folder)
            .joinpath(filename)
            .with_suffix(self.chart_ext)
        )
        plt.clf()
        sns.histplot(
            data=self.ddf.rename(columns=self.lx),
            x=self.lx["L"],
            y=self.lx[y],
            cbar=True,
        )
        sns.lineplot(
            x=df["L"], y=fit_func(df["L"], *p), **self.fitline_plot_style, color="black"
        )
        plt.ylim(lim[0], lim[1])
        filename = f"fit_{y}_hist"
        plt.savefig(
            self.charts_path.joinpath(folder)
            .joinpath(filename)
            .with_suffix(self.chart_ext)
        )
        plt.clf()

    def plot_asym(self, y, color, folder="asymmetry"):
        if len(y) > 1:
            for i, yy in enumerate(y):
                sns.lineplot(
                    data=self.ddf.rename(columns=self.lx),
                    x=self.lx["L"],
                    y=self.lx[yy],
                    **self.line_plot_style,
                    color=color[i],
                )
        else:
            sns.lineplot(
                data=self.ddf.rename(columns=self.lx),
                x=self.lx["L"],
                y=self.lx[y[0]],
                **self.line_plot_style,
                color=color[0],
            )
        filename = f"asym_{y}"
        plt.savefig(
            self.charts_path.joinpath(folder)
            .joinpath(filename)
            .with_suffix(self.chart_ext)
        )
        plt.clf()

    def plot_signal(self, fit_func):
        self.plot_lin(fit_func, "lnQ", "Qlambda", lim=(-4, 4))
        self.plot_lin(fit_func, "lnA", "Alambda", lim=(-4, 4))

        y = [
            ["sqrtQ"],
            ["asymQ"],
            ["sqrtA"],
            ["asymA"],
            ["Q_ch0", "Q_ch1"],
            ["A_ch0", "A_ch1"],
        ]
        colors = ["black", "red"]
        for yy in y:
            self.plot_asym(yy, colors)

    def plot_joint(self):
        xy = [("t_0_ch0", "t_0_ch1"), ("dt", "lnQ")]
        sns.jointplot(
            data=self.df.rename(columns=self.lx),
            x=self.lx["t_r"],
            y=self.lx["t_f"],
            kind="hist",
        )
        filename = "joint_t_r-t_f"
        plt.savefig(
            self.charts_path.joinpath("time")
            .joinpath(filename)
            .with_suffix(self.chart_ext)
        )
        plt.clf()
        for plot in xy:
            sns.jointplot(
                data=self.ddf.rename(columns=self.lx),
                x=self.lx[plot[0]],
                y=self.lx[plot[1]],
                kind="hist",
            )
            filename = f"joint_{plot[0]}-{plot[1]}"
            plt.savefig(
                self.charts_path.joinpath("time")
                .joinpath(filename)
                .with_suffix(self.chart_ext)
            )
            plt.clf()

    def get_waveform_fit(
        self, fit_func: Callable[[Any], float], waveform: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        t_max = self.t[np.argmin(waveform)]
        t_r = 2.9
        t_f = 3
        t0 = t_max - t_r * t_f / (t_f - t_r) * np.log(t_f / t_r)
        V0 = np.mean(waveform[:10])
        dV = (np.mean(waveform[:-10]) - V0) / self.t[-1]
        Q = np.sum(waveform * 0.2) - V0 * self.t[-1] - 0.5 * dV * self.t[-1] ** 2

        p0 = [t0, t_r, t_f, Q, dV, V0]
        try:
            p, q = curve_fit(fit_func, self.t, waveform, p0=p0)
            return p, q
        except Exception as e:
            print(f"Can't calcuclate fit params. Skip'n{e}")
            return np.full((len(p0)), np.inf), np.full((len(p0), len(p0)), np.inf)

    def get_hist_avg(
        self, key, fit_func, ext="pdf", tp="hist", tries=20, dh=100, max_err=0.01
    ):
        t: list[list[npt.DTypeLike]] = [[], [], [], [], []]
        params_len = len(inspect.signature(fit_func).parameters.keys()) - 1
        color = ["red", "blue"]
        L = list(set(self.df["L"]))
        for pos in L:
            tm: list[npt.DTypeLike] = [np.float64(0) for _ in range(4)]
            plt.cla()
            plt.clf()
            for ch in range(self.channels):
                c = self.df[(self.df["L"] == pos) & (self.df["CH"] == ch)][key]
                a, b = np.histogram(c, dh, density=True)
                wg = np.average(b[:-1], weights=a)
                try:
                    try:
                        p, q = curve_fit(fit_func, b[:-1], a)
                    except Exception as e:
                        print(f"first {e}")
                        p = np.ones(params_len)
                        q = np.full((params_len, params_len), 2)
                    print(f"L={pos} {p[0]}   {np.sqrt(q[0, 0])}")
                    dhh = dh
                    _tries = tries
                    while (
                        np.abs(np.sqrt(np.diag(q)[0]) / p[0]) > max_err and _tries > 0
                    ):
                        dhh += int((200) / tries)
                        _tries -= 1
                        a, b = np.histogram(c, dhh, density=True)
                        try:
                            pp, qq = curve_fit(
                                fit_func,
                                b[:-1],
                                a,
                                p0=[wg] + list(range(params_len - 1)),
                            )
                            print(f"\t try {_tries} {pp[0]} {np.sqrt(qq[0, 0])}")
                            if np.sqrt(np.diag(q)[0]) > np.sqrt(np.diag(qq)[0]):
                                p = pp
                                q = qq
                                # dh = dhh
                        except Exception as e:
                            print(f"->\t {e}")

                except Exception as e:
                    p = np.array([wg, 5e-2, 5e2])
                    q = np.array([1])
                    print(f"ayy {e}")
                    continue
                tm[ch] = p[0]
                tm[2 + ch] = np.sqrt(np.diag(q)[0])
                if self.charts_path:
                    sns.lineplot(x=b[:-1], y=a, color=color[ch], label=f"CH{ch}")
                    sns.lineplot(
                        x=b[:-1],
                        y=fit_func(b[:-1], *p),
                        ls="--",
                        alpha=0.6,
                        color="black",
                    )
            if self.charts_path:
                plt.title(f"L={pos}cm")
                plt.xlabel(key)
                plt.legend()
                # plt.show()
                plt.savefig(
                    self.charts_path.joinpath("stats")
                    .joinpath(f"{tp}_{key}_{pos}cm")
                    .with_suffix(self.chart_ext)
                )
                plt.clf()
                plt.cla()

            t[0].append(pos)
            t[1].append(tm[0])
            t[2].append(tm[1])
            t[3].append(tm[2])
            t[4].append(tm[3])
        return t

    def get_wg_avg(self, key, ext="pdf", tp="wg", dh=200):
        t: list[list[npt.DTypeLike]] = [[], [], [], [], []]
        color = ["red", "blue"]
        L = list(set(self.df["L"]))
        for pos in L:
            tm: list[npt.DTypeLike] = [np.float64(0) for _ in range(4)]
            cc = [0, 0]
            for ch in range(2):
                cc[ch] = self.df[(self.df["L"] == pos) & (self.df["CH"] == ch)][key]
                a, b = np.histogram(cc[ch], dh)
                tm[ch] = np.average(b[:-1], weights=a)
                tm[2 + ch] = np.float64(1)
                t[0].append(pos)
                t[ch + 1].append(tm[ch])
                t[ch + 3].append(tm[2 + ch])
                # p, q = curve_fit(gauss_fit, b[:-1], a)
            if self.charts_path:
                g1 = sns.histplot(
                    x=cc[0], bins=dh, color=color[0], label="CH0", alpha=0.3
                )
                g1.axvline(tm[ch], ls="--", lw=2, c=color[0])
                g2 = sns.histplot(
                    x=cc[1], bins=dh, color=color[1], label="CH1", alpha=0.3
                )
                g2.axvline(tm[ch], ls="--", lw=2, c=color[1])
                plt.title(f"L={pos}cm")
                plt.xlabel(key)
                plt.legend()
                # plt.savefig(self.charts_path.joinpath(f'{tp}_{key}_{l}cm').with_suffix(self.chart_ext)); plt.clf(); plt.cla()
                plt.show()
        return t


def gauss_fit(x, x0, a, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def asymgauss_fit(x, m1, m0, m2, m3):
    amp = m0 / (m2 * np.sqrt(2 * np.pi))
    spread = np.exp((-((x - m1) ** 2)) / (2 * m2**2))
    skew = 1 + erf((m3 * (x - m1)) / (m2 * np.sqrt(2)))
    return amp * spread * skew


def lin_fit(x, a, b):
    return a * x + b


def landau_fit(x, E, S, N):
    return N / np.sqrt(2 * np.pi) * np.exp((((E - x) / S) - np.exp(-((x - E) / S))) / 2)


def sig_fit(t, t0, t_r, t_f, Q, dV, V0=0):
    return (
        V0
        + dV * t
        + np.heaviside(t - t0, 0)
        * Q
        / (t_r - t_f)
        * (np.exp(-(t - t0) / t_r) - np.exp(-(t - t0) / t_f))
    )
    # return V0 + dV*t + np.heaviside(t-t0, 0) * Q/t_f * (1 + t_r/t_f) * (1 - np.exp(-(t-t0)/t_r)) * np.exp(-(t-t0)/t_f)


if __name__ == "__main__":
    print("Usage: python analysis.py <data_folder>")
    run = Analysis(argv[1], charts_path=argv[3], limit_file=argv[2])
    df = pd.read_csv(argv[4])
    run.df = df[df["Q"] > -1.2]
    run.chart_ext = ".png"

    # dff = run.load_waveforms(run.decode_filename, sig_fit)
    # run.prepare(dff)
    # run.plot_lin(lin_fit)
    # run.plot_signal(lin_fit)
    # run.plot_joint()
