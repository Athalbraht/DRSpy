import numpy as np
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import DRSpy.optimize as optim
import DRSpy.visual as vis

from DRSpy.data_struct import click, log

matplotlib.use("Agg")


class Analysis:
    """ """

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
        """ """
        if pos in self.positions and dist in self.distances:
            _data = self._data[self._data["Position"] == pos]
            _data = _data[_data["Distance [cm]"] == dist]
        else:
            log(f"---> Cannot find {pos} or {dist} in database. Rejected", "red")
            exit()
        if dtype == "CH":
            _data = _data.loc[:, ["Channel [V]", "Counts CH0", "Counts CH1"]]
        elif dtype == "T":
            _data = _data.loc[
                :,
                [
                    "Delay [ns]",
                    "Counts delay",
                ],
            ]
        return _data.dropna().astype(float)

    def get_avg_delay(self, edges=["C", "D", "U"], fit=False, filename="delay_avg.png"):
        data = {}
        fig, ax = vis.create_figure()
        for edge in edges:
            wavg = [[], []]  # POS, WAVG
            with click.progressbar(
                self.distances,
                label=f"---> Calculating weighted averages for edge {edge} ",
            ) as distances:
                for distance in distances:
                    try:
                        _cut = np.array(self.getdata(distance, edge, "T")).T
                        _wa = optim.w_avg(_cut[0], _cut[1])
                    except Exception as e:
                        log(
                            f"---> Missing column for pos: {edge} and dist: {distance}: ",
                            wait=True,
                        )
                        log(f"skip", "yellow")
                    else:
                        wavg[0].append(distance)
                        wavg[1].append(_wa)
            data[edge] = wavg
            vis.add_plot(
                ax,
                wavg[0],
                wavg[1],
                xlabel="Distance [cm]",
                ylabel="Delay [ns]",
                title="Delay Weighted Mean",
                legend=True,
                grid=True,
                fmt="o",
                label=edge,
            )
        if fit and "C" in data.keys():
            xfit, yfit, params, pcovv = optim.fit(
                optim.linear, data["C"][0], data["C"][1]
            )
            vis.add_plot(
                ax,
                xfit,
                yfit,
                xlabel="Distance [cm]",
                ylabel="Delay [ns]",
                title="Delay Weighted Mean",
                legend=True,
                grid=True,
                fmt="--",
                label=f"fit: ax+b",
            )
            log("--> Fit ax+b: ", wait=True)
            log(
                f"a = {params[0]} +- {pcovv[0,0]**0.5}, b = {params[1]} +- {pcovv[1,1]**0.5}",
                "green",
            )
            log(f"--> Speed of light in scintillator: ", wait=True)
            log(f"c = {-2/params[0]}", "green")
            # source position fix fit

        vis.save(fig, filename)

    ### TO FIX !!!
    def get_p2p_spectra(self, dist, filename="P2P.png", fit=False):
        """ """
        landaupeaks = {"C": [], "D": [], "U": [], "PPA": []}
        asym_data = {"C": [], "D": [], "U": [], "PPA": []}
        with click.progressbar(
            self.distances, label=f"---> Calculating P2P specturm "
        ) as distances:
            for distance in distances:
                if distance >= dist[0] and distance <= dist[-1]:
                    fig, ax = vis.create_figure()
                    for edge in self.positions:
                        # print(f"pos: {edge} dist: {distance}")
                        try:
                            p2p = np.array(self.getdata(distance, edge, "CH")).T
                        except Exception as e:
                            log(
                                f"---> Missing column for pos: {edge} and dist: {distance}: ",
                                wait=True,
                            )
                            log(f"skip", "yellow")
                        else:
                            vis.add_plot(
                                ax,
                                p2p[0],
                                p2p[1],
                                xlabel="Channel [V]",
                                ylabel="Counts",
                                title=f"Peak to Peak. Distance: {distance}cm",
                                legend=True,
                                grid=True,
                                fmt="-",
                                label=f"CH0-{edge}",
                                alpha=0.8,
                            )
                            vis.add_plot(
                                ax,
                                p2p[0],
                                p2p[2],
                                xlabel="Channel [V]",
                                ylabel="Counts",
                                title=f"Peak to Peak. Distance: {distance}cm",
                                legend=True,
                                grid=True,
                                fmt="-",
                                label=f"CH1-{edge}",
                                alpha=0.8,
                            )
                            if fit:
                                x0_fit, y0_fit, params0, pcov0 = optim.fit(
                                    optim.landau, p2p[0], p2p[1]
                                )
                                x1_fit, y1_fit, params1, pcov1 = optim.fit(
                                    optim.landau, p2p[0], p2p[2]
                                )
                                landaupeaks[edge].append(
                                    [
                                        distance,
                                        params0[0],
                                        params1[0],
                                        params0[1],
                                        params1[1],
                                    ]
                                )
                                vis.add_plot(
                                    ax,
                                    x0_fit,
                                    y0_fit,
                                    xlabel="Channel [V]",
                                    ylabel="Counts",
                                    title=f"Peak to Peak. Distance: {distance}cm",
                                    legend=True,
                                    grid=True,
                                    fmt="--",
                                    label=None,
                                    alpha=0.8,
                                )
                                vis.add_plot(
                                    ax,
                                    x1_fit,
                                    y1_fit,
                                    xlabel="Channel [V]",
                                    ylabel="Counts",
                                    title=f"Peak to Peak. Distance: {distance}cm",
                                    legend=True,
                                    grid=True,
                                    fmt="--",
                                    label=None,
                                    alpha=0.8,
                                )

                            #### asym
                            as1, as2, as3, cc = self.get_assymetry(p2p)
                            asym_data[edge].append(
                                [distance, as1, as2, as3, cc[0], cc[1]]
                            )
                            ####
                    vis.save(fig, f"{distance}_{filename}")
        asym = {}
        asym["PPA"] = np.array(asym_data["PPA"]).T
        pp = ["$\\frac{c_1-c_2}{c_1+c_2}$", "$\\sqrt{c_1c_2}$", "$ln\\frac{c_1}{c_2}$"]
        figg, axx = vis.create_figure()
        for i, j in enumerate(asym.keys()):
            vis.add_plot(
                axx,
                asym[j][0],
                asym[j][1],
                xlabel="Distance [cm]",
                ylabel=pp[0],
                legend=True,
                grid=True,
                fmt="o",
                label=f"{j}",
                alpha=0.8,
            )
        vis.save(figg, "asym.png")
        figg, axx = vis.create_figure()
        for i, j in enumerate(asym.keys()):
            vis.add_plot(
                axx,
                asym[j][0],
                asym[j][2],
                xlabel="Distance [cm]",
                ylabel=pp[1],
                legend=True,
                grid=True,
                fmt="o",
                label=f"{j}",
                alpha=0.8,
            )
        vis.save(figg, "sqrt.png")
        figg, axx = vis.create_figure()
        for i, j in enumerate(asym.keys()):
            vis.add_plot(
                axx,
                asym[j][0],
                asym[j][3],
                xlabel="Distance [cm]",
                ylabel=pp[2],
                legend=True,
                grid=True,
                fmt="o",
                label=f"{j}",
                alpha=0.8,
            )

        lxfit, lyfit, lparams, lpcovv = optim.fit(
            optim.linear, asym["PPA"][0], asym["PPA"][3]
        )
        vis.add_plot(
            axx,
            lxfit,
            lyfit,
            xlabel="Distance [cm]",
            ylabel=pp[2],
            legend=True,
            grid=True,
            fmt="--",
            label=f"fit",
            alpha=0.8,
        )
        log(
            f"---> FIT ln(c1/c2): a = {lparams[0]} +- {lpcovv[0,0]**0.5}, b = {lparams[1]} +- {lpcovv[1,1]**0.5}\n lambda = {round(2/lparams[0],2)} +- {round(2/(lparams[0])**2 * lpcovv[0,0]**0.5,2)}",
            "green",
        )
        vis.save(figg, "ln.png")
        figg, axx = vis.create_figure()
        for i, j in enumerate(asym.keys()):
            vis.add_plot(
                axx,
                asym[j][0],
                asym[j][4],
                xlabel="Distance [cm]",
                ylabel="Channel [V]",
                legend=True,
                grid=True,
                fmt="o",
                label=f"CH0 - {j}",
                alpha=0.8,
            )
            vis.add_plot(
                axx,
                asym[j][0],
                asym[j][5],
                xlabel="Distance [cm]",
                ylabel="Channel [V]",
                legend=True,
                grid=True,
                fmt="o",
                label=f"CH1 - {j}",
                alpha=0.8,
            )
        vis.save(figg, "c1c2.png")

        if fit:
            land = {}
            land["C"] = np.array(landaupeaks["C"]).T
            land["D"] = np.array(landaupeaks["D"]).T
            land["U"] = np.array(landaupeaks["U"]).T
            figgg, axxx = vis.create_figure()
            vis.add_plot(
                axxx,
                land["C"][0],
                land["C"][1],
                yerr=land["C"][3],
                xlabel="Distance [cm]",
                ylabel="Landau Channel [V]",
                legend=True,
                grid=True,
                fmt="o",
                label=f"CH0",
                alpha=0.8,
            )
            vis.add_plot(
                axxx,
                land["C"][0],
                land["C"][2],
                yerr=land["C"][3],
                xlabel="Distance [cm]",
                ylabel="Landau Channel [V]",
                legend=True,
                grid=True,
                fmt="o",
                label=f"CH1",
                alpha=0.8,
            )
            vis.save(figgg, "landau-C.png")
            figgg, axxx = vis.create_figure()
            vis.add_plot(
                axxx,
                land["D"][0],
                land["D"][1],
                yerr=land["D"][3],
                xlabel="Distance [cm]",
                ylabel="Landau Channel [V]",
                legend=True,
                grid=True,
                fmt="o",
                label=f"CH0",
                alpha=0.8,
            )
            vis.add_plot(
                axxx,
                land["D"][0],
                land["D"][2],
                yerr=land["D"][3],
                xlabel="Distance [cm]",
                ylabel="Landau Channel [V]",
                legend=True,
                grid=True,
                fmt="o",
                label=f"CH1",
                alpha=0.8,
            )
            vis.save(figgg, "landau-D.png")
            figgg, axxx = vis.create_figure()
            vis.add_plot(
                axxx,
                land["U"][0],
                land["U"][1],
                yerr=land["U"][3],
                xlabel="Distance [cm]",
                ylabel="Landau Channel [V]",
                legend=True,
                grid=True,
                fmt="o",
                label=f"CH0",
                alpha=0.8,
            )
            vis.add_plot(
                axxx,
                land["U"][0],
                land["U"][2],
                yerr=land["U"][3],
                xlabel="Distance [cm]",
                ylabel="Landau Channel [V]",
                legend=True,
                grid=True,
                fmt="o",
                label=f"CH1",
                alpha=0.8,
            )
            vis.save(figgg, "landau-U.png")

    def det_delay_spectra(self, dist, filename="delay.png"):
        fig, ax = vis.create_figure()
        with click.progressbar(
            self.distances, label=f"---> Calculating delay specturm "
        ) as distances:
            for distance in distances:
                if distance >= dist[0] and distance <= dist[-1]:
                    for edge in self.positions:
                        try:
                            delay = np.array(self.getdata(distance, edge, "T")).T
                        except Exception as e:
                            log(
                                f"---> Missing column for pos: {edge} and dist: {distance}: ",
                                wait=True,
                            )
                            log(f"skip", "yellow")
                        else:
                            vis.add_plot(
                                ax,
                                delay[0],
                                delay[1],
                                xlabel="Delay [ns]",
                                ylabel="Counts",
                                title=f"Delay. Range {dist[0]} - {dist[-1]} cm",
                                legend=True,
                                grid=True,
                                fmt="-",
                                label=f"{distance} {edge}",
                                alpha=0.8,
                            )
        vis.save(fig, f"{dist[0]}-{dist[-1]}_{filename}")

    def get_assymetry(self, p2p):
        # ch0_max = p2p[1].max()
        # ch1_max = p2p[2].max()
        # nc1 = p2p[1].tolist().index(ch0_max)
        # nc2 = p2p[2].tolist().index(ch1_max)
        # c1 = p2p[0][nc1]
        # c2 = p2p[0][nc2]
        c1 = optim.w_avg(p2p[0], p2p[1])
        c2 = optim.w_avg(p2p[0], p2p[2])
        return (
            optim.assymetry(c1, c2),
            optim.chspec(c1, c2),
            optim.chspec_ln(c1, c2),
            (c1, c2),
        )

    def get_teo(self, filename="teo.png", A=1, mu=1):
        fig, ax = vis.create_figure()
        x = np.linspace(-5, 5, 1000)
        c1 = optim.decay_law(x, A, mu)
        c2 = optim.decay_law(x, A, mu)[::-1]
        vis.add_plot(ax, x, c1, label="$c_1 = Ae^{-\\mu x}$", fmt="--")
        vis.add_plot(ax, x, c2, label="$c_2 = Ae^{-\\mu x}$", fmt="--")
        vis.add_plot(ax, x, optim.chspec(c1, c2), label="$\\sqrt{c_1 c_2}$", fmt="--")
        vis.add_plot(
            ax,
            x,
            optim.chspec_ln(c1, c2),
            label="$ln\\frac{c_1}{c_2}$",
            fmt="--",
            legend=True,
            ylim=(-5, 40),
            xlabel="Distance",
        )
        vis.save(fig, filename)
