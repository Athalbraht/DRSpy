from __future__ import annotations

import os
import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot  as plt

from sys import argv
from DRSpy.plugins.digitizer_reader import *
from scipy.optimize import curve_fit 
from scipy import stats

from pathlib import Path
from typing import *

sns.set_theme()

class Analysis():
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.files = []
        for file in self.data_path.iterdir():
            if file.suffix == '.root':
                self.files.append(self.data_path.joinpath(file))
        evt = DigitizerEventData.create_from_file(self.files[0])
        self.T_samples = len(evt[0][0].waveform)
        self.t = 0.2 * np.arange(0, self.T_samples)

        self.channels = len(evt)
        self.df = pd.DataFrame()
        print(f'\nFound {len(self.files)} .root files. {self.T_samples=}, {self.channels=} \n')

        self.fit_params = ['event', 'L', 'CH', 't_0', 't_r', 't_f', 'Q', 'dV', 'V_0']
        self.fit_sig = ['sig_t_0', 'sig_t_r', 'sig_t_f', 'sig_Q', 'sig_dV', 'sig_V_0']
        self.fit_params2 = ['L [cm]', 'CH', '$t_0$', '$t_r$', '$t_f$', '$Q$', '$dV$', '$V_0$']
        self.fit_sig2 = ['$\sigma t_0$', '$\sigma t_r', '$\sigma t_f$', '$\sigma Q$', '$\sigma dV$', '$\sigma V_0$']

    def decode_filename(self, filename: str) -> float|bool:
        try:
            position = float(filename.split('_')[1])
            return position
        except: 
            return False


    def load_waveform(self, root_filename: Path) -> List[Any]:
        events = DigitizerEventData.create_from_file(root_filename)
        waveforms = [] #waveform[channel] = [waveforms list, amplitudes list]
        for channel in range(self.channels):
            waveforms.append([])
            waveforms[channel].append([event.waveform for event in events[channel]])
            waveforms[channel].append([event.amplitude for event in events[channel]])
        return waveforms

    def multi_loader(self, filename_decoder: Callable[str, float|bool], fit_func: Callable[Any, float]) -> pd.DataFrame|None:
        evt_counter = 0
        p_list = [ [] for i in range(len(self.fit_params)) ]
        q_list = [ [] for i in range(len(self.fit_params)) ]
        for nfile, filename in enumerate(self.files):
            source_position = filename_decoder(filename.name)
            if not source_position:
                print(f"cant decode {filename=}. Skip")
                continue
            #waveforms = self.load_waveform(filename)
            waveforms = DigitizerEventData.create_from_file(filename)
            with click.progressbar(range(len(waveforms[0])),label=f'Analyzing waveforms {nfile}/{len(self.files)} ({source_position}cm)') as wbar:
                for nevt in wbar:
                    evt_counter += 1
                    for channel in range(self.channels):
                        p_list[0].append(evt_counter)
                        p_list[1].append(source_position)
                        p_list[2].append(channel)
                        p,q = self.get_waveform_fit(fit_func, waveforms[channel][nevt].waveform) 
                        for i in range(len(p)):
                            p_list[i+3].append(p[i])
                            q_list[i].append(np.sqrt(np.diag(q))[i])
                        
                        #sns.lineplot(x=self.t, y=waveforms[0][nevt].waveform, label='CH0')
                        #sns.lineplot(x=self.t, y=waveforms[1][nevt].waveform, label='CH1')
                        #sns.lineplot(x=self.t, y=fit_func(self.t,*p0))
                        #sns.lineplot(x=self.t, y=fit_func(self.t,*p1))
                        #plt.show()
        params_dict = dict(zip(self.fit_params+self.fit_sig, p_list+q_list))
        #return params_dict
        return params_dict

    def get_waveform_fit(self, fit_func: Callable[Any, float], waveform: np.ndarray):
        get_t0 = lambda t_m, t_r, t_f: t_m - t_r * t_f / (t_f - t_r) * np.log(t_f/t_r)
        t_max = self.t[np.argmin(waveform)]
        t_r = 2.9
        t_f = 3
        t0 = get_t0(t_max, t_r, t_f)
        V0 = np.mean(waveform[:10])
        dV = (np.mean(waveform[:-10]) - V0)/self.t[-1]
        Q = np.sum(waveform*0.2) - V0*self.t[-1] - 0.5 * dV*self.t[-1]**2

        p0=[t0, t_r, t_f, Q, dV, V0]
        try:
            p,q = curve_fit(fit_func, self.t, waveform, p0=p0)
            return p, q
        except: 
            return np.full((len(p0)), np.inf), np.full((len(p0),len(p0)), np.inf)

        


def get_t0(t_m, t_r, t_f):
    return t_m - t_r * t_f / (t_f - t_r) * np.log(t_f/t_r)
    #return t_m - t_r * np.log(t_f/t_r+1)

def sig_fit(t, t0, t_r, t_f, Q, dV, V0=0):
	return V0 + dV*t + np.heaviside(t-t0, 0) * Q/(t_r-t_f) * (np.exp(-(t-t0)/t_r) - np.exp(-(t-t0)/t_f)) 
	#return V0 + dV*t + np.heaviside(t-t0, 0) * Q/t_f * (1 + t_r/t_f) * (1 - np.exp(-(t-t0)/t_r)) * np.exp(-(t-t0)/t_f)

def get_wf_params(root_filename):
    events = DigitizerEventData.create_from_file(root_filename)
    waveforms_channel = [[event.waveform for event in events[0]], [event.waveform for event in events[1]]]
    with click.progressbar(range(len(waveforms_channel[0]))) as bar:
        p_tab = [[],[]]
        p_cov = [[],[]]
        for nwf in bar:
            tmp_p = [False, False]
            tmp_c = [False, False]
            for nchannel in range(2):
                wf = waveforms_channel[nchannel][nwf]
                _w = 5
                wf = np.convolve(wf, np.ones(_w), 'valid')/_w
                T = 0.2 * np.arange(0, len(wf))
                t_max = T[np.argmin(wf)]
                t_r = 2.9
                t_f = 3
                t0 = get_t0(t_max, t_r, t_f)

                V0 = np.mean(wf[nwf:nwf+10])
                dV = (np.mean(wf[-10-nwf:]) - V0)/T[-1]
                Q = np.sum(wf*0.2) - V0*T[-1] - 0.5 * dV*T[-1]**2

                try:
                    p0=[t0, t_r, t_f, Q, dV, V0]
                    p,q = curve_fit(sig_fit, T, wf, p0=p0)
                    if np.any(np.diag(q) != np.inf):
                        tmp_p[nchannel] = list(p)
                        tmp_c[nchannel] = list(np.sqrt(np.diag(q)))

                    if not argv[1] == 'quiet':
                        plt.plot(T, sig_fit(T,*p), label="fit")
                        plt.plot(T, wf, label="waveform")
                        plt.legend()
                        plt.title(f'Waveform no. {nwf} {nchannel=}')
                        print(f"\nFit params: {p0[nchannel]=}")
                        print(f"\nFit params: {p[nchannel]=}")
                        plt.show()
                except:
                    pass
            if tmp_p[0] and tmp_p[1]:
                p_tab[0].append(tmp_p[0])
                p_tab[1].append(tmp_p[1])
                p_cov[0].append(tmp_c[0])
                p_cov[1].append(tmp_c[1])


        ##### TMP

        p_tab = [np.array(p_tab[0]).T, np.array(p_tab[1]).T]
        p_cov = [np.array(p_cov[0]).T, np.array(p_cov[1]).T]

        plt.clf()
        if argv[2] == 'trf':
            plt.hist(p_tab[0][1],bins=100, range=[0,10], label='t_r (0)', alpha=0.3)
            plt.hist(p_tab[0][2],bins=100, range=[0,10], label='t_f (0)', alpha=0.3)
            plt.hist(p_tab[1][1],bins=100, range=[0,10], label='t_r (1)', alpha=0.3)
            plt.hist(p_tab[1][2],bins=100, range=[0,10], label='t_f (1)', alpha=0.3)
            plt.legend()
            plt.show()
            plt.clf()
            plt.hist(p_tab[0][0],bins=100, range=[0,10], label='t0 (0)', alpha=0.3)
            plt.hist(p_tab[1][0],bins=100, range=[0,10], label='t0 (1)', alpha=0.3)
            plt.legend()
            plt.show()
            plt.clf()


        hp = plt.hist(p_tab[1][0] - p_tab[0][0] ,bins=100, range=[-10,10], label='t1-t0', alpha=0.3,density=True)
        hm = np.sum(hp[1][:-1]*hp[0])/np.sum(hp[0])# np.mean(p_tab[1][0]-p_tab[0][0])
        #plt.title(f't1-t0 ~ {hm}')
        #plt.legend()
        #plt.show()

        ## GET STATS
        
        dt0 = 1/(p_cov[0]**2 + p_cov[1]**2)
        #t0_delay = np.sum()/np.sum()

        t0_0 = p_tab[0][0]
        dt0_0 = p_cov[0][0] * nstd
        t0_1 = p_tab[1][0]
        dt0_1 = p_cov[1][0] * nstd

        Q_0 = p_tab[0][3]
        dQ_0 = p_cov[0][3] * nstd
        Q_1 = p_tab[1][3]
        dQ_1 = p_cov[1][3] * nstd

        return [t0_0, dt0_0, t0_1, dt0_1, Q_0, dQ_0, Q_1, dQ_1]

def decode_filename(filename):
    return int(filename.split('_')[1])

def main(path):
    files = os.listdir(path)
    L = []
    T = []
    with click.progressbar(files) as bar:
        for _file in bar:
            L.append(decode_filename(_file))
            T.append(get_wf_params(os.path.join(argv[3], _file)))
    return L, T


if __name__ == '__main__':
    #get_wf_params('sipm-tests2/mxveto_20_C.root')
    #L,T = main(argv[3])
    print('Usage: python analysis.py <data_folder>')
    run = Analysis(argv[1])
    params = run.multi_loader(run.decode_filename, sig_fit)
