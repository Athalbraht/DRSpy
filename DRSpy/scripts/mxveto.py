from __future__ import annotations

import os
import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot  as plt

from sys import argv
#from DRSpy.plugins.digitizer_reader import *
from digitizer_reader import *
from scipy.optimize import curve_fit 
from scipy import stats

from pathlib import Path
from typing import *

sns.set_theme()

class Analysis():
    def __init__(self, data_path: str, limit_file: str|None=None, charts_path: Path|None=None) -> None:
        self.files = []
        self.df = pd.DataFrame()
        self.ddf = pd.DataFrame()

        self.data_path = Path(data_path)
        if charts_path:
            self.charts_path = Path(charts_path)
            if self.charts_path.exists(): self.charts_path.mkdir()
        else:
            self.charts_path = False
        self.read_limit = len(DigitizerEventData.create_from_file(limit_file)[0]) if limit_file else 0

        choice = 0
        for file in self.data_path.iterdir() :
            if file.suffix == '.df' and not choice:
                print('---------------------------------')
                df_files = {f'{i}':df_file for i, df_file in enumerate(self.data_path.iterdir()) if df_file.suffix == '.df'}
                for df_file in df_files: print(f'\tID:{df_file}\t{df_files[df_file].name}')
                print('---------------------------------')
                print(f'->\tFound DataFrame files in {self.data_path}.\n\tLoad file? [<ID>/N]')
                choice = input('Choice: ')
                if df_files.get(choice, False):
                    print('->\tLoading DataFrames...')
                    self.df = pd.read_pickle(df_files[choice].absolute())
                    self.ddf = pd.read_pickle(df_files[choice].with_suffix('.ddf').absolute())
            if file.suffix == '.root':
                self.files.append(self.data_path.joinpath(file))

        evt = DigitizerEventData.create_from_file(self.files[0])
        self.channels = len(evt)
        self.T_samples = len(evt[0][0].waveform)
        self.t = 0.2 * np.arange(0, self.T_samples)

        print(f'\nFound {len(self.files)} .root files. {self.T_samples=}, {self.channels=} \n')

        self.df_cols = ['event', 'timestamp', 'L', 'CH', 'A', 't_0', 't_r', 't_f', 'Q', 'dV', 'V_0']
        self.df_cols_sigma = ['sig_t_0', 'sig_t_r', 'sig_t_f', 'sig_Q', 'sig_dV', 'sig_V_0']
        self.ddf_cols = ['event', 't_0_ch0', 't_0_ch1', 'L', 'Q_ch0', 'Q_ch1', 'dt', 'ln', 'sqrt', 'asym']

        df_cols_latex = ['evt_ID', 'timestamp [a.u.]', 'L [cm]', 'Channel_ID', 'Amplitude [V]', '$t_0$ [ns]', '$t_r$ [ns]', '$t_f$ [ns]', 'Charge', '$dV$ [V]', '$V_0$ [V]']
        df_cols_sigma_latex = ['$\sigma t_0$', '$\sigma t_r', '$\sigma t_f$', '$\sigma Q$', '$\sigma dV$', '$\sigma V_0$']
        ddf_cols_latex = ['evt_ID', '$t_0^{CH0}$ [ns]', 't_0^{CH1} [ns]', 'L [cm]', 'Q^{CH0}', 'Q^{CH1}', '$t_1^{CH1} - t_0^{CH0}$', '$\ln{\\frac{C_1}{C_0}}$', '$\sqrt{C_0C_1}$', '$\\frac{C_0-C_1}{C_1+C_1}$']
        # change default df column names to LaTEX
        self.lx = dict(zip(self.df_cols+self.df_cols_sigma+self.ddf_cols, df_cols_latex+df_cols_sigma_latex+ddf_cols_latex))

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

    def multi_loader(self, filename_decoder: Callable[str, float|bool], fit_func: Callable[Any, float], wf_num:int=-1) -> pd.DataFrame|None:
        evt_counter = 0
        p_list = [ [] for i in range(len(self.df_cols)) ]
        q_list = [ [] for i in range(len(self.df_cols)) ]
        for nfile, filename in enumerate(self.files):
            source_position = filename_decoder(filename.name)
            if not isinstance(source_position, float):
                print(f"cant decode {filename=}. Skip")
                continue
            waveforms = DigitizerEventData.create_from_file(filename)
            with click.progressbar(range(len(waveforms[0][:wf_num])),label=f'Analyzing waveforms {nfile}/{len(self.files)} ({source_position}cm)\t') as wbar:
                for nevt in wbar:
                    if self.read_limit and nevt > self.read_limit:
                        print(f'\tFile to big. SKIP')
                        break
                    evt_counter += 1
                    for channel in range(self.channels):
                        p_list[0].append(evt_counter)
                        p_list[1].append(waveforms[channel][nevt].timestamp)
                        p_list[2].append(source_position)
                        p_list[3].append(channel)
                        p_list[4].append(waveforms[channel][nevt].amplitude)
                        p,q = self.get_waveform_fit(fit_func, waveforms[channel][nevt].waveform) 
                        for i in range(len(p)):
                            p_list[i+len(self.df_cols)-len(self.df_cols_sigma)].append(p[i])
                            q_list[i].append(np.sqrt(np.diag(q))[i])
        params_dict = dict(zip(self.df_cols+self.df_cols_sigma, p_list+q_list))
        return params_dict

    def to_pandas(self, params_dict: Dict|str) -> Tuple[pd.DataFrame]:
        print('-> Creating DataFrame')
        if isinstance(params_dict, str):
            df = pd.read_csv(params_dict)
        else:
            df = pd.DataFrame(params_dict)
        dcol = ['event','t_0', 'L', 'Q']
        print('-> Clearing nan and inf values')
        df.replace([np.inf,-np.inf], np.nan, inplace=True)
        df = df.dropna()
        df = df.reset_index()
        del df['index']
        
        print('-> Calculating rel error')
        tmp = [[] for i in range(len(self.df_cols_sigma)-1)]
        with click.progressbar(range(len(df))) as bar:
            for i in bar:
                for j in range(len(tmp)):
                    diff = round(df.loc[i,df.columns[len(self.df_cols)+1+j]]/df.loc[i,df.columns[(len(self.df_cols)-len(self.df_cols_sigma)+1)+j]],4)
                    if np.abs(diff) > 1 or diff == np.nan or np.abs(diff) == np.inf:
                        tmp[j].append(1)
                    else:
                        tmp[j].append(diff)
        tmp = dict(zip('rel_'+df.columns[(len(self.df_cols)-len(self.df_cols_sigma)+1):(len(self.df_cols)+1)], tmp))
        for i in tmp.keys():
            df[i] = tmp[i]

        print('-> Filltering data...')
        df = df[(df['t_r'] > 0) & (df['t_f'] > 0) & (df['t_r'] < 10) & (df['t_f'] < 10) & (df['t_0'] >20) & (df['t_0'] < 80) & (df['Q'] <= 0) & (df['Q'] > -5)]

        print('-> Calculating asymeties')
        df_0 = df[df['CH']==0][dcol]
        df_1 = df[df['CH']==1][dcol]
        ddf = df_0.merge(df_1, how='inner', on=['event', 'L'], suffixes=['_ch0','_ch1'])
        ddf[self.ddf_cols[6]] = ddf.apply(lambda x: x.t_0_ch1-x.t_0_ch0, axis=1)
        ddf[self.ddf_cols[7]] = ddf.apply(lambda x: np.log(x.Q_ch1/x.Q_ch0), axis=1)
        ddf[self.ddf_cols[8]] = ddf.apply(lambda x: np.sqrt(x.Q_ch1*x.Q_ch0), axis=1)
        ddf[self.ddf_cols[9]] = ddf.apply(lambda x: (x.Q_ch0-x.Q_ch1)/(x.Q_ch0+x.Q_ch1), axis=1)
        print('-> Filltering data...')
        ddf = ddf[(np.abs(ddf['asym'])<0) & (np.abs(ddf['ln']))<1 ]
        return df, ddf

    def get_waveforms(self,):
        pass

    def get_delay(self):
        pass

    def get_signal(self):
        pass

    def get_waveform_fit(self, fit_func: Callable[Any, float], waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

if __name__ == '__main__':
    print('Usage: python analysis.py <data_folder>')
    run = Analysis(argv[1])
    params = run.multi_loader(run.decode_filename, sig_fit)
