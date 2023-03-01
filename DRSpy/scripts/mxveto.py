from __future__ import annotations

import os
import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot  as plt

from sys import argv
from DRSpy.plugins.digitizer_reader import *
#from digitizer_reader import *
from scipy.optimize import curve_fit 
from scipy.special import erf
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

        print(f'\n->\tFound:\n\t*\t{len(self.files)} .root files\n\t*\t{self.T_samples=}\n\t*\t{self.channels=}\n')

        self.df_cols = ['event', 'timestamp', 'L', 'CH', 'A', 't_0', 't_r', 't_f', 'Q', 'dV', 'V_0']
        self.df_cols_sigma = ['sig_t_0', 'sig_t_r', 'sig_t_f', 'sig_Q', 'sig_dV', 'sig_V_0']
        self.ddf_cols = ['event', 't_0_ch0', 't_0_ch1', 'L', 'Q_ch0', 'Q_ch1', 'A_ch0', 'A_ch1', 'dt', 'lnQ', 'sqrtQ', 'asymQ', 'lnA', 'sqrtA', 'asymA']

        df_cols_latex = ['evt_ID', 'timestamp [a.u.]', 'L [cm]', 'Channel_ID', 'Amplitude [V]', '$t_0$ [ns]', '$t_r$ [ns]', '$t_f$ [ns]', 'Charge', '$dV$ [V]', '$V_0$ [V]']
        df_cols_sigma_latex = ['$\sigma t_0$', '$\sigma t_r', '$\sigma t_f$', '$\sigma Q$', '$\sigma dV$', '$\sigma V_0$']
        ddf_cols_latex = ['evt_ID', '$t_0^{CH0}$ [ns]', 't_0^{CH1} [ns]', 'L [cm]', 'Q^{CH0}', 'Q^{CH1}', 'Amplitude CH0 [V]', 'Amplitude CH1 [V]', '$t_1^{CH1} - t_0^{CH0}$', '$`\ln{\\frac{Q_1}{Q_0}}$', '$\sqrt{Q_0Q_1}$', '$\\frac{Q_0-Q_1}{Q_1+Q_1}$', '$`\ln{\\frac{A_1}{A_0}}$', '$\sqrt{A_0A_1}$', '$\\frac{A_0-A_1}{A_1+A_1}$']
        # change default df column names to LaTEX
        self.lx = dict(zip(self.df_cols+self.df_cols_sigma+self.ddf_cols, df_cols_latex+df_cols_sigma_latex+ddf_cols_latex))

        self.line_plot_style = {'ls':'-', 'lw':1}

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
                print(f'\t->\tcant decode {filename=}. Skip')
                continue
            waveforms = DigitizerEventData.create_from_file(filename)
            with click.progressbar(range(len(waveforms[0][:wf_num])),label=f'Analyzing waveforms {nfile}/{len(self.files)} ({source_position}cm)\t') as wbar:
                for nevt in wbar:
                    if self.read_limit and nevt > self.read_limit:
                        print('\n\t->\tFile to big. SKIP')
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
        print('->\tCreating DataFrame')
        if isinstance(params_dict, str):
            df = pd.read_csv(params_dict)
        else:
            df = pd.DataFrame(params_dict)
        dcol = ['event','t_0', 'L', 'Q', 'A']
        print('->\tClearing nan and inf values')
        df.replace([np.inf,-np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df.reset_index(inplace=True, drop=True)
        
        print('->\tCalculating rel error')
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

        print('->\tFilltering data...')
        df = df[(df['t_r'] > 0) & (df['t_f'] > 0) & (df['t_r'] < 10) & (df['t_f'] < 10) & (df['t_0'] >20) & (df['t_0'] < 80) & (df['Q'] <= 0) & (df['Q'] > -5)]

        print('->\tCalculating asymeties')
        ddf = df[df['CH']==0][dcol].merge(df[df['CH']==1][dcol], how='inner', on=['event', 'L'], suffixes=['_ch0','_ch1'])
        ddf[self.ddf_cols[8]] = ddf.apply(lambda x: x.t_0_ch1-x.t_0_ch0, axis=1)
        ddf[self.ddf_cols[9]] = ddf.apply(lambda x: np.log(x.Q_ch1/x.Q_ch0), axis=1)
        ddf[self.ddf_cols[10]] = ddf.apply(lambda x: np.sqrt(x.Q_ch1*x.Q_ch0), axis=1)
        ddf[self.ddf_cols[11]] = ddf.apply(lambda x: (x.Q_ch0-x.Q_ch1)/(x.Q_ch0+x.Q_ch1), axis=1)
        ddf[self.ddf_cols[12]] = ddf.apply(lambda x: np.log(x.A_ch1/x.A_ch0), axis=1)
        ddf[self.ddf_cols[13]] = ddf.apply(lambda x: np.sqrt(x.A_ch1*x.A_ch0), axis=1)
        ddf[self.ddf_cols[14]] = ddf.apply(lambda x: (x.A_ch0-x.A_ch1)/(x.A_ch0+x.A_ch1), axis=1)
        print('->\tFilltering data...')
        ddf = ddf[(np.abs(ddf['asym'])<4) & (np.abs(ddf['ln']))<4]
        return df, ddf

    def get_waveforms(self, w0, w1, fit_func, note) -> None:
        g = sns.lineplot(x=self.t, y=w0, color='green', label='CH0', **self.line_plot_style)
        sns.lineplot(x=self.t, y=w1, color='red', label='CH1', **self.line_plot_style)
        sns.lineplot(x=self.t, y=fit_func(t,*p0), color='green', **self.line_plot_style)
        sns.lineplot(x=self.t, y=fit_func(t,*p1), color='red', **self.line_plot_style)
        g.annotate(note, size=10, xy={0,np.min(np.append(w0, w1)/2)} )
        #g.axvline(30, ls='--', lw=0.9, c='red')
        plt.show()


    def get_delay(self, fit_func):
        df = self.ddf.groupby('L').mean().reset_index()
        p, q = curve_fit(fit_func, df['L'], df['dt'])
        c = round(-2/p[0],2)
        print('f->\tFound {c=} +- cm/ns')
        sns.lineplot(data=self.df, x='L', y='dt', **self.line_plot_style)
        sns.lineplot(x=df['L'], y=fit_func(df['L'], *p), **self.line_plot_style, ls='--', color='black')
        plt.show()
        sns.histplot(data=self.df, x='L', y='dt', cbar=True)
        sns.lineplot(x=df['L'], y=fit_func(df['L'], *p), **self.line_plot_style, ls='--', color='black')
        plt.ylim(-20,20)
        plt.show()

    def get_signal(self, fit_func):
        df = self.ddf.groupby('L').mean().reset_index()
        p, q = curve_fit(fit_func, df['L'], df['lnQ'])
        lm = round(2/p[0],2)
        print('f->\tFound (Q) {lm=} +- cm')
        sns.lineplot(data=self.df, x='L', y='lnQ', **self.line_plot_style)
        sns.lineplot(x=df['L'], y=fit_func(df['L'], *p), **self.line_plot_style, ls='--', color='black')
        plt.show()
        sns.histplot(data=self.df, x='L', y='lnQ', cbar=True)
        sns.lineplot(x=df['L'], y=fit_func(df['L'], *p), **self.line_plot_style, ls='--', color='black')
        plt.ylim(-4,4)
        plt.show()

        p, q = curve_fit(fit_func, df['L'], df['lnA'])
        lm = round(2/p[0],2)
        print('f->\tFound (A) {lm=} +- cm')
        sns.lineplot(data=self.df, x='L', y='lnA', **self.line_plot_style)
        sns.lineplot(x=df['L'], y=fit_func(df['L'], *p), **self.line_plot_style, ls='--', color='black')
        plt.show()
        sns.histplot(data=self.df, x='L', y='lnA', cbar=True)
        sns.lineplot(x=df['L'], y=fit_func(df['L'], *p), **self.line_plot_style, ls='--', color='black')
        plt.ylim(-4,4)
        plt.show()

        sns.lineplot(data=self.df, x='L', y='sqrtQ', **self.line_plot_style, color='black')
        plt.show()
        sns.lineplot(data=self.df, x='L', y='asymQ', **self.line_plot_style, color='black')
        plt.show()
        sns.lineplot(data=self.df, x='L', y='sqrtA', **self.line_plot_style, color='black')
        plt.show()
        sns.lineplot(data=self.df, x='L', y='asymA', **self.line_plot_style, color='black')
        plt.show()
        sns.lineplot(data=self.df, x='L', y='Q_ch0', **self.line_plot_style, color='red')
        sns.lineplot(data=self.df, x='L', y='Q_ch1', **self.line_plot_style, color='blue')
        plt.show()
        sns.lineplot(data=self.df, x='L', y='A_ch0', **self.line_plot_style, color='red')
        sns.lineplot(data=self.df, x='L', y='A_ch1', **self.line_plot_style, color='blue')
        plt.show()

    def get_time(self):
        sns.jointplot(data=self.ddf, x='t_0_ch0', y='t_0_ch1',kind='hist')
        plt.show()
        sns.jointplot(data=ddf,x='dt', y='ln',kind='hist')
        plt.show()
        sns.jointplot(data=ddf,x='t_r', y='t_f',kind='hist')
        plt.show()


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

def asymgauss_fit(x, m0, m1, m2):
    amp = (m0 / (m2 * np.sqrt(2 * np.pi)))
    spread = np.exp((-(x - m1)**2 ) / (2 * m2**2))
    skew = (1 + erf((m3 * (x - m1)) / (m2 * np.sqrt(2))))
    return amp*spread*skew

def lin_fit(x, a, b):
    return a*x+b

def landau_fit(x, E, S, N):
    return N/np.sqrt(2*np.pi) * np.exp((((E-x)/S)-np.exp(-((x-E)/S))) /2)

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
