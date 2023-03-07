from __future__ import annotations

__version__ = 'v0.1'

import os
import toml
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
        '''
        data_path:      path to folder with .root files
        limit_file:     path to .root file. Reading events from other files will be limited to limit_file size.
        charts_path:    path to plots output folder
        '''
        # init params
        self.files = []
        self.df = pd.DataFrame()
        self.ddf = pd.DataFrame()
        self.data_path = Path(data_path)
        # Check charts_path
        if charts_path:
            self.charts_path = Path(charts_path)
            for folder in ['waveforms', 'time', 'asymmetry', 'stats']:
                self.charts_path.joinpath(folder).mkdir(exist_ok=True, parents=True)
        else: self.charts_path = False
        # Check limit_file
        self.read_limit = len(DigitizerEventData.create_from_file(limit_file)[0]) if limit_file else 0
        # Read self.data_path
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
        # Get waveform properties
        evt = DigitizerEventData.create_from_file(self.files[0])
        self.channels = len(evt)
        self.T_samples = len(evt[0][0].waveform)
        self.t = 0.2 * np.arange(0, self.T_samples)
        # Define graphs styling
        self.chart_ext = '.pdf'
        sns.set_theme()
        self.line_plot_style = {'ls':'-', 'lw':1}
        self.fitline_plot_style = {'ls':'--', 'lw':1}
        self.scatter_plot_style = {'s':10}
        # ADD translator func
        # Define DataFrame column names
        self.df_cols = ['event', 'timestamp', 'L', 'CH', 'A', 't_0', 't_r', 't_f', 'Q', 'dV', 'V_0']
        self.df_cols_sigma = ['sig_t_0', 'sig_t_r', 'sig_t_f', 'sig_Q', 'sig_dV', 'sig_V_0']
        self.ddf_cols = ['event', 't_0_ch0', 't_0_ch1', 'L', 'Q_ch0', 'Q_ch1', 'A_ch0', 'A_ch1', 'dt', 'lnQ', 'sqrtQ', 'asymQ', 'lnA', 'sqrtA', 'asymA']
        self.df_cols_latex = ['evt_ID', 'timestamp', 'L [cm]', 'Channel', 'Amp. [V]', '$t_0$ [ns]', '$t_r$ [ns]', '$t_f$ [ns]', 'Charge', '$dV$ [V]', '$V_0$ [V]']
        self.df_cols_sigma_latex = ['$\sigma t_0$', '$\sigma t_r', '$\sigma t_f$', '$\sigma Q$', '$\sigma dV$', '$\sigma V_0$']
        self.ddf_cols_latex = ['evt_ID', '$t_0^{CH0}$ [ns]', '$t_0^{CH1}$ [ns]', 'L [cm]', '$Q^{CH0}$', '$Q^{CH1}$', 'Amp. CH0 [V]', 'Amp. CH1 [V]', '$t_1^{CH1} - t_0^{CH0}$', '$\ln{\\frac{Q_1}{Q_0}}$', '$\sqrt{Q_0Q_1}$', '$\\frac{Q_0-Q_1}{Q_1+Q_1}$', '$\ln{\\frac{A_1}{A_0}}$', '$\sqrt{A_0A_1}$', '$\\frac{A_0-A_1}{A_1+A_1}$']
        # change default df column names to LaTEX
        self.lx = dict(zip(self.df_cols+self.df_cols_sigma+self.ddf_cols, self.df_cols_latex+self.df_cols_sigma_latex+self.ddf_cols_latex))

        self.toml_manager()
        print(f'\n->\tFound:\n\t\t* {len(self.files)} .root files\n\t\t* {self.T_samples=}\n\t\t* {self.channels=}\n')

    def toml_manager(self) -> None:
        '''Init and read config.toml'''
        print('\->\tChecking config.toml...')
        self.config_path = Path(self.data_path.joinpath('config.toml'))
        if self.config_path.exists():
            self.data = toml.load(self.config_path)
        else:
            print('\t->File not found. Initialization...')
            self.data = {'metadata' :   {},
                         'filters'  :   {},
                         'style'    :   {},
                         'history'  :   {},
                         'progress' :   {},}
            self.config_path.touch()
            self.toml.data['mxveto_version'] = __version__
            self.dump(self.config_path)

    def decode_filename(self, filename: str, separator: str='_', pos: int=1) -> float|bool:
        '''Extract position data from filename'''
        try:
            position = float(filename.split(separator)[pos])
            return position
        except: 
            return False

    def load_waveform(self, root_filename: Path) -> List[Any]:
        '''TMP'''
        events = DigitizerEventData.create_from_file(root_filename)
        waveforms = [] #waveform[channel] = [waveforms list, amplitudes list]
        for channel in range(self.channels):
            waveforms.append([])
            waveforms[channel].append([event.waveform for event in events[channel]])
            waveforms[channel].append([event.amplitude for event in events[channel]])
        return waveforms

    def load_waveforms(self, filename_decoder: Callable[str, float|bool], fit_func: Callable[Any, float], wf_num:int=-1) -> pd.DataFrame:
        '''Load waveforms from self.data_path'''
        evt_counter = 0
        p_list = [ [] for i in range(len(self.df_cols)) ] # params & cov list
        q_list = [ [] for i in range(len(self.df_cols)) ]
        for nfile, filename in enumerate(self.files):
            source_position = filename_decoder(filename.name)
            if not isinstance(source_position, float):
                print(f'\n\t->\tcant decode {filename=}. Skip')
                continue
            waveforms = DigitizerEventData.create_from_file(filename)
            plot_counter = 10
            with click.progressbar(range(len(waveforms[0][:wf_num])),label=f'Analyzing waveforms {nfile}/{len(self.files)} ({source_position}cm)\t') as wbar:
                for nevt in wbar:
                    if self.read_limit and nevt > self.read_limit:
                        print('\n\t->\tFile is to big. SKIP')
                        break
                    evt_counter += 1
                    for channel in range(self.channels):
                        # FIX
                        p_list[0].append(evt_counter)
                        p_list[1].append(waveforms[channel][nevt].timestamp)
                        p_list[2].append(source_position)
                        p_list[3].append(channel)
                        p_list[4].append(waveforms[channel][nevt].amplitude)
                        p,q = self.get_waveform_fit(fit_func, waveforms[channel][nevt].waveform) 
                        for i in range(len(p)):
                            p_list[i+len(self.df_cols)-len(self.df_cols_sigma)].append(p[i])
                            q_list[i].append(np.sqrt(np.diag(q))[i])
                    if plot_counter != 0:
                        plot_counter -= 1
                        plot_df = pd.DataFrame(dict(zip(self.df_cols_latex, [i[-2:] for i in p_list])))
                        try:
                            self.plot_waveforms(waveforms[0][nevt].waveform, waveforms[1][nevt].waveform, fit_func, f'Waveform $L={source_position}$cm', plot_df)
                        except:
                            plt.clf()
                            plt.cla()
        params_dict = dict(zip(self.df_cols+self.df_cols_sigma, p_list+q_list))
        return pd.DataFrame(params_dict)

    def prepare(self, raw_df: pd.DataFrame|str) -> Tuple[pd.DataFrame]:
        '''Filter raw DataFrame and extract only useful entries. The method generates filtered df with relative errors, groups by event and source position (ddf) and '''
        print('->\tCreating DataFrame')
        if isinstance(raw_df, str):
            self.df = pd.read_csv(raw_df)
        else:
            self.df = pd.DataFrame(raw_df)
        dcol = ['event','t_0', 'L', 'Q', 'A']
        print('->\tClearing nan and inf values')
        self.df.replace([np.inf,-np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
        self.df.reset_index(inplace=True, drop=True)
        
        print('->\tCalculating rel error')
        tmp = [[] for i in range(len(self.df_cols_sigma)-1)]
        with click.progressbar(range(len(self.df))) as bar:
            for i in bar:
                for j in range(len(tmp)):
                    diff = round(self.df.loc[i,self.df.columns[len(self.df_cols)+1+j]]/self.df.loc[i,self.df.columns[(len(self.df_cols)-len(self.df_cols_sigma)+1)+j]],4)
                    if np.abs(diff) > 1 or diff == np.nan or np.abs(diff) == np.inf:
                        tmp[j].append(1)
                    else:
                        tmp[j].append(diff)
        tmp = dict(zip('rel_'+self.df.columns[(len(self.df_cols)-len(self.df_cols_sigma)+1):(len(self.df_cols)+1)], tmp))
        for i in tmp.keys():
            self.df[i] = tmp[i]

        print('->\tFiltering df data...')
        self.df = self.df[(self.df['t_r'] > 0) & (self.df['t_f'] > 0) & (self.df['t_r'] < 10) & (self.df['t_f'] < 10) & (self.df['t_0'] >20) & (self.df['t_0'] < 80) & (self.df['Q'] <= 0) & (self.df['Q'] > -5)]

        print('->\tCalculating asymmeties')
        self.ddf = self.df[self.df['CH']==0][dcol].merge(self.df[self.df['CH']==1][dcol], how='inner', on=['event', 'L'], suffixes=['_ch0','_ch1'])
        self.ddf[self.ddf_cols[8]] = self.ddf.apply(lambda x: x.t_0_ch1-x.t_0_ch0, axis=1)
        self.ddf[self.ddf_cols[9]] = self.ddf.apply(lambda x: np.log(x.Q_ch1/x.Q_ch0), axis=1)
        self.ddf[self.ddf_cols[10]] = self.ddf.apply(lambda x: np.sqrt(x.Q_ch1*x.Q_ch0), axis=1)
        self.ddf[self.ddf_cols[11]] = self.ddf.apply(lambda x: (x.Q_ch0-x.Q_ch1)/(x.Q_ch0+x.Q_ch1), axis=1)
        self.ddf[self.ddf_cols[12]] = self.ddf.apply(lambda x: np.log(x.A_ch1/x.A_ch0), axis=1)
        self.ddf[self.ddf_cols[13]] = self.ddf.apply(lambda x: np.sqrt(x.A_ch1*x.A_ch0), axis=1)
        self.ddf[self.ddf_cols[14]] = self.ddf.apply(lambda x: (x.A_ch0-x.A_ch1)/(x.A_ch0+x.A_ch1), axis=1)
        print('->\tFiltering ddf data...')
        self.ddf = self.ddf[(np.abs(self.ddf['asymQ'])<4) & (np.abs(self.ddf['lnQ']))<4]

    def plot_waveforms(self, w0, w1, fit_func, note, df) -> None:
        f, ax = plt.subplots(2,1, figsize=(10, 6), gridspec_kw={'height_ratios':[1,4]})
        sns.lineplot(x=self.t, y=w0, color='green', label='CH0', **self.line_plot_style, alpha=0.6, ax=ax[1])
        sns.lineplot(x=self.t, y=w1, color='red', label='CH1', **self.line_plot_style, alpha=0.6, ax=ax[1])
        sns.lineplot(x=self.t, y=fit_func(self.t,*df.iloc[0, range(len(self.df_cols)-len(self.df_cols_sigma), len(self.df_cols))]), color='green', **self.line_plot_style, alpha=0.6, ax=ax[1])
        sns.lineplot(x=self.t, y=fit_func(self.t,*df.iloc[1, range(len(self.df_cols)-len(self.df_cols_sigma), len(self.df_cols))]), color='red', **self.line_plot_style, alpha=0.6, ax=ax[1])
        ax[0].axis('off')
        ax[0].axis('tight')
        ax[0].set_title(note)
        #ax[1].axis('tight')
        table = ax[0].table(cellText=df.round(4).to_numpy(),colLabels=df.columns,loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        #g.annotate(note, size=10, xy={0,np.min(np.append(w0, w1)/2)} )
        #g.axvline(30, ls='--', lw=0.9, c='red')
        #plt.show()
        filename = f'waveform_{df.iloc[0, 0]}'
        f.savefig(self.charts_path.joinpath('waveforms').joinpath(filename).with_suffix(self.chart_ext))
        plt.clf()


    def plot_lin(self, fit_func, y='dt', fit_param='c', folder='asymmetry', lim=(-20, 20)):
        df = self.ddf.groupby('L').mean().reset_index()
        p, q = curve_fit(fit_func, df['L'], df[y])
        c = round(-2/p[0], 3)
        err = round(np.abs(np.sqrt(q[0][0])/p[0]), 3)
        print(f'->\tFound {fit_param}={c} +- {err}')
        sns.scatterplot(data=df.rename(columns=self.lx), x=self.lx['L'], y=self.lx[y], **self.scatter_plot_style)
        sns.lineplot(x=df['L'], y=fit_func(df['L'], *p), **self.fitline_plot_style, color='black')
        filename = f'fit_{y}'
        plt.savefig(self.charts_path.joinpath(folder).joinpath(filename).with_suffix(self.chart_ext))
        plt.clf()
        sns.histplot(data=self.ddf.rename(columns=self.lx), x=self.lx['L'], y=self.lx[y], cbar=True)
        sns.lineplot(x=df['L'], y=fit_func(df['L'], *p), **self.fitline_plot_style, color='black')
        plt.ylim(lim[0], lim[1])
        filename = f'fit_{y}_hist'
        plt.savefig(self.charts_path.joinpath(folder).joinpath(filename).with_suffix(self.chart_ext))
        plt.clf()

    def plot_asym(self, y, color, folder='asymmetry'):
        if len(y)>1:
            for i, yy in enumerate(y):
                sns.lineplot(data=self.ddf.rename(columns=self.lx), x=self.lx['L'], y=self.lx[yy], **self.line_plot_style, color=color[i])
        else:
            sns.lineplot(data=self.ddf.rename(columns=self.lx), x=self.lx['L'], y=self.lx[y[0]], **self.line_plot_style, color=color[0])
        filename = f'asym_{y}'
        plt.savefig(self.charts_path.joinpath(folder).joinpath(filename).with_suffix(self.chart_ext))
        plt.clf()

    def plot_signal(self, fit_func):
        self.plot_lin(fit_func, 'lnQ', 'Qlambda', lim=(-4, 4))
        self.plot_lin(fit_func, 'lnA', 'Alambda', lim=(-4, 4))

        y = [['sqrtQ'], ['asymQ'], ['sqrtA'], ['asymA'], ['Q_ch0', 'Q_ch1'], ['A_ch0', 'A_ch1']]
        colors = ['black', 'red']
        for yy in y:
            self.plot_asym(yy, colors)

    def plot_joint(self):
        xy = [('t_0_ch0', 't_0_ch1'),
              ('dt', 'lnQ')]
        sns.jointplot(data=self.ddf.rename(columns=self.lx), x=self.lx['t_r'], y=self.lx['t_f'],kind='hist')
        filename = f'joint_t_r-t_f'
        plt.savefig(self.charts_path.joinpath('time').joinpath(filename).with_suffix(self.chart_ext))
        plt.clf()
        for plot in xy:
            sns.jointplot(data=self.ddf.rename(columns=self.lx), x=self.lx[plot[0]], y=self.lx[plot[1]],kind='hist')
            filename = f'joint_{plot[0]}-{plot[1]}'
            plt.savefig(self.charts_path.joinpath('time').joinpath(filename).with_suffix(self.chart_ext))
            plt.clf()

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
    run = Analysis(argv[1], charts_path=argv[2], limit_file=argv[3])
    dff = run.load_waveforms(run.decode_filename, sig_fit)
    run.prepare(dff)
    run.plot_lin(lin_fit)
    run.plot_signal(lin_fit)
    run.plot_joint()

