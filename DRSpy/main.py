###############################################################
###############################################################
import os
import click
import DRSpy
from DRSpy.data_struct import *

from scipy.optimize import curve_fit
import numpy as np

###############################################################

def print_version(ctx, param, value):
    """ Print DRSpy Verison """
    if not value or ctx.resilient_parsing:
        return
    print(f"DRSpy v{DRSpy.__version__}")
    ctx.exit()
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

###############################################################

@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.option('-d','--db', 'fdb', help='Path to spectra database', type=str, default="drs_spectra.csv", show_default=True)
@click.option('-x','--dbxml', 'fdbxml', help='Path to waveforms database', type=str, default="drs_waveforms.csv", show_default=True)
@click.option('-m','--meta', 'fmeta', help='Path to metadata file', type=str, default="drs_info.ini", show_default=True)
@click.option('-v', '--verbose', 'fverbose', help='Enable verbosity mode', is_flag=True, default=False)
@click.option('--version', help="Show DRSpy version", is_flag=True, callback=print_version, expose_value=False, is_eager=True)
@click.pass_context
def main(ctx, fdb, fdbxml, fmeta, fverbose):
    """
        Data analysis tool for DRS4(PSI) board.
    """
    ctx.obj = DataStruct(fverbose=fverbose, data_file=fdb)
    """
    print(f"\n{ctx.get_help()}\n")
    ctx.info_name = "run"
    print(main.get_command(ctx, "run").get_help(ctx))
    ctx.info_name = "gencfg"
    print(main.get_command(ctx, "gencfg").get_help(ctx))
    """
@main.command(short_help="Load new files")
@click.option('-a','--auto', 'fauto', help='Recognize file automatically in specified folder', is_flag=True, default=False)
@click.option('-f','--format', 'fformat', help='Input file format', type=click.Choice(["xml","PtP", "delay"]), default="PtP", show_default=True)
@click.option('-t','--tag', 'ftag', help='Add <tag> to data headers', default="", type=str)
@click.argument("files", nargs=-1, metavar="<files or dir>")
@click.pass_context
def update(ctx, files, fformat, fauto, ftag):
    """
        Load new data files.
    """
    if fauto:
        ctx.obj.auto_decode(files[0], ftag)
    else:
        for file in files:
            ctx.obj.load_file(file, fformat, ftag)

@main.command(short_help="DataBase description")
@click.pass_context
def describe(ctx):
    """ 
        Print information saved in metadata file.
    """
    log(ctx.obj.data.describe(), "green")

@main.command(short_help="Command line")
@click.option('--exec', 'fexec', help='Execute without print()', is_flag=True, default=False)
@click.argument("command", nargs=1, default="""print("DRSpy CLI")""")
@click.pass_context
def cli(ctx, command, fexec):
    if fexec: eval(command)
    else: print(eval(command))
    while True:
        log("> ", wait=True)
        command = input()
        try: print(eval(command))
        except Exception as e: log(e, "red")

@main.command(short_help="Generate graphs")
@click.option("-e", "--ext", "fext", help="Picture extension", default="png", type=click.Choice(["png", "pdf"]), show_default=True)
@click.option("-k", "--kind", "fkind", help="Plot type", default="line", type=str, show_default=True)
@click.option('--live', 'flive', help='Enable live preview', is_flag=True, default=False)
@click.argument("expression", nargs=2, metavar="<expression>")
@click.pass_context
def plot(ctx, expression, fext, flive, fkind):
    ctx.obj.plot(*expression, flive=flive, ext=fext, fkind=fkind)

@main.command(short_help="Load macro")
@click.argument("macro", metavar="<macro>")
@click.pass_context
def macro(ctx, macro):
    db = ctx.obj
    dataset = ctx.obj.data
    with open(macro, "r") as file:
        code = file.read()
        exec(code)

@main.command()
@click.argument('subcommand')
@click.pass_context
def help(ctx, subcommand):
    ctx.info_name = subcommand
    subcommand_obj = main.get_command(ctx, subcommand)
    if subcommand_obj is None:
        click.echo("I don't know that command.")
    else:
        click.echo(subcommand_obj.get_help(ctx))

@main.command(short_help="run")
@click.pass_context
def run(ctx):
    dataset = ctx.obj.data
    def nconv(n, spl="_"):
        ss = float(n.split(spl)[0])
        return ss

    def t_sum(df, v, w):
        #ww = df[w]
        #vv = df[v]
        ddf = df[[w,v]].dropna().to_numpy().T
        tw = (ddf[0]*ddf[1]).sum()/ddf[1].sum()
        print(f"xxx -> {v} {tw}")
        return tw
        
        #rr = float((ww*vv).sum()/ww.sum())
        print(f" ----> Tsum  {v}   {w}     {rr} ")
        return rr

    def asym(df, c1, c2):
        return  (df[c1] - df[c2])/(df[c1] + df[c2])

    def rr(dxr, dyr):
        return ctx.obj.plot(dxr,dyr, regx=True)   

    def f_l(X,a,b):
        return a*X+b

    dl = ".*Delay.*"
    chn = ".*Channel"

    dall = ".*_t"
    ccent = ".*C_t"
    ucent = ".*U_t"
    dcent = ".*D_t"


    dall_w = rr(dcent, dcent)
    uall_w = rr(ucent, dcent)
    call_w = rr(ccent, dcent)
    dall_w_l = []
    uall_w_l = []
    call_w_l = []
    
    dw_C = [[],[]]
    dw_U = [[],[]]
    dw_D = [[],[]]
    for dw in rr(ccent, ccent)[0]:  
        dw_C[0].append(float(nconv(dw)))
        dw_C[1].append(t_sum(ctx.obj.data, dw, "Delay [ns]"))
    for dw in rr(dcent, dcent)[0]:
        dw_D[0].append(float(nconv(dw)))
        dw_D[1].append(t_sum(ctx.obj.data, dw, "Delay [ns]"))
    for dw in rr(ucent, dcent)[0]:
        dw_U[0].append(float(nconv(dw)))
        dw_U[1].append(t_sum(ctx.obj.data, dw, "Delay [ns]"))
    dw_C = np.array(dw_C,dtype=float)
    dw_D = np.array(dw_D, dtype=float)
    dw_U = np.array(dw_U,dtype=float)
    plt.figure(figsize=(12,7))
    dw_X =  np.linspace(-40,40,100)
    dw_ppc, _ = curve_fit(f_l, dw_C[0], dw_C[1])
    dw_ppu, _ = curve_fit(f_l, dw_U[0], dw_U[1])
    dw_ppd, _ = curve_fit(f_l, dw_D[0], dw_D[1])
    plt.plot(dw_C[0], dw_C[1],"go", label="Center")
    plt.plot(dw_X, f_l(dw_X,*dw_ppc),"g--", alpha=0.5)
    plt.plot(dw_D[0], dw_D[1], "ro", label="Edge D")
    #plt.plot(dw_X, f_l(dw_X,*dw_ppd),"r--", alpha=0.5)
    plt.plot(dw_U[0], dw_U[1], "bo", label="Edge U")
    #plt.plot(dw_X, f_l(dw_X,*dw_ppu),"b--", alpha=0.5)
    plt.legend()
    plt.grid(True)
    plt.xlabel("Distance [cm]")
    plt.ylabel("Delay [ns] ")
    #plt.show()
    plt.savefig("delay_avg.png")
    plt.clf()


    """
    time_C = ".*\-[1-4][0-9]_C.*_t"
    time_D = ".*\-[1-4][0-9]_D.*_t"
    time_U = ".*\-[1-4][0-9]_U.*_t"
    t_C = [[],[]]
    t_U = [[],[]]
    t_D = [[],[]]

    print("PASS4")
    fig, (ax1,ax2,ax3) = plt.subplots(3, figsize=(12,5))
    for i in  rr(time_C,time_C):
        ctx.obj.data.plot.line("Delay [ns]", i, ax=ax1, c="blue", label=False, alpha=0.4)
    ax1.get_legend().remove()
    ax1.set_xlim(-8,2)
    ax1.set_xlabel("")
    ax1.set_title("Center")
    print("PASS5")
    for i in  rr(time_U,time_C):
        ctx.obj.data.plot.line("Delay [ns]", i, ax=ax2, c="red", label=False, alpha=0.4)
    print("PASS55")
    ax2.get_legend().remove()
    ax2.set_xlim(-8,2)
    ax2.set_title("Edge U")
    ax2.set_xlabel("")
    print("PASS6")
    for i in  rr(time_D,time_C):
        ctx.obj.data.plot.line("Delay [ns]", i, ax=ax3, c="green", label=False, alpha=0.4)
    ax3.get_legend().remove()
    ax3.set_xlim(-8,2)
    ax3.set_title("Edge D")
    ax3.set_xlabel("Delay [ns]")
    
    print("PASS7")
    #plt.show()
    plt.savefig("delay_spectrum.png")
    plt.clf()
    """
    c_call = ".*-CH[0-1]"
    c0_call = ".*C-CH0"
    c1_call = ".*C-CH1"
    c0_uall = ".*U-CH0"
    c1_uall = ".*U-CH1"
    c0_dall = ".*D-CH0"
    c1_dall = ".*D-CH1"
    
    cx0 = rr(".*C.*CH0", c_call)[0]
    cx1 = rr(".*C.*CH1", c_call)[0]
    plt.clf()

    def landau(X,E,S, N):
        return 1/np.sqrt(2*np.pi) * np.exp(-((((X-E)/S)+np.exp(-((X-E)/S)))/2)) *N 
    cD = []
    uD = []
    dD = []
    cH0e = []
    cH0 = []
    cH1e = []
    cH1 = []

    uH0 = []
    uH1 = []
    uH0e =[]
    uH1e = []

    dH0 = []
    dH1 = []
    dH0e =[]
    dH1e = []

    for i,j  in enumerate(cx0): 
        fl = nconv(j)
        fig, ax = plt.subplots(1, figsize=(12,7))
        cx_max = np.linspace(10, 160,300)
        cx_x = ctx.obj.data[["Channel", cx0[i], cx1[i]]].dropna().to_numpy().T
        tm = [[],[],[]]
        for e in range(cx_x.shape[1]):
            if cx_x[1][e] > 20 or cx_x[2][e] > 20:
                tm[0].append(cx_x[0][e])
                tm[1].append(cx_x[1][e])
                tm[2].append(cx_x[2][e])
        cx_x = np.array(tm)
        cx_pp0, _ = curve_fit(landau, cx_x[0][::1][:-15],cx_x[1][::1][:-15], maxfev=6500)
        cx_pp1, _ = curve_fit(landau, cx_x[0][::1][:-15],cx_x[2][::1][:-15], maxfev=6500)
        cD.append(nconv(j))
        cH0.append(cx_pp0[0])
        cH1.append(cx_pp1[0])
        cH0e.append(cx_pp0[1])
        cH1e.append(cx_pp1[1])
        ctx.obj.data.plot.line("Channel", cx0[i], ax=ax,alpha=0.7, c="red", markersize=1)
        ctx.obj.data.plot.line("Channel", cx1[i], ax=ax,alpha=0.7,c="green", markersize=1)
        ax.plot(cx_max, landau(cx_max, *cx_pp0), "r--",alpha=0.9, markersize=1)
        ax.plot(cx_max, landau(cx_max, *cx_pp1), "g--",alpha=0.9, markersize=1)


        if j.replace("_C", "_U") in ctx.obj.data.columns:
            ux_x = ctx.obj.data[["Channel", cx0[i].replace("_C","_U"), cx1[i].replace("_C","_U")]].dropna().to_numpy().T
            um = [[],[],[]]
            for e in range(ux_x.shape[1]):
                if ux_x[1][e] > 20 or ux_x[2][e] > 20:
                    um[0].append(ux_x[0][e])
                    um[1].append(ux_x[1][e])
                    um[2].append(ux_x[2][e])
            ux_x = np.array(ux_x)
            ux_pp0, _ = curve_fit(landau, ux_x[0][::1][:-15],ux_x[1][::1][:-15], maxfev=6500)
            ux_pp1, _ = curve_fit(landau, ux_x[0][::1][:-15],ux_x[2][::1][:-15], maxfev=6500)
            ctx.obj.data.plot.line("Channel", cx0[i].replace("_C", "_U"), ax=ax, c="yellow", markersize=1, alpha=0.7)
            ctx.obj.data.plot.line("Channel", cx1[i].replace("_C", "_U"), ax=ax,c="black", markersize=1, alpha=0.7)
            ax.plot(cx_max, landau(cx_max, *ux_pp0), "r--",alpha=0.9, markersize=1)
            ax.plot(cx_max, landau(cx_max, *ux_pp1), "g--",alpha=0.9, markersize=1)
            uD.append(nconv(j))
            uH0.append(ux_pp0[0])
            uH1.append(ux_pp1[0])
            uH0e.append(ux_pp0[1])
            uH1e.append(ux_pp1[1])

        if j.replace("_C", "_D") in ctx.obj.data.columns:
            dx_x = ctx.obj.data[["Channel", cx0[i].replace("_C", "_D"), cx1[i].replace("_C", "_D")]].dropna().to_numpy().T
            dm = [[],[],[]]
            for e in range(dx_x.shape[1]):
                if dx_x[1][e] > 20 or dx_x[2][e] > 20:
                    dm[0].append(dx_x[0][e])
                    dm[1].append(dx_x[1][e])
                    dm[2].append(dx_x[2][e])
            if dx_x.shape != (3,0):
                dx_x = np.array(dx_x)
                dx_pp0, _ = curve_fit(landau, dx_x[0][::1][:-15],dx_x[1][::1][:-15], maxfev=6500)
                dx_pp1, _ = curve_fit(landau, dx_x[0][::1][:-15],dx_x[2][::1][:-15], maxfev=6500)
                ctx.obj.data.plot.line("Channel", cx0[i].replace("_C", "_D"), ax=ax, c="orange", markersize=1, alpha=0.7)
                ctx.obj.data.plot.line("Channel", cx1[i].replace("_C", "_D"), ax=ax,c="blue", markersize=1,alpha=0.7)
                ax.plot(cx_max, landau(cx_max, *dx_pp0), "r--",alpha=0.9, markersize=1)
                ax.plot(cx_max, landau(cx_max, *dx_pp1), "g--",alpha=0.9, markersize=1)
                dD.append(nconv(j))
                dH0.append(dx_pp0[0])
                dH1.append(dx_pp1[0])
                dH0e.append(dx_pp0[1])
                dH1e.append(dx_pp1[1])

        ax.set_xlim(-10,160)
        ax.set_title(f"")
        plt.grid(True)
        plt.ylabel("Counts")
        
        #plt.show()
        plt.savefig(f"CH_{fl}.png")
        ax.cla()
    
    
    def fxx(X,A, B,):
        return A + B*np.log(X)
        #return A*np.exp(B*X)
    cD = np.array(cD,dtype=float)
    cH0 = np.array(cH0,dtype=float)
    cH1 = np.array(cH1,dtype=float)
    wx = np.linspace(-40, 40,300)
    pp0, _ = curve_fit(fxx, cD, cH0)
    pp1, _ = curve_fit(fxx, cD, cH1)
    p0 = np.polyfit(cD, cH0, 3)
    p1 = np.polyfit(cD, cH1, 3, w=np.sqrt(cH1))
    pp0 = np.poly1d(p0)
    pp1 = np.poly1d(p1)
    plt.plot(wx, pp0(wx), "r--", alpha=0.5)
    plt.plot(wx, pp1(wx), "g--", alpha=0.5)
    #plt.plot(wx, fxx(wx, *pp0), "r--", alpha=0.5)
    #plt.plot(wx, fxx(wx, *pp1), "g--", alpha=0.5)
    uD = np.array(uD)
    dD = np.array(dD)
    plt.errorbar(cD, cH0,cH0e, 0.001, "ro",label="CH0", elinewidth=1, capsize=1, markersize=6, alpha=0.9)
    plt.errorbar(cD, cH1,cH1e,0.001, "go", label="CH1", elinewidth=1, capsize=1,markersize=6,alpha=0.9)
    #plt.errorbar(uD, uH0,uH0e, 0.001, c="yellow", marker="v",label="CH0 Up", elinewidth=1, capsize=1, markersize=6, alpha=0.5, linewidth=0)
    #plt.errorbar(uD, uH1,uH1e,0.001, c="black",marker="v", label="CH1 Up", elinewidth=1, capsize=1,markersize=6, alpha=0.5, linewidth=0)
    #plt.errorbar(dD, dH0,dH0e, 0.001, c="orange", marker="^",label="CH0 Down", elinewidth=1, capsize=1, markersize=6,alpha=0.5, linewidth=0)
    #plt.errorbar(dD, dH1,dH1e,0.001, c="blue", marker="^", label="CH1 Down", elinewidth=1, capsize=1,markersize=6,alpha=0.5, linewidth=0)
    plt.grid(True)
    plt.legend()
    plt.xlabel("Distance [cm]")
    plt.ylabel("Channel [mV]")
    plt.title("Landau Peaks")
    plt.autoscale()
    plt.xlim(-41,41)
    plt.show()
    #plt.savefig("landau.png")
    plt.cla()
    fl, (af1,af2,af3) = plt.subplots(3, figsize=(12,7))
    af1.errorbar(cD, cH0,cH0e, 0.001, "ro",label="CH0", elinewidth=1, capsize=1, markersize=6, alpha=0.9)
    af1.errorbar(cD, cH1,cH1e,0.001, "go", label="CH1", elinewidth=1, capsize=1,markersize=6,alpha=0.9)
    af1.set_xlim(-41,41)
    af2.errorbar(uD, uH0,uH0e, 0.001, c="yellow", marker="v",label="CH0 Up", elinewidth=1, capsize=1, markersize=6, alpha=0.5, linewidth=0)
    af2.errorbar(uD, uH1,uH1e,0.001, c="black",marker="v", label="CH1 Up", elinewidth=1, capsize=1,markersize=6, alpha=0.5, linewidth=0)
    af2.set_xlim(-41,41)
    af3.errorbar(dD, dH0,dH0e, 0.001, c="orange", marker="^",label="CH0 Down", elinewidth=1, capsize=1, markersize=6,alpha=0.5, linewidth=0)
    af3.errorbar(dD, dH1,dH1e,0.001, c="blue", marker="^", label="CH1 Down", elinewidth=1, capsize=1,markersize=6,alpha=0.5, linewidth=0)
    af3.set_xlim(-41,41)
    plt.grid(True)
    plt.legend()
    plt.xlabel("Distance [cm]")
    plt.ylabel("Channel [mV]")
    plt.title("Landau Peaks")
    plt.savefig("landau3.png")
    plt.cla()
    #plt.show()

    def moyal(X, E, S, N):
        #xx = (X-E)/S
        #mm = -0.5 * (xx + np.exp(-xx))
        #return 1/np.sqrt(2)/S * np.exp(mm)
        #xx = (X - P)/S
        #fm = xx/S
        #threexp = np.exp(-0.5*(fm+np.exp(-fm)))
        #return threexp/np.sqrt(2*np.pi)
        return 1/np.sqrt(2*np.pi) * np.exp(-0.5*(((X-E)/S)+np.exp(-((X-E)/S)))) / S * N
        

    xc0, xc00 = rr(c0_call, c0_call)
    xc0x = ctx.obj.data[["Channel", xc0[0]]].dropna().to_numpy().T
    xc0t = np.linspace(xc0x[0].min(), xc0x[0].max(), 300)
    xc0pp, xc0pc = curve_fit(landau, xc0x[0], xc0x[1] )
    f, ax = plt.subplots(figsize=(12,7))
    ctx.obj.data.plot("Channel", xc0[0],ax=ax,xerr=1)
    plt.plot(xc0t, landau(xc0t, *xc0pp))
    plt.autoscale()
    #plt.show()
        
    

if __name__ == '__main__':
    print(f"Started: DRSpy v{DRSpy.__version__}")

