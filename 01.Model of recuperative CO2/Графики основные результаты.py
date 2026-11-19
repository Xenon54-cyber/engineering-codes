"""Article-ready main-result contour maps for the recuperative CO2 cycle."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter

ROOT=Path(__file__).resolve().parent; DATA=ROOT/"recuperative_CO2_DTmin5K_recuperator_losses.xlsx"; OUT=ROOT/"figures_main"
FONT={"family":"Arial","label":12,"tick":10,"legend":9}
FIG={"single":(5.2,4.8),"dpi":600}; REF={"P1_MPa":35.,"P2_MPa":15.}; COLORS={"reference":"#1A1A1A"}
N_COLOR_INTERVALS=10
def fmt(v: float)->str: return f"{float(v):.2f}".rstrip("0").rstrip(".")
def config()->None:
    mpl.rcParams.update({"font.family":FONT["family"],"font.size":FONT["tick"],"axes.labelsize":FONT["label"],"xtick.labelsize":FONT["tick"],"ytick.labelsize":FONT["tick"],"legend.fontsize":FONT["legend"],"mathtext.fontset":"stix","axes.unicode_minus":False,"axes.linewidth":.9,"xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,"xtick.minor.visible":True,"ytick.minor.visible":True,"svg.fonttype":"none"})
def save(fig:plt.Figure,name:str)->None:
    OUT.mkdir(exist_ok=True)
    for ext in ("png","svg"): fig.savefig(OUT/f"{name}.{ext}",dpi=FIG["dpi"],facecolor="white")
    plt.close(fig)
def style(ax:plt.Axes)->None:
    ax.tick_params(which="major",length=4,width=.8,pad=3); ax.tick_params(which="minor",length=2.1,width=.65); ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator()); ax.grid(alpha=.18,linewidth=.6)
def box(ax:plt.Axes,text:str)->None:
    ax.text(.98,.03,text,transform=ax.transAxes,ha="right",va="bottom",multialignment="left",fontsize=8.2,bbox={"boxstyle":"round,pad=0.25","fc":"white","ec":"0.45","alpha":.94})
def load()->tuple[pd.DataFrame,pd.Series,dict[str,np.ndarray]]:
    need=["P1_MPa","P2_MPa","physical_root","eta_el_net_plant_%","N_TURB_i_MW","N_MCOMP_i_MW","N_RCOMP_i_MW"]
    d=pd.read_excel(DATA,sheet_name="Results"); missing=set(need)-set(d.columns)
    if missing: raise KeyError(f"Missing columns: {sorted(missing)}")
    grid={"P1_MPa":np.sort(d["P1_MPa"].dropna().unique()),"P2_MPa":np.sort(d["P2_MPa"].dropna().unique())}
    d=d.loc[d.physical_root.astype(str).str.lower().eq("true")].copy(); d["N_COMP_TOTAL_MW"]=d["N_MCOMP_i_MW"]+d["N_RCOMP_i_MW"]; d["N_INTERNAL_MW"]=d["N_TURB_i_MW"]-d["N_RCOMP_i_MW"]-d["N_MCOMP_i_MW"]
    m=np.isclose(d.P1_MPa,REF["P1_MPa"])&np.isclose(d.P2_MPa,REF["P2_MPa"])
    if not m.any(): raise RuntimeError("Design point P1=35 MPa, P2=15 MPa is absent.")
    return d,d.loc[m].iloc[0],grid
def annotation(d:pd.DataFrame,col:str,math:str,unit:str,r:pd.Series)->str:
    mx,mn=d.loc[d[col].idxmax()],d.loc[d[col].idxmin()]
    def pp(q:pd.Series)->str:return rf"$P_1={fmt(q.P1_MPa)}\,{{\rm MPa}}$; $P_2={fmt(q.P2_MPa)}\,{{\rm MPa}}$"
    return rf"Design point: ${math}={r[col]:.2f}\,{unit}$"+"\n"+pp(r)+"\n"+rf"$\max({math})={mx[col]:.2f}\,{unit}$"+"\n"+pp(mx)+"\n"+rf"$\min({math})={mn[col]:.2f}\,{unit}$"+"\n"+pp(mn)
def contour(d:pd.DataFrame,col:str,math:str,unit:str,label:str,name:str,r:pd.Series,grid:dict[str,np.ndarray])->None:
    p=d.pivot_table(index="P1_MPa",columns="P2_MPa",values=col,aggfunc="mean").reindex(index=grid["P1_MPa"],columns=grid["P2_MPa"])
    x,y=p.columns.to_numpy(float),p.index.to_numpy(float); z=np.ma.masked_invalid(p.to_numpy(float)); lev=np.linspace(float(z.min()),float(z.max()),N_COLOR_INTERVALS+1)
    fig,ax=plt.subplots(figsize=FIG["single"],layout="constrained"); cf=ax.contourf(x,y,z,levels=lev,cmap="RdBu",extend="neither"); cs=ax.contour(x,y,z,levels=lev,colors="black",linewidths=.7,alpha=.8); ax.clabel(cs,inline=True,fontsize=8,fmt=fmt)
    # Use the same masked-contour hatching as the recompression maps so that
    # the boundary follows the physical-solution domain rather than grid cells.
    missing=p.isna().to_numpy(dtype=float)
    ax.contourf(x,y,missing,levels=[0.5,1.5],colors="none",hatches=["///"],zorder=2)
    ax.set(xlabel=r"$P_2$, MPa",ylabel=r"$P_1$, MPa"); ax.set(xlim=(x.min(),x.max()),ylim=(y.min(),y.max())); style(ax)
    no_solution=Patch(facecolor="#F2F2F2",edgecolor="black",hatch="///",label="No physical solution"); ax.legend(handles=[no_solution],loc="lower left",frameon=True,edgecolor=".65",fontsize=FONT["legend"])
    box(ax,annotation(d,col,math,unit,r)); cb=fig.colorbar(cf,ax=ax,pad=.035,aspect=22); cb.set_label(label); cb.set_ticks(lev); cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:fmt(v))); cb.ax.tick_params(labelsize=FONT["tick"],length=3); save(fig,name)
def main()->None:
    config();d,r,grid=load(); figs=[("eta_el_net_plant_%",r"\eta_{\rm el,net}",r"\%","Net electrical efficiency, %","01_net_electrical_efficiency"),("eta_el_net_CO2_%",r"\eta_{\rm el,net,CO_2}",r"\%","CO$_2$ cycle net efficiency, %","02_co2_net_efficiency"),("eta_cycle_CO2_%",r"\eta_{\rm cycle,CO_2}",r"\%","CO$_2$ cycle efficiency, %","03_co2_cycle_efficiency"),("eta_plant_%",r"\eta_{\rm plant}",r"\%","Plant efficiency, %","04_plant_efficiency"),("N_TURB_i_MW",r"N_{\rm turb,i}",r"{\rm MW}","Turbine internal power, MW","05_turbine_internal_power"),("N_COMP_TOTAL_MW",r"N_{\rm comp,i}",r"{\rm MW}","Total compressor internal power, MW","06_compressor_internal_power"),("N_INTERNAL_MW",r"N_{\rm net,i}",r"{\rm MW}","Cycle internal power, MW","07_cycle_internal_power"),("N_el_gross_MW",r"N_{\rm el,gross}",r"{\rm MW}","Gross electrical power, MW","08_gross_electrical_power"),("N_el_net_MW",r"N_{\rm el,net}",r"{\rm MW}","Net electrical power, MW","09_net_electrical_power"),("Q_RECUP_MW",r"Q_{\rm recup}",r"{\rm MW}","Recuperator heat duty, MW","10_recuperator_heat_duty"),("Q_COOL_MW",r"Q_{\rm cool}",r"{\rm MW}","Cooler heat duty, MW","11_cooler_heat_duty"),("G_CO2_kg_s",r"G_{\rm CO_2}",r"{\rm kg\,s^{-1}}",r"CO$_2$ mass flow rate, kg s$^{-1}$","12_co2_mass_flow")]
    for f in figs: contour(d,*f,r,grid)
    print(f"Created {len(figs)} figures in: {OUT}")
if __name__=="__main__": main()
