"""Article-ready sensitivity figures for the recuperative CO2 cycle."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "sensitivity_CO2_recuperative_cycle.xlsx"
OUT = ROOT / "figures_sensitivity"
FONT = {"family": "Arial", "label": 12, "tick": 10, "legend": 9}
FIG, COLORS = {"single": (5.2, 4.8), "dpi": 600}, {"reference": "#1A1A1A", "turbine": "#B2182B", "compressor": "#2166AC"}
REF = {"eta_T_%": 90., "eta_C_%": 80., "dTmin_target_K": 5., "DP_rec_hot_%": 3., "DP_rec_cold_%": 2.}
N_COLOR_INTERVALS = 10

def fmt(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")

def config() -> None:
    mpl.rcParams.update({"font.family": FONT["family"], "font.size": FONT["tick"], "axes.labelsize": FONT["label"], "xtick.labelsize": FONT["tick"], "ytick.labelsize": FONT["tick"], "legend.fontsize": FONT["legend"], "mathtext.fontset": "stix", "axes.unicode_minus": False, "axes.linewidth": .9, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True, "svg.fonttype": "none"})

def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"): fig.savefig(OUT / f"{name}.{ext}", dpi=FIG["dpi"], facecolor="white")
    plt.close(fig)

def style(ax: plt.Axes) -> None:
    ax.tick_params(which="major", length=4, width=.8, pad=3); ax.tick_params(which="minor", length=2.1, width=.65)
    ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator()); ax.grid(alpha=.18, linewidth=.6)

def physical(df: pd.DataFrame) -> pd.DataFrame:
    grids = {column: np.sort(df[column].dropna().unique()) for column in df.columns if pd.api.types.is_numeric_dtype(df[column])}
    result = df.loc[df["physical_root"].astype(str).str.lower().eq("true")].replace([np.inf, -np.inf], np.nan).copy()
    result.attrs["grids"] = grids
    return result

def reference(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    mask = np.ones(len(df), bool)
    for c in cols: mask &= np.isclose(df[c].astype(float), REF[c])
    hit = df.loc[mask]
    if hit.empty: raise RuntimeError(f"Baseline is absent for {cols}.")
    return hit.iloc[0]

def box_above_legend(fig: plt.Figure, ax: plt.Axes, legend: mpl.legend.Legend, text: str,
                     size: float = 8.2, left_aligned: bool = False) -> None:
    fig.canvas.draw(); b = legend.get_window_extent(fig.canvas.get_renderer())
    x_px = b.x0 if left_aligned else b.x1
    x, y = ax.transAxes.inverted().transform((x_px, b.y1 + 2 * fig.dpi / 72))
    ax.text(x, y, text, transform=ax.transAxes, ha="left" if left_aligned else "right",
            va="bottom", multialignment="left", fontsize=size,
            bbox={"boxstyle":"round,pad=0.25", "fc":"white", "ec":"0.45", "alpha":.94})

def extrema(df: pd.DataFrame, x: str, y: str, xm: str, ym: str, xu: str, yu: str, ref: pd.Series) -> str:
    mx, mn = df.loc[df["eta_el_net_plant_%"].idxmax()], df.loc[df["eta_el_net_plant_%"].idxmin()]
    def p(row: pd.Series) -> str: return rf"${xm}={fmt(row[x])}\,{xu}$; ${ym}={fmt(row[y])}\,{yu}$"
    return (rf"Design point: $\eta_{{\rm el,net}}={ref['eta_el_net_plant_%']:.2f}\,\%$"+"\n"+p(ref)+"\n"+rf"$\max(\eta_{{\rm el,net}})={mx['eta_el_net_plant_%']:.2f}\,\%$"+"\n"+p(mx)+"\n"+rf"$\min(\eta_{{\rm el,net}})={mn['eta_el_net_plant_%']:.2f}\,\%$"+"\n"+p(mn))

def extrema_1d(df: pd.DataFrame, x: str, xm: str, unit: str, ref: pd.Series) -> str:
    mx, mn = df.loc[df["eta_el_net_plant_%"].idxmax()], df.loc[df["eta_el_net_plant_%"].idxmin()]
    return (rf"Design point: $\eta_{{\rm el,net}}={ref['eta_el_net_plant_%']:.2f}\,\%$; ${xm}={fmt(ref[x])}\,{unit}$" + "\n"
            + rf"$\max(\eta_{{\rm el,net}})={mx['eta_el_net_plant_%']:.2f}\,\%$ at ${xm}={fmt(mx[x])}\,{unit}$" + "\n"
            + rf"$\min(\eta_{{\rm el,net}})={mn['eta_el_net_plant_%']:.2f}\,\%$ at ${xm}={fmt(mn[x])}\,{unit}$")

def contour(df: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, xm: str, ym: str, xu: str, yu: str, ref: pd.Series, name: str, show_no_solution: bool = True) -> None:
    zc="eta_el_net_plant_%"; grids=df.attrs["grids"]
    pivot=df.pivot_table(index=y, columns=x, values=zc, aggfunc="mean").reindex(index=grids[y],columns=grids[x]); xx,yy=pivot.columns.to_numpy(float),pivot.index.to_numpy(float); zz=np.ma.masked_invalid(pivot.to_numpy(float))
    levels=np.linspace(float(zz.min()),float(zz.max()),N_COLOR_INTERVALS+1)
    fig,ax=plt.subplots(figsize=FIG["single"],layout="constrained"); cf=ax.contourf(xx,yy,zz,levels=levels,cmap="RdBu",extend="neither"); cs=ax.contour(xx,yy,zz,levels=levels,colors="black",linewidths=.7,alpha=.8); ax.clabel(cs,inline=True,fontsize=8,fmt=fmt)
    if show_no_solution:
        missing=pivot.isna().to_numpy(float); ax.contourf(xx,yy,missing,levels=[.5,1.5],colors="none",hatches=["///"],zorder=2)
    if show_no_solution:
        xlim, ylim = (xx.min(), xx.max()), (yy.min(), yy.max())
    else:
        finite = ~np.ma.getmaskarray(zz)
        xlim = (xx[np.any(finite, axis=0)].min(), xx[np.any(finite, axis=0)].max())
        ylim = (yy[np.any(finite, axis=1)].min(), yy[np.any(finite, axis=1)].max())
    ax.plot(ref[x],ref[y],"*",ms=13,color=COLORS["reference"],markeredgecolor="white",markeredgewidth=.8,label="Design point",zorder=10); ax.set(xlabel=xlabel,ylabel=ylabel,xlim=xlim,ylim=ylim); style(ax)
    design_handle=plt.Line2D([],[],marker="*",markersize=13,color=COLORS["reference"],markeredgecolor="white",markeredgewidth=.8,linestyle="None",label="Design point")
    handles=[design_handle]
    if show_no_solution: handles.append(Patch(facecolor="white",edgecolor="black",hatch="///",label="No physical solution"))
    legend=ax.legend(handles=handles,loc="lower right",frameon=True,edgecolor=".65",fontsize=FONT["legend"])
    box_above_legend(fig,ax,legend,extrema(df,x,y,xm,ym,xu,yu,ref),size=8.2); cb=fig.colorbar(cf,ax=ax,pad=.035,aspect=22); cb.set_label(r"Net electrical efficiency, %"); cb.set_ticks(levels); cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:fmt(v))); cb.ax.tick_params(labelsize=FONT["tick"],length=3); save(fig,name)

def eta_lines(df: pd.DataFrame, ref: pd.Series) -> None:
    fig,ax=plt.subplots(figsize=FIG["single"],layout="constrained"); a=df[np.isclose(df["eta_C_%"],80)].sort_values("eta_T_%"); b=df[np.isclose(df["eta_T_%"],90)].sort_values("eta_C_%")
    ax.plot(a["eta_T_%"],a["eta_el_net_plant_%"],"o-",color=COLORS["turbine"],label=r"$\eta_t$ at $\eta_c=80\,\%$"); ax.plot(b["eta_C_%"],b["eta_el_net_plant_%"],"s-",color=COLORS["compressor"],label=r"$\eta_c$ at $\eta_t=90\,\%$"); ax.plot(90,ref["eta_el_net_plant_%"],"*",ms=13,color=COLORS["reference"],markeredgecolor="white",label="Design point",zorder=10); ax.set(xlabel="Isentropic efficiency, %",ylabel=r"Net electrical efficiency, $\eta_{\rm el,net}$, %"); style(ax); legend=ax.legend(loc="lower right",fontsize=8.2,frameon=True,edgecolor=".65"); box_above_legend(fig,ax,legend,extrema(df,"eta_T_%","eta_C_%",r"\eta_t",r"\eta_c",r"\%",r"\%",ref)); save(fig,"02_efficiency_sensitivity_lines")

def dT_lines(df: pd.DataFrame, ref: pd.Series) -> None:
    d=df.loc[np.isclose(df["dTmin_target_K"], np.round(df["dTmin_target_K"]))].sort_values("dTmin_target_K"); fig,ax=plt.subplots(figsize=FIG["single"],layout="constrained"); ax.plot(d["dTmin_target_K"],d["eta_el_net_plant_%"],"o-",color=COLORS["turbine"],label=r"$\Delta T_{\rm min}$ variation"); ax.plot(5,ref["eta_el_net_plant_%"],"*",ms=13,color=COLORS["reference"],markeredgecolor="white",label="Design point",zorder=10); ax.set(xlabel="Minimum temperature difference, K",ylabel=r"Net electrical efficiency, $\eta_{\rm el,net}$, %"); style(ax); ax.xaxis.set_major_locator(MultipleLocator(1)); legend=ax.legend(loc="lower left",fontsize=8.2,frameon=True,edgecolor=".65"); box_above_legend(fig,ax,legend,extrema_1d(d,"dTmin_target_K",r"\Delta T_{\rm min}","K",ref),left_aligned=True); save(fig,"04_minimum_temperature_difference")

def dp_lines(df: pd.DataFrame, ref: pd.Series) -> None:
    fig,ax=plt.subplots(figsize=FIG["single"],layout="constrained"); a=df[np.isclose(df["DP_rec_cold_%"],2)].sort_values("DP_rec_hot_%"); b=df[np.isclose(df["DP_rec_hot_%"],3)].sort_values("DP_rec_cold_%"); ax.plot(a["DP_rec_hot_%"],a["eta_el_net_plant_%"],"o-",color=COLORS["turbine"],label=r"$\Delta p_{hot}$ at $\Delta p_{cold}=2\,\%$"); ax.plot(b["DP_rec_cold_%"],b["eta_el_net_plant_%"],"s-",color=COLORS["compressor"],label=r"$\Delta p_{cold}$ at $\Delta p_{hot}=3\,\%$"); ax.plot(3,ref["eta_el_net_plant_%"],"*",ms=13,color=COLORS["reference"],markeredgecolor="white",label="Design point",zorder=10); ax.set(xlabel="Relative pressure loss, %",ylabel=r"Net electrical efficiency, $\eta_{\rm el,net}$, %"); style(ax); legend=ax.legend(loc="lower left",fontsize=8.2,frameon=True,edgecolor=".65"); box_above_legend(fig,ax,legend,extrema(df,"DP_rec_hot_%","DP_rec_cold_%",r"\Delta p_{hot}",r"\Delta p_{cold}",r"\%",r"\%",ref),left_aligned=True); save(fig,"06_pressure_loss_lines")

def main() -> None:
    config(); eta=physical(pd.read_excel(DATA,sheet_name="Results_eta")); dp=physical(pd.read_excel(DATA,sheet_name="Results_pressure")); dt=physical(pd.read_excel(DATA,sheet_name="Results_dT")); er=reference(eta,("eta_T_%","eta_C_%")); dr=reference(dp,("DP_rec_hot_%","DP_rec_cold_%")); tr=reference(dt,("dTmin_target_K",))
    contour(eta,"eta_T_%","eta_C_%",r"$\eta_t$, %",r"$\eta_c$, %",r"\eta_t",r"\eta_c",r"\%",r"\%",er,"01_efficiency_contour"); eta_lines(eta,er)
    contour(dt,"dTmin_target_K","dTmin_target_K","Minimum temperature difference, K","Minimum temperature difference, K",r"\Delta T_{\rm min}",r"\Delta T_{\rm min}","K","K",tr,"03_temperature_contour") if False else dT_lines(dt,tr)
    contour(dp,"DP_rec_hot_%","DP_rec_cold_%",r"$\Delta p_{hot}$, %",r"$\Delta p_{cold}$, %",r"\Delta p_{hot}",r"\Delta p_{cold}",r"\%",r"\%",dr,"05_pressure_loss_contour",show_no_solution=False); dp_lines(dp,dr)
    print(f"Created figures in: {OUT}")
if __name__ == "__main__": main()
