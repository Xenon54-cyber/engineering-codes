"""Article-ready plots for the recompression-cycle sensitivity analysis.

Reads ``Чувствительность Рекомпрессия.xlsx`` from this directory and writes
PNG and SVG versions of every figure to ``figures``.  The visual language
follows plot_results.py: compact Arial figures, inward ticks, restrained grid,
Viridis contour maps, and a red marker for the reference design point.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FuncFormatter


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "Чувствительность Рекомпрессия.xlsx"
OUT_DIR = ROOT / "figures"

FONT = {"family": "Arial", "title": 14, "label": 11, "tick": 9, "legend": 9}
FIG = {"single": (5.2, 4.8), "dpi": 600}
COLORS = {"reference": "#1A1A1A", "turbine": "#B2182B", "compressor": "#2166AC"}
N_COLOR_INTERVALS = 10

REFERENCE = {
    "KPDcomp_%": 80.0,
    "KPDturb_%": 90.0,
    "dT_HTHE_K": 5.0,
    "dT_LTHE_K": 5.0,
    "dP_hot_%": 3.0,
    "dP_cold_%": 2.0,
}


def configure() -> None:
    mpl.rcParams.update({
        "font.family": FONT["family"], "font.size": FONT["tick"],
        "axes.titlesize": FONT["title"], "axes.labelsize": FONT["label"],
        "xtick.labelsize": FONT["tick"], "ytick.labelsize": FONT["tick"],
        "legend.fontsize": FONT["legend"], "mathtext.fontset": "stix",
        "axes.unicode_minus": False, "axes.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(OUT_DIR / f"{name}.{suffix}", dpi=FIG["dpi"], facecolor="white")
    plt.close(fig)


def style_axes(ax: plt.Axes, grid: bool = True) -> None:
    ax.tick_params(which="major", length=4.0, width=0.8, pad=3)
    ax.tick_params(which="minor", length=2.1, width=0.65)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if grid:
        ax.grid(alpha=0.18, linewidth=0.6)


def fmt(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def physical(df: pd.DataFrame) -> pd.DataFrame:
    grids = {
        column: np.sort(df[column].dropna().unique())
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
    }
    out = df.copy()
    for col in ("physical_root", "root_success"):
        if col in out:
            out = out.loc[out[col].astype(str).str.lower().eq("true")]
    out = out.replace([np.inf, -np.inf], np.nan)
    out.attrs["grids"] = grids
    return out


def reference_row(df: pd.DataFrame, required: tuple[str, ...]) -> pd.Series:
    mask = np.ones(len(df), dtype=bool)
    for col in required:
        mask &= np.isclose(df[col].astype(float), REFERENCE[col])
    subset = df.loc[mask]
    if subset.empty:
        raise RuntimeError(f"Reference point is absent from columns {required}.")
    return subset.iloc[0]


def extrema_text(df: pd.DataFrame, x_col: str, y_col: str, x_label: str,
                 y_label: str, x_unit: str, y_unit: str,
                 reference: pd.Series) -> str:
    maximum = df.loc[df["eta_el_net_%"].idxmax()]
    minimum = df.loc[df["eta_el_net_%"].idxmin()]
    text = (
        rf"Design point: $\eta_{{\rm el,net}}={reference['eta_el_net_%']:.2f}\%$" + "\n"
        rf"${x_label}={reference[x_col]:g}\,{x_unit}$; ${y_label}={reference[y_col]:g}\,{y_unit}$" + "\n"
        rf"$\max(\eta_{{\rm el,net}})={maximum['eta_el_net_%']:.2f}\%$" + "\n"
        rf"${x_label}={maximum[x_col]:g}\,{x_unit}$; ${y_label}={maximum[y_col]:g}\,{y_unit}$" + "\n"
        rf"$\min(\eta_{{\rm el,net}})={minimum['eta_el_net_%']:.2f}\%$" + "\n"
        rf"${x_label}={minimum[x_col]:g}\,{x_unit}$; ${y_label}={minimum[y_col]:g}\,{y_unit}$"
    )
    return text


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Input workbook not found: {DATA_FILE}")
    eta = physical(pd.read_excel(DATA_FILE, sheet_name="KPD_comp_turb"))
    dt = physical(pd.read_excel(DATA_FILE, sheet_name="dT_HTHE_LTHE"))
    dp = physical(pd.read_excel(DATA_FILE, sheet_name="dP_hot_cold"))
    required = {"eta_el_net_%", "N_el_net_MW"}
    for name, df in (("efficiency", eta), ("temperature approach", dt), ("pressure loss", dp)):
        missing = required.difference(df.columns)
        if missing:
            raise KeyError(f"{name}: missing columns {sorted(missing)}")
    return eta, dt, dp


def contour(df: pd.DataFrame, x_col: str, y_col: str, z_col: str,
            xlabel: str, ylabel: str, cbar_label: str, filename: str,
            x_math: str, y_math: str, reference: pd.Series,
            show_no_solution: bool = False, show_design_point: bool = True) -> None:
    grids = df.attrs["grids"]
    pivot = (df.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc="mean")
             .reindex(index=grids[y_col], columns=grids[x_col]))
    x, y = pivot.columns.to_numpy(float), pivot.index.to_numpy(float)
    z = np.ma.masked_invalid(pivot.to_numpy(float))
    levels = np.linspace(float(z.min()), float(z.max()), N_COLOR_INTERVALS + 1)
    line_levels = levels
    fig, ax = plt.subplots(figsize=FIG["single"], layout="constrained")
    filled = ax.contourf(x, y, z, levels=levels, cmap="RdBu", extend="neither")
    lines = ax.contour(x, y, z, levels=line_levels, colors="black", linewidths=0.7, alpha=0.80)
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.1f")
    if show_no_solution:
        missing = pivot.isna().to_numpy(dtype=float)
        ax.contourf(x, y, missing, levels=[0.5, 1.5], colors="none", hatches=["///"], zorder=2)
    units = {"KPDcomp_%": r"\%", "KPDturb_%": r"\%", "dT_HTHE_K": "K",
             "dT_LTHE_K": "K", "dP_hot_%": r"\%", "dP_cold_%": r"\%"}
    box_text = extrema_text(df, x_col, y_col, x_math, y_math,
                            units[x_col], units[y_col], reference)
    if show_design_point:
        ax.plot(reference[x_col], reference[y_col], marker="*", markersize=13,
                color=COLORS["reference"], markeredgecolor="white", markeredgewidth=0.8,
                label="Design point", zorder=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_axes(ax)
    handles = []
    if show_design_point:
        handles.append(plt.Line2D([], [], marker="*", markersize=13, color=COLORS["reference"],
                                  markeredgecolor="white", markeredgewidth=0.8, linestyle="None", label="Design point"))
    if show_no_solution:
        handles.append(Patch(facecolor="white", edgecolor="black", hatch="///", label="No physical solution"))
    if handles:
        ax.legend(handles=handles, loc="lower right", frameon=True, edgecolor="0.65")
    ax.text(0.02, 0.98, box_text, transform=ax.transAxes, ha="left", va="top", fontsize=9.0,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.45", "alpha": 0.94})
    cbar = fig.colorbar(filled, ax=ax, pad=0.035, aspect=22)
    cbar.set_label(cbar_label)
    cbar.set_ticks(levels)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: fmt(value)))
    cbar.ax.tick_params(labelsize=FONT["tick"], length=3)
    save(fig, filename)


def efficiency_lines(df: pd.DataFrame, reference: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=FIG["single"], layout="constrained")
    turb = df.loc[np.isclose(df["KPDcomp_%"], REFERENCE["KPDcomp_%"])].sort_values("KPDturb_%")
    comp = df.loc[np.isclose(df["KPDturb_%"], REFERENCE["KPDturb_%"])].sort_values("KPDcomp_%")
    ax.plot(turb["KPDturb_%"], turb["eta_el_net_%"], "o-", color=COLORS["turbine"],
            label=r"$\eta_t$ variation at $\eta_c=80\%$")
    ax.plot(comp["KPDcomp_%"], comp["eta_el_net_%"], "s-", color=COLORS["compressor"],
            label=r"$\eta_c$ variation at $\eta_t=90\%$")
    ax.axhline(reference["eta_el_net_%"], color="0.45", lw=0.8, ls="--")
    ax.plot(90, reference["eta_el_net_%"], "*", ms=13, color=COLORS["reference"],
            markeredgecolor="white", label="Design point")
    max_row = df.loc[df["eta_el_net_%"].idxmax()]
    min_row = df.loc[df["eta_el_net_%"].idxmin()]
    ax.set(xlabel="Isentropic efficiency, %", ylabel=r"Net electrical efficiency, $\eta_{el,net}$, %")
    style_axes(ax)
    ax.legend(loc="lower right", frameon=True, edgecolor="0.65", fontsize=8.2)
    ax.text(0.02, 0.98,
            rf"Design point: $\eta_{{\rm el,net}}={reference['eta_el_net_%']:.2f}\%$" + "\n" +
            rf"$\max(\eta_{{\rm el,net}})={max_row['eta_el_net_%']:.2f}\%$ at $\eta_t={max_row['KPDturb_%']:.0f}\%$, $\eta_c={max_row['KPDcomp_%']:.0f}\%$" + "\n" +
            rf"$\min(\eta_{{\rm el,net}})={min_row['eta_el_net_%']:.2f}\%$ at $\eta_t={min_row['KPDturb_%']:.0f}\%$, $\eta_c={min_row['KPDcomp_%']:.0f}\%$",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.45", "alpha": 0.94})
    save(fig, "02_efficiency_sensitivity_lines")


def temperature_lines(df: pd.DataFrame, reference: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=FIG["single"], layout="constrained")
    htr = df.loc[np.isclose(df["dT_LTHE_K"], REFERENCE["dT_LTHE_K"])].sort_values("dT_HTHE_K")
    ltr = df.loc[np.isclose(df["dT_HTHE_K"], REFERENCE["dT_HTHE_K"])].sort_values("dT_LTHE_K")
    ax.plot(htr["dT_HTHE_K"], htr["eta_el_net_%"], "o-", color=COLORS["turbine"],
            label=r"$\Delta T_{HTR}$ at $\Delta T_{LTR}=5$ K")
    ax.plot(ltr["dT_LTHE_K"], ltr["eta_el_net_%"], "s-", color=COLORS["compressor"],
            label=r"$\Delta T_{LTR}$ at $\Delta T_{HTR}=5$ K")
    ax.plot(5, reference["eta_el_net_%"], "*", ms=13, color=COLORS["reference"], markeredgecolor="white")
    text = extrema_text(df, "dT_HTHE_K", "dT_LTHE_K", r"\Delta T_{HTR}",
                        r"\Delta T_{LTR}", "K", "K", reference)
    ax.set(xlabel="Minimum temperature difference, K", ylabel=r"Net electrical efficiency, %")
    style_axes(ax)
    ax.legend(loc="lower left", frameon=True, edgecolor="0.65", fontsize=8.2)
    # Directly above the lower-left legend.
    ax.text(0.02, 0.52, text, transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.45", "alpha": 0.94})
    save(fig, "04_temperature_approach_lines")


def pressure_lines(df: pd.DataFrame, reference: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=FIG["single"], layout="constrained")
    hot = df.loc[np.isclose(df["dP_cold_%"], REFERENCE["dP_cold_%"])].sort_values("dP_hot_%")
    cold = df.loc[np.isclose(df["dP_hot_%"], REFERENCE["dP_hot_%"])].sort_values("dP_cold_%")
    ax.plot(hot["dP_hot_%"], hot["eta_el_net_%"], "o-", color=COLORS["turbine"],
            label=r"$\Delta p_{hot}$ at $\Delta p_{cold}=2\%$")
    ax.plot(cold["dP_cold_%"], cold["eta_el_net_%"], "s-", color=COLORS["compressor"],
            label=r"$\Delta p_{cold}$ at $\Delta p_{hot}=3\%$")
    ax.plot(3, reference["eta_el_net_%"], "*", ms=13, color=COLORS["reference"], markeredgecolor="white")
    text = extrema_text(df, "dP_hot_%", "dP_cold_%", r"\Delta p_{hot}",
                        r"\Delta p_{cold}", r"\%", r"\%", reference)
    ax.set(xlabel="Relative pressure loss, %", ylabel=r"Net electrical efficiency, %")
    style_axes(ax)
    ax.legend(loc="lower left", frameon=True, edgecolor="0.65", fontsize=8.2)
    # Place the annotation above the lower-left legend.
    ax.text(0.02, 0.52, text, transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.45", "alpha": 0.94})
    save(fig, "06_pressure_loss_lines")


def energy_balance(reference: pd.Series) -> None:
    fields = [("Q_loss_reactor_MW", "Reactor\nloss"),
        ("Q_loss_tr1_MW", "Na loop\nloss"),
        ("Q_loss_RTO_MW", "RTO\nloss"),
        ("Q_transport_loss_CO2_MW", "CO$_2$ transport\nloss"),
        ("Q_EM_ASMM_MW", "Electromechanical\nloss"),
        ("Q_COOL_eff_MW", "Cooler"),
        ("N_el_net_MW", "Net electrical\npower"),
    ]
    values = np.array([float(reference[k]) for k, _ in fields])
    labels = [label for _, label in fields]
    fig, ax = plt.subplots(figsize=FIG["single"], layout="constrained")
    colors = mpl.colormaps["RdBu"](np.linspace(0.18, 0.82, len(values)))
    colors[-1] = mpl.colors.to_rgba(COLORS["reference"])
    bars = ax.bar(np.arange(len(values)), values, color=colors, edgecolor="0.25", linewidth=0.45)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f} MW",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(values)), labels)
    ax.set_ylabel("Power, MW")
    style_axes(ax)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.text(0.02, 0.96, r"Design point: $P_1=31$ MPa; $P_2=8.5$ MPa; $x=0.70$" + "\n" +
            rf"$\eta_{{el,net}}={reference['eta_el_net_%']:.2f}\%$",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.45", "alpha": 0.96})
    save(fig, "07_design_point_energy_balance")


def main() -> None:
    configure()
    eta, dt, dp = load()
    eta_ref = reference_row(eta, ("KPDcomp_%", "KPDturb_%"))
    dt_ref = reference_row(dt, ("dT_HTHE_K", "dT_LTHE_K"))
    dp_ref = reference_row(dp, ("dP_hot_%", "dP_cold_%"))
    values = [eta_ref["eta_el_net_%"], dt_ref["eta_el_net_%"], dp_ref["eta_el_net_%"]]
    if not np.allclose(values, values[0], rtol=0, atol=1e-8):
        raise RuntimeError(f"Reference design point differs across sheets: {values}")

    contour(eta, "KPDcomp_%", "KPDturb_%", "eta_el_net_%", r"$\eta_c$, %", r"$\eta_t$, %",
            r"Net electrical efficiency, %", "01_efficiency_sensitivity_contour", r"\eta_c", r"\eta_t", eta_ref)
    efficiency_lines(eta, eta_ref)
    contour(dt, "dT_HTHE_K", "dT_LTHE_K", "eta_el_net_%", r"$\Delta T_{HTR}$, K", r"$\Delta T_{LTR}$, K",
            r"Net electrical efficiency, %", "03_temperature_approach_contour", r"\Delta T_{HTR}", r"\Delta T_{LTR}", dt_ref,
            show_design_point=False)
    temperature_lines(dt, dt_ref)
    contour(dp, "dP_cold_%", "dP_hot_%", "eta_el_net_%", r"$\Delta p_{cold}$, %", r"$\Delta p_{hot}$, %",
            r"Net electrical efficiency, %", "05_pressure_loss_contour", r"\Delta p_{cold}", r"\Delta p_{hot}", dp_ref,
            show_no_solution=True)
    pressure_lines(dp, dp_ref)
    energy_balance(eta_ref)
    print(f"Created 14 files in: {OUT_DIR}")


if __name__ == "__main__":
    main()
