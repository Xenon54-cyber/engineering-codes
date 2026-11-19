"""Article-ready plots for the main recompression-cycle calculation results.

Reads ``Результаты рекомпрессии.xlsx`` and saves all figures in PNG and SVG
formats to ``figures_main``.  The design point is P1 = 31 MPa, P2 = 8.5 MPa,
and x = 0.70.  The source-table column ``P7_MPa`` is denoted as P2 in figures,
following the notation used in the article.
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
DATA_FILE = ROOT / "Результаты рекомпрессии.xlsx"
OUT_DIR = ROOT / "figures_main"

FONT = {"family": "Arial", "title": 14, "label": 11, "tick": 9, "legend": 9}
FIG = {"single": (5.2, 4.8), "dpi": 600}
COLORS = {"reference": "#1A1A1A", "turbine": "#B2182B", "compressor": "#2166AC"}
DESIGN = {"P1_MPa": 31.0, "P7_MPa": 8.5, "x": 0.70}
X_FIXED = DESIGN["x"]
ETA_LIMIT = 30.0
P1_GRID = np.arange(24.0, 36.0, 1.0)
P2_GRID = np.arange(8.0, 15.5, 0.5)
N_COLOR_INTERVALS = 10


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


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(which="major", length=4.0, width=0.8, pad=3)
    ax.tick_params(which="minor", length=2.1, width=0.65)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(alpha=0.18, linewidth=0.6)


def fmt(value: float) -> str:
    """Format every displayed numeric value with no more than two decimals."""
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def text_two_points_above_legend(fig: plt.Figure, ax: plt.Axes, legend: mpl.legend.Legend,
                                 text: str, fontsize: float) -> None:
    """Add a left-aligned annotation exactly 2 pt above the legend frame."""
    fig.canvas.draw()
    legend_box = legend.get_window_extent(fig.canvas.get_renderer())
    # Anchor the right edge of the wide text box to the legend's right edge.
    # This keeps every line inside the axes while its contents remain left-aligned.
    x_px, y_px = legend_box.x1, legend_box.y1 + 2.0 * fig.dpi / 72.0
    x_ax, y_ax = ax.transAxes.inverted().transform((x_px, y_px))
    ax.text(x_ax, y_ax, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=fontsize,
            multialignment="left",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.45", "alpha": 0.94})


def read_data() -> pd.DataFrame:
    required = [
        "P1_MPa", "P7_MPa", "x", "eta_cycle_%", "eta_el_gross_%", "eta_el_net_%",
        "N_TURB_i_MW", "N_MCOMP_i_MW", "N_RCOMP_i_MW", "N_el_gross_MW", "N_el_net_MW",
    ]
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Input workbook not found: {DATA_FILE}")
    data = pd.read_excel(DATA_FILE, sheet_name="Results")
    missing = set(required).difference(data.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    data = data.dropna(subset=required).copy()
    data["N_COMP_TOTAL_MW"] = data["N_MCOMP_i_MW"] + data["N_RCOMP_i_MW"]
    data["N_INTERNAL_MW"] = data["N_TURB_i_MW"] - data["N_COMP_TOTAL_MW"]
    return data


def design_row(data: pd.DataFrame) -> pd.Series:
    mask = np.ones(len(data), dtype=bool)
    for column, value in DESIGN.items():
        mask &= np.isclose(data[column].astype(float), value)
    point = data.loc[mask]
    if point.empty:
        raise RuntimeError("The specified design point is absent from the results table.")
    return point.iloc[0]


def point_text(row: pd.Series, value_col: str, value_math: str, value_unit: str) -> str:
    return (
        rf"Design point: ${value_math}={row[value_col]:.2f}\,{value_unit}$" + "\n"
        rf"$P_1={fmt(row['P1_MPa'])}\,{{\rm MPa}}$; $P_2={fmt(row['P7_MPa'])}\,{{\rm MPa}}$; $x={fmt(row['x'])}$"
    )


def extrema_text(data: pd.DataFrame, value_col: str, value_math: str, value_unit: str,
                 reference: pd.Series) -> str:
    maximum = data.loc[data[value_col].idxmax()]
    minimum = data.loc[data[value_col].idxmin()]
    return (
        point_text(reference, value_col, value_math, value_unit) + "\n"
        + rf"$\max({value_math})={maximum[value_col]:.2f}\,{value_unit}$ at "
        + rf"$P_1={fmt(maximum['P1_MPa'])}\,{{\rm MPa}}$, $P_2={fmt(maximum['P7_MPa'])}\,{{\rm MPa}}$" + "\n"
        + rf"$\min({value_math})={minimum[value_col]:.2f}\,{value_unit}$ at "
        + rf"$P_1={fmt(minimum['P1_MPa'])}\,{{\rm MPa}}$, $P_2={fmt(minimum['P7_MPa'])}\,{{\rm MPa}}$"
    )


def contour(data: pd.DataFrame, value_col: str, value_math: str, value_unit: str,
            colorbar_label: str, filename: str, reference: pd.Series) -> None:
    pivot = (data.pivot_table(index="P1_MPa", columns="P7_MPa", values=value_col, aggfunc="mean")
             .reindex(index=P1_GRID, columns=P2_GRID))
    x, y = pivot.columns.to_numpy(float), pivot.index.to_numpy(float)
    z = np.ma.masked_invalid(pivot.to_numpy(float))
    levels = np.linspace(float(z.min()), float(z.max()), N_COLOR_INTERVALS + 1)
    line_levels = levels
    fig, ax = plt.subplots(figsize=FIG["single"], layout="constrained")
    filled = ax.contourf(x, y, z, levels=levels, cmap="RdBu", extend="neither")
    lines = ax.contour(x, y, z, levels=line_levels, colors="black", linewidths=0.7, alpha=0.80)
    ax.clabel(lines, inline=True, fontsize=7, fmt=fmt)

    # Preserve the requested calculation domain and make unavailable points explicit.
    missing = pivot.isna().to_numpy(dtype=float)
    # A single hatched contour follows the boundary of the unavailable region.
    # This avoids a checkerboard of individual rectangles at missing grid points.
    ax.contourf(x, y, missing, levels=[0.5, 1.5], colors="none",
                hatches=["///"], zorder=2)

    ax.plot(reference["P7_MPa"], reference["P1_MPa"], marker="*", markersize=13,
            color=COLORS["reference"], markeredgecolor="white", markeredgewidth=0.8,
            label="Design point", zorder=10, clip_on=False)
    ax.set(xlabel=r"$P_2$, MPa", ylabel=r"$P_1$, MPa")
    # The design point lies inside this map; use the exact calculation limits
    # to avoid artificial blank margins outside the P1–P2 grid.
    ax.set_xlim(P2_GRID.min(), P2_GRID.max())
    ax.set_ylim(P1_GRID.min(), P1_GRID.max())
    ax.set_yticks(P1_GRID)
    style_axes(ax)
    no_solution = Patch(facecolor="white", edgecolor="black", hatch="///",
                        label="No physical solution")
    legend = ax.legend(handles=[
        plt.Line2D([], [], marker="*", markersize=13, color=COLORS["reference"],
                   markeredgecolor="white", markeredgewidth=0.8, linestyle="None", label="Design point"),
        no_solution,
    ], loc="lower right", frameon=True, edgecolor="0.65", fontsize=8.2)
    text_two_points_above_legend(
        fig, ax, legend, extrema_text(data, value_col, value_math, value_unit, reference), fontsize=9.0
    )
    cbar = fig.colorbar(filled, ax=ax, pad=0.035, aspect=22)
    cbar.set_label(colorbar_label)
    # One labelled tick at every colour boundary, including the actual maximum.
    cbar.set_ticks(levels)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: fmt(value)))
    cbar.ax.tick_params(labelsize=FONT["tick"], length=3)
    save(fig, filename)


def recompression_fraction(data: pd.DataFrame, reference: pd.Series) -> None:
    work = data.loc[
        np.isclose(data["P1_MPa"], DESIGN["P1_MPa"])
        & np.isclose(data["P7_MPa"], DESIGN["P7_MPa"])
        & (data["eta_el_net_%"] >= ETA_LIMIT)
    ].sort_values("x")
    if work.empty:
        raise RuntimeError("The fixed-pressure flow-split series is absent from the results table.")
    fig, ax = plt.subplots(figsize=FIG["single"], layout="constrained")
    ax.plot(work["x"], work["eta_el_net_%"], "o-", color=COLORS["turbine"], lw=1.8, ms=5,
            label="Calculated fixed-pressure series")
    ax.plot(reference["x"], reference["eta_el_net_%"], "*", ms=13, color=COLORS["reference"],
            markeredgecolor="white", markeredgewidth=0.8, label="Design point", zorder=10)
    best = work.loc[work["eta_el_net_%"].idxmax()]
    text = (
        rf"Fixed-pressure series: $P_1={DESIGN['P1_MPa']:.0f}\,{{\rm MPa}}$; "
        rf"$P_2={DESIGN['P7_MPa']:.1f}\,{{\rm MPa}}$" + "\n"
        rf"$\max(\eta_{{\rm el,net}})={best['eta_el_net_%']:.2f}\,\%$ at $x={best['x']:.2f}$"
    )
    ax.set(xlabel=r"Flow-split ratio, $x$", ylabel=r"Net electrical efficiency, $\eta_{\rm el,net}$, %")
    style_axes(ax)
    legend = ax.legend(loc="lower right", frameon=True, edgecolor="0.65", fontsize=8.2)
    text_two_points_above_legend(fig, ax, legend, text, fontsize=9.0)
    save(fig, "08_recompression_fraction")


def main() -> None:
    configure()
    data = read_data()
    reference = design_row(data)
    fixed = data.loc[np.isclose(data["x"], X_FIXED) & (data["eta_el_net_%"] >= ETA_LIMIT)].copy()
    if len(fixed) < 3:
        raise RuntimeError("Insufficient physical calculation points at the fixed recompression fraction.")
    figures = [
        ("eta_el_net_%", r"\eta_{\rm el,net}", r"\%", r"Net electrical efficiency, %", "01_net_electrical_efficiency"),
        ("eta_el_gross_%", r"\eta_{\rm el,gross}", r"\%", r"Gross electrical efficiency, %", "02_gross_electrical_efficiency"),
        ("eta_cycle_%", r"\eta_{\rm cycle}", r"\%", r"Cycle efficiency, %", "03_cycle_efficiency"),
        ("N_TURB_i_MW", r"N_{\rm turb,i}", r"{\rm MW}", r"Turbine internal power, MW", "04_turbine_internal_power"),
        ("N_COMP_TOTAL_MW", r"N_{\rm comp,i}", r"{\rm MW}", r"Total compressor internal power, MW", "05_compressor_internal_power"),
        ("N_el_gross_MW", r"N_{\rm el,gross}", r"{\rm MW}", r"Gross electrical power, MW", "06_gross_electrical_power"),
        ("N_el_net_MW", r"N_{\rm el,net}", r"{\rm MW}", r"Net electrical power, MW", "07_net_electrical_power"),
    ]
    for args in figures:
        contour(fixed, *args, reference)
    recompression_fraction(data, reference)
    print(f"Created {len(figures) + 1} figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
