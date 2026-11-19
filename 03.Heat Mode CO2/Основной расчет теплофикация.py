from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures_heat_supply"

# ==== ФИКСИРОВАННЫЕ ПАРАМЕТРЫ ====
Q_REACTOR_MW = 2.75
T_NA1_IN_C, T_NA1_OUT_C = 550.0, 730.0
TR1_C, TR2_C, T1_C = 730.0, 550.0, 715.0
T_WATER_IN_C, T_WATER_OUT_C = 40.0, 70.0

# Thermal-transport efficiencies: high-pressure and intermediate sodium pipelines.
ETA_REACTOR, ETA_PIPE_1, ETA_PIPE_2 = 0.99, 0.99, 0.99
ETA_RTO = 0.99

CP_NA, RHO_NA = 1300.0, 850.0
CP_WATER, RHO_WATER = 4180.0, 970.0
ETA_PUMP_NA1, ETA_PUMP_NA3, ETA_PUMP_WATER = 0.75, 0.75, 0.75
# Common electric-motor efficiency for the circulation-pump drives.
ETA_MOTOR = 0.98
P_NA1_MPA, P_NA3_MPA, P_WATER_MPA = 1.2, 1.0, 1.6

# ==== ВАРЬИРУЕМЫЕ ПАРАМЕТРЫ ====
T2_MIN_C, T2_MAX_C, T2_STEP_C = 380.0, 545.0, 5.0
DP_MIN_PERCENT, DP_MAX_PERCENT, DP_STEP_PERCENT = 1.0, 15.0, 1.0
ETA_TO_MIN, ETA_TO_MAX, ETA_TO_STEP = 0.90, 0.99, 0.01

# ==== GRAPH STYLE ====
FONT = {"family": "Arial", "label": 11, "tick": 9, "legend": 8}
FIG = {"single": (4.3, 3.7), "dpi": 300}
COLORS = ("#2166AC", "#B2182B", "#4D9221", "#762A83")
N_COLOR_INTERVALS = 10


def configure() -> None:
    mpl.rcParams.update({
        "font.family": FONT["family"], "font.size": FONT["tick"],
        "axes.labelsize": FONT["label"], "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"], "legend.fontsize": FONT["legend"],
        "mathtext.fontset": "stix", "axes.unicode_minus": False,
        "axes.linewidth": 0.9, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(FIGURES_DIR / f"{name}.{suffix}", dpi=FIG["dpi"], facecolor="white")
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(which="major", length=4.0, width=0.8, pad=3)
    ax.tick_params(which="minor", length=2.1, width=0.65)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(alpha=0.18, linewidth=0.6)


def fmt(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")

# ==== РАСХОД Na1 (фикс) ====
delta_T_NA1_C = T_NA1_OUT_C - T_NA1_IN_C
G_NA1_kg_s = Q_REACTOR_MW * 1e6 / (CP_NA * delta_T_NA1_C)

def pump_power_kW(G, dP_MPa, rho, eta_pump, eta_motor):
    return (G / rho) * (dP_MPa * 1e6) / eta_pump / eta_motor / 1000.0

T2_VALUES = np.arange(T2_MIN_C, T2_MAX_C + T2_STEP_C/2, T2_STEP_C)
DP_VALUES = np.arange(DP_MIN_PERCENT, DP_MAX_PERCENT + DP_STEP_PERCENT/2, DP_STEP_PERCENT)
ETA_TO_VALUES = np.round(np.arange(ETA_TO_MIN, ETA_TO_MAX + ETA_TO_STEP/2, ETA_TO_STEP), 2)

def calculate_point(T2_C, DP_percent, ETA_TO):
    Q_ЯТ = Q_REACTOR_MW
    Q_Р   = Q_ЯТ  * ETA_REACTOR
    Q_ТР1 = Q_Р   * ETA_PIPE_1
    Q_HE1 = Q_ТР1 * ETA_RTO
    Q_ТР2 = Q_HE1 * ETA_PIPE_2
    Q_ТО  = Q_ТР2 * ETA_TO

    delta_T_TO_C = T1_C - T2_C
    if delta_T_TO_C <= 0:
        return None
    G_NA3_kg_s = Q_HE1 * 1e6 / (CP_NA * delta_T_TO_C)

    delta_T_WATER_C = T_WATER_OUT_C - T_WATER_IN_C
    G_WATER_kg_s = Q_ТО * 1e6 / (CP_WATER * delta_T_WATER_C)

    dP_NA1   = P_NA1_MPA   * DP_percent / 100.0
    dP_NA3   = P_NA3_MPA   * DP_percent / 100.0
    dP_WATER = P_WATER_MPA * DP_percent / 100.0

    N_NA1   = pump_power_kW(G_NA1_kg_s,   dP_NA1,   RHO_NA,    ETA_PUMP_NA1, ETA_MOTOR)
    N_NA3   = pump_power_kW(G_NA3_kg_s,   dP_NA3,   RHO_NA,    ETA_PUMP_NA3, ETA_MOTOR)
    N_WATER = pump_power_kW(G_WATER_kg_s, dP_WATER, RHO_WATER, ETA_PUMP_WATER, ETA_MOTOR)
    N_SN_TOTAL = N_NA1 + N_NA3 + N_WATER
    N_SN_MW = N_SN_TOTAL / 1000.0

    eta_heat_percent = Q_ТО / Q_ЯТ * 100.0
    eta_net_percent  = (Q_ТО - N_SN_MW) / Q_ЯТ * 100.0

    return {"T2_C": T2_C, "DP_percent": DP_percent, "ETA_TO": ETA_TO,
            "delta_T_TO_C": delta_T_TO_C,
            "Q_ЯТ_MW": Q_ЯТ, "Q_ТО_MW": Q_ТО, "Q_HE1_MW": Q_HE1,
            "G_NA1_kg_s": G_NA1_kg_s, "G_NA3_kg_s": G_NA3_kg_s, "G_WATER_kg_s": G_WATER_kg_s,
            "N_NA1_kW": N_NA1, "N_NA3_kW": N_NA3, "N_WATER_kW": N_WATER,
            "N_SN_TOTAL_kW": N_SN_TOTAL, "N_SN_MW": N_SN_MW,
            "eta_heat_percent": eta_heat_percent, "eta_net_percent": eta_net_percent}

results = []
for T2 in T2_VALUES:
    for DP in DP_VALUES:
        for ETA_TO in ETA_TO_VALUES:
            r = calculate_point(T2, DP, ETA_TO)
            if r is not None:
                results.append(r)
df = pd.DataFrame(results)
df.to_excel("heat_supply_sensitivity.xlsx", index=False)

def plot_lines(x, y, xlabel, ylabel, selector, filename) -> None:
    fig, ax = plt.subplots(figsize=FIG["single"], layout="constrained")
    for color, (label, data) in zip(COLORS, selector()):
        ax.plot(data[x], data[y], "o-", color=color, lw=1.6, ms=3.6, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    style_axes(ax)
    ax.legend(loc="best", frameon=True, edgecolor="0.65")
    save(fig, filename)


def build_map(column: str, i_col: str, j_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = df.pivot_table(index=i_col, columns=j_col, values=column, aggfunc="mean")
    x = pivot.columns.to_numpy(float)
    y = pivot.index.to_numpy(float)
    z = np.ma.masked_invalid(pivot.to_numpy(float))
    return x, y, z


def plot_map(column: str, i_col: str, j_col: str, xlabel: str, ylabel: str,
             cbar_label: str, filename: str) -> None:
    x, y, z = build_map(column, i_col, j_col)
    # Use the same discrete ten-interval colour scale as the other contour maps.
    # Every colour boundary is labelled on the colour bar, including min and max.
    levels = np.linspace(float(z.min()), float(z.max()), N_COLOR_INTERVALS + 1)
    line_levels = levels
    fig, ax = plt.subplots(figsize=FIG["single"], layout="constrained")
    filled = ax.contourf(x, y, z, levels=levels, cmap="RdBu", extend="neither")
    lines = ax.contour(x, y, z, levels=line_levels, colors="black", linewidths=0.7, alpha=0.80)
    ax.clabel(lines, inline=True, fontsize=7, fmt=fmt)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    style_axes(ax)
    cbar = fig.colorbar(filled, ax=ax, pad=0.035, aspect=22)
    cbar.set_label(cbar_label)
    cbar.set_ticks(levels)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: fmt(value)))
    cbar.ax.tick_params(labelsize=FONT["tick"], length=3)
    save(fig, filename)


def selected_by_pressure_loss() -> list[tuple[str, pd.DataFrame]]:
    return [
        (rf"$\Delta P_{{\rm rel}}={pressure:.0f}\,\%$",
         df[np.isclose(df["DP_percent"], pressure) & np.isclose(df["ETA_TO"], 0.99)].sort_values("T2_C"))
        for pressure in (1, 5, 10, 15)
    ]


def selected_by_temperature() -> list[tuple[str, pd.DataFrame]]:
    return [
        (rf"$T_2={temperature:.0f}\,^\circ\mathrm{{C}}$",
         df[np.isclose(df["T2_C"], temperature) & np.isclose(df["ETA_TO"], 0.99)].sort_values("DP_percent"))
        for temperature in (390, 430, 470, 510)
    ]


def selected_by_efficiency() -> list[tuple[str, pd.DataFrame]]:
    return [
        (rf"$\eta_{{\rm HE3}}={efficiency:.2f}$",
         df[np.isclose(df["ETA_TO"], efficiency) & np.isclose(df["DP_percent"], 5.0)].sort_values("T2_C"))
        for efficiency in (0.90, 0.93, 0.96, 0.99)
    ]


def selected_efficiency_by_pressure_loss() -> list[tuple[str, pd.DataFrame]]:
    return [
        (rf"$\Delta P_{{\rm rel}}={pressure:.0f}\,\%$",
         df[np.isclose(df["DP_percent"], pressure) & np.isclose(df["T2_C"], 430.0)].sort_values("ETA_TO"))
        for pressure in (1, 5, 10, 15)
    ]


def make_figures() -> None:
    plot_lines("T2_C", "N_SN_TOTAL_kW", r"Intermediate-sodium temperature, $T_2$, $^\circ$C",
               r"Auxiliary power, kW", selected_by_pressure_loss, "01_auxiliary_power_vs_T2")
    plot_lines("T2_C", "eta_net_percent", r"Intermediate-sodium temperature, $T_2$, $^\circ$C",
               r"Net efficiency, $\eta_{\rm net}$, %", selected_by_pressure_loss, "02_net_efficiency_vs_T2")
    plot_lines("DP_percent", "eta_net_percent", r"Relative pressure loss, $\Delta P_{\rm rel}$, %",
               r"Net efficiency, $\eta_{\rm net}$, %", selected_by_temperature, "03_net_efficiency_vs_pressure_loss")
    plot_lines("ETA_TO", "eta_net_percent", r"Heat-exchanger efficiency, $\eta_{\rm HE3}$",
               r"Net efficiency, $\eta_{\rm net}$, %", selected_efficiency_by_pressure_loss,
               "04_net_efficiency_vs_HE3_efficiency")
    plot_lines("T2_C", "G_NA3_kg_s", r"Intermediate-sodium temperature, $T_2$, $^\circ$C",
               r"Intermediate-sodium mass flow rate, kg s$^{-1}$", selected_by_pressure_loss,
               "05_intermediate_sodium_flow_vs_T2")
    plot_lines("T2_C", "eta_net_percent", r"Intermediate-sodium temperature, $T_2$, $^\circ$C",
               r"Net efficiency, $\eta_{\rm net}$, %", selected_by_efficiency,
               "06_net_efficiency_vs_T2_HE3_efficiency")

    plot_map("eta_net_percent", "T2_C", "DP_percent",
             r"Relative pressure loss, $\Delta P_{\rm rel}$, %",
             r"Intermediate-sodium temperature, $T_2$, $^\circ$C",
             r"Net efficiency, $\eta_{\rm net}$, %", "07_net_efficiency_map_T2_pressure_loss")
    plot_map("eta_net_percent", "T2_C", "ETA_TO",
             r"Heat-exchanger efficiency, $\eta_{\rm HE3}$",
             r"Intermediate-sodium temperature, $T_2$, $^\circ$C",
             r"Net efficiency, $\eta_{\rm net}$, %", "08_net_efficiency_map_T2_HE3_efficiency")
    plot_map("N_SN_TOTAL_kW", "T2_C", "DP_percent",
             r"Relative pressure loss, $\Delta P_{\rm rel}$, %",
             r"Intermediate-sodium temperature, $T_2$, $^\circ$C",
             r"Auxiliary power, kW", "09_auxiliary_power_map_T2_pressure_loss")
    plot_map("G_NA3_kg_s", "T2_C", "ETA_TO",
             r"Heat-exchanger efficiency, $\eta_{\rm HE3}$",
             r"Intermediate-sodium temperature, $T_2$, $^\circ$C",
             r"Intermediate-sodium mass flow rate, kg s$^{-1}$", "10_intermediate_sodium_flow_map_T2_HE3_efficiency")
    plot_map("Q_ТО_MW", "T2_C", "ETA_TO",
             r"Heat-exchanger efficiency, $\eta_{\rm HE3}$",
             r"Intermediate-sodium temperature, $T_2$, $^\circ$C",
             r"Delivered heat, MW", "11_delivered_heat_map_T2_HE3_efficiency")


if __name__ == "__main__":
    configure()
    make_figures()
    print(f"Created 11 figures in: {FIGURES_DIR}")
