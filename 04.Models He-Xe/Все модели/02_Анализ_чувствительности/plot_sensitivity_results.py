from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "Результаты_чувствительности_HeXe"
SCENARIOS_FILE = DATA_DIR / "01_сценарии_чувствительности.csv"
BASE_FILE = DATA_DIR / "00_базовый_режим.csv"
CHANGES_FILE = DATA_DIR / "02_отклонения_от_базы.csv"
PINCH_FILE = DATA_DIR / "03_кпд_от_фактического_pinch.csv"
OUT_DIR = DATA_DIR / "figures"

# Все параметры оформления графиков находятся здесь.
FONT = {
    "family": "Times New Roman",
    "title": 11.0,
    "label": 10.0,
    "tick": 8.0,
    "legend": 7.5,
    "annotation": 8.0,
    "weight": "normal",
}

FIGURE = {
    # Единый горизонтальный кадр для слайдов: ширина 5, высота 3 дюйма.
    "single_column": (5.0, 3.0),
    "double_column": (5.0, 3.0),
    "dpi": 300,
    "formats": ("png", "pdf", "svg"),
    "cmap": "viridis",
}

LINE = {"width": 1.45, "marker_size": 4.8, "base_color": "#C00000"}

METRICS = [
    ("eta_asmm_net_initial_heat", 100.0, "КПД нетто АСММ, %"),
    ("N_el_net_MW", 1.0, "Электрическая мощность нетто, МВт"),
    ("N_T_el_MW", 1.0, "Электрическая мощность турбины, МВт"),
    ("N_K_el_consumed_MW", 1.0, "Потребление компрессора, МВт"),
    ("T_pinch_C", 1.0, "Минимальный температурный напор, К"),
    ("Q_pipe_loss_kW", 1.0, "Потери трубопровода, кВт"),
]

GROUPS = {
    "КПД турбины и компрессора": ("eta_turbine", "КПД турбины, доля"),
    "Потери давления рекуператора": ("parameter_value", "Потери давления в рекуператоре, %"),
}


def configure() -> None:
    mpl.rcParams.update({
        "font.family": FONT["family"], "font.weight": FONT["weight"],
        "font.size": FONT["tick"], "axes.titlesize": FONT["title"],
        "axes.labelsize": FONT["label"], "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"], "legend.fontsize": FONT["legend"],
        "mathtext.fontset": "stix", "axes.unicode_minus": False,
        "axes.linewidth": 0.8, "lines.linewidth": LINE["width"],
        "lines.markersize": LINE["marker_size"], "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "legend.frameon": True, "legend.framealpha": 0.96,
        "legend.facecolor": "white", "legend.edgecolor": "0.65",
        "legend.fancybox": False, "savefig.bbox": "standard", "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    })


def save(fig: plt.Figure, name: str, formats: Iterable[str] | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in formats or FIGURE["formats"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=FIGURE["dpi"],
                    facecolor="white", bbox_inches=None)
    plt.close(fig)


def style_axes(ax, *, grid: bool = True) -> None:
    """Единый стиль осей: как на основных графиках проекта."""
    ax.tick_params(which="major", length=3.5, width=0.75, pad=2.5)
    ax.tick_params(which="minor", length=1.8, width=0.6)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if grid:
        ax.grid(alpha=0.18, lw=0.5)


def styled_legend(ax, *, loc: str = "best", ncol: int = 1) -> None:
    ax.legend(loc=loc, ncol=ncol, frameon=True, framealpha=0.96,
              facecolor="white", edgecolor="0.65", fancybox=False)


def load() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    for path in (SCENARIOS_FILE, BASE_FILE, CHANGES_FILE, PINCH_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Сначала запустите sensitivity_analysis.py: {path}")
    scenarios = pd.read_csv(SCENARIOS_FILE)
    scenarios["valid"] = scenarios["valid"].astype(str).str.lower().eq("true")
    base = pd.read_csv(BASE_FILE).iloc[0]
    changes = pd.read_csv(CHANGES_FILE)
    pinch = pd.read_csv(PINCH_FILE)
    return scenarios, base, changes, pinch


def group_response_plots(data: pd.DataFrame, base: pd.Series) -> None:
    group = "Потери давления рекуператора"
    x_col, x_label = GROUPS[group]
    frame = data.loc[data["group"].eq(group) & data["valid"]].sort_values(x_col)
    if frame.empty:
        return
    for number, (column, scale, label) in (
        ("02", METRICS[0]), ("03", METRICS[3]),
    ):
        fig, ax = plt.subplots(figsize=FIGURE["single_column"], constrained_layout=True)
        ax.plot(frame[x_col], scale * frame[column], "o-", color="#1F4E79", label="Сценарии")
        ax.axhline(scale * base[column], color=LINE["base_color"], ls="--", lw=1.0,
                   label="Базовый режим")
        ax.set(xlabel=x_label, ylabel=label, title=label)
        style_axes(ax)
        styled_legend(ax)
        save(fig, f"{number}_{label.replace(', ', '_').replace(' ', '_')}")


def auxiliary_response_plots(data: pd.DataFrame, base: pd.Series) -> None:
    for index, (group, (x_col, x_label)) in enumerate(GROUPS.items(), 1):
        frame = data.loc[data["group"].eq(group) & data["valid"]].sort_values(x_col)
        if frame.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=FIGURE["double_column"], constrained_layout=True)
        for ax, (column, scale, label) in zip(axes, METRICS[4:]):
            ax.plot(frame[x_col], scale * frame[column], "o-", color="#2166AC", label="Сценарии")
            ax.axhline(scale * base[column], color=LINE["base_color"], ls="--", lw=1.0,
                       label="Базовый режим")
            ax.set(xlabel=x_label, ylabel=label, title=label)
            style_axes(ax)
            styled_legend(ax)
        fig.suptitle(f"Дополнительные показатели: {group}", fontsize=FONT["title"])
        save(fig, f"{index:02d}a_дополнительные_{group.replace(' ', '_')}")


def paired_efficiency_map(data: pd.DataFrame) -> None:
    frame = data.loc[data["group"].eq("КПД турбины и компрессора") & data["valid"]].copy()
    if frame.empty:
        return
    pivot = frame.pivot_table(index="eta_turbine", columns="eta_compressor",
                              values="eta_asmm_net_initial_heat", aggfunc="max")
    x, y = pivot.columns.to_numpy(float), pivot.index.to_numpy(float)
    z = np.ma.masked_invalid(100.0 * pivot.to_numpy(float))
    fig, ax = plt.subplots(figsize=FIGURE["double_column"], constrained_layout=True)
    filled = ax.contourf(x, y, z, levels=18, cmap=FIGURE["cmap"], extend="both")
    lines = ax.contour(x, y, z, levels=9, colors="black", linewidths=0.5, alpha=0.6)
    ax.clabel(lines, fmt="%.2f", fontsize=FONT["annotation"] - 1)
    best = frame.loc[frame["eta_asmm_net_initial_heat"].idxmax()]
    ax.plot(best["eta_compressor"], best["eta_turbine"], "*", ms=13,
            color=LINE["base_color"], markeredgecolor="white", label="Максимум КПД")
    ax.annotate(
        f"Максимум\n$\\eta_T$ = {best['eta_turbine']:.3f}\n"
        f"$\\eta_K$ = {best['eta_compressor']:.3f}\n"
        f"КПД = {100 * best['eta_asmm_net_initial_heat']:.3f}%",
        (best["eta_compressor"], best["eta_turbine"]), xytext=(0.57, 0.28),
        textcoords="axes fraction", fontsize=FONT["annotation"], va="top",
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "0.45", "alpha": 0.96},
    )
    colorbar = fig.colorbar(filled, ax=ax, pad=0.025, label="КПД нетто АСММ, %")
    colorbar.ax.tick_params(labelsize=FONT["tick"], length=3)
    ax.set(xlabel="КПД компрессора, доля", ylabel="КПД турбины, доля",
           title="Контурная карта КПД нетто АСММ")
    style_axes(ax)
    styled_legend(ax, loc="lower right")
    save(fig, "04_контурная_карта_кпд_турбомашин")


def relative_change_plot(changes: pd.DataFrame) -> None:
    selected = ["КПД нетто АСММ", "Электрическая мощность нетто, МВт",
                "Электрическая мощность турбины, МВт", "Потребление компрессора, МВт"]
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(FIGURE["double_column"][0], 3.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, group in zip(axes, GROUPS):
        frame = changes.loc[(changes["group"] == group) & changes["valid"] &
                            changes["metric"].isin(selected)].copy()
        for metric, series in frame.groupby("metric"):
            series = series.sort_values("parameter_value")
            ax.plot(series["parameter_value"], series["relative_change_pct"], "o-", label=metric)
        ax.axhline(0, color="0.2", lw=0.7)
        ax.set_title(group)
        ax.set_xlabel(GROUPS[group][1])
        ax.set_ylabel("Отклонение от базы, %")
        style_axes(ax)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True,
               bbox_to_anchor=(0.5, 0.01), fontsize=FONT["legend"])
    save(fig, "05_относительные_изменения")


def actual_pinch_dependence(pinch: pd.DataFrame) -> None:
    """Огибающие максимального КПД по фактическому температурному напору."""
    fig, ax = plt.subplots(figsize=FIGURE["double_column"], constrained_layout=True)
    for t6, group in pinch.groupby("T6_C"):
        group = group.sort_values("T_pinch_C")
        ax.plot(group["T_pinch_C"], 100.0 * group["eta_asmm_net_initial_heat"],
                "o-", label=f"$T_6$ = {t6:.0f} °C")
    best = pinch.loc[pinch["eta_asmm_net_initial_heat"].idxmax()]
    ax.plot(best["T_pinch_C"], 100.0 * best["eta_asmm_net_initial_heat"], "*", ms=13,
            color=LINE["base_color"], markeredgecolor="white", label="Максимум КПД")
    ax.set(xlabel="Фактический минимальный температурный напор, К",
           ylabel="Максимальный КПД нетто АСММ, %",
           title="Влияние фактического температурного напора рекуператора")
    ax.set_xlim(1, 15)
    style_axes(ax)
    styled_legend(ax, loc="lower right", ncol=1)
    ax.text(
        0.97, 0.96,
        f"Максимум: КПД = {100 * best['eta_asmm_net_initial_heat']:.3f}%; "
        f"$T_6$ = {best['T6_C']:.0f} °C; "
        f"$\\Delta T_{{pinch}}$ = {best['T_pinch_C']:.2f} К",
        transform=ax.transAxes, ha="right", va="top", fontsize=FONT["annotation"],
        bbox={"boxstyle": "round,pad=0.30", "fc": "white", "ec": "0.45", "alpha": 0.96},
    )
    save(fig, "06_кпд_от_фактического_pinch")


def base_energy_balance(base: pd.Series) -> None:
    labels = ["Потери\nреактора", "Потери\nтрубопровода", "Охладитель",
              "Электромех.\nпотери", "Собственные\nнужды", "Мощность\nнетто"]
    columns = ["Q_R_loss_W", "Q_pipe_loss_W", "Q_cool_W", "N_electromech_loss_W",
               "N_aux_total_W", "N_el_net_W"]
    values = np.array([base[column] for column in columns], float) / 1e6
    fig, ax = plt.subplots(figsize=FIGURE["double_column"], constrained_layout=True)
    bars = ax.bar(range(len(values)), values, color=mpl.colormaps["Blues"](np.linspace(.35, .85, len(values))))
    bars[-1].set_color(LINE["base_color"])
    for bar, value in zip(bars, values):
        height = bar.get_height()
        inside = height >= 0.35 * values.max()
        ax.text(bar.get_x() + bar.get_width() / 2,
                height * 0.53 if inside else height + 0.018 * values.max(),
                f"{value:.3f}", ha="center", va="center" if inside else "bottom",
                color="white" if inside else "black", fontweight="bold",
                fontsize=FONT["annotation"])
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Мощность, МВт")
    ax.set_title("Энергетический баланс базового режима")
    style_axes(ax, grid=False)
    ax.grid(axis="y", alpha=0.18, lw=0.5)
    save(fig, "07_энергетический_баланс_базы")


def main() -> None:
    configure()
    scenarios, base, changes, pinch = load()
    group_response_plots(scenarios, base)
    paired_efficiency_map(scenarios)
    actual_pinch_dependence(pinch)
    print(f"Графики сохранены: {OUT_DIR}")


if __name__ == "__main__":
    main()
