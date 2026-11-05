from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "pressure_sweep_full.csv"
PROFILE_FILE = ROOT / "recuperator_profile.csv"
STATES_FILE = ROOT / "optimum_states.csv"
OUT_DIR = ROOT / "figures"
# None — автоматически выбрать T6 глобального оптимума; либо число, например 750.
T6_SELECT: float | None = None
FONT = {
    # Параметры соответствуют оформлению контурной карты-образца.
    # Times New Roman хорошо читается в отчёте и сохраняется как текст в PDF/SVG.
    "family": "Times New Roman",
    "title": 18.0,
    "label": 17.0,
    "tick": 13.0,
    "legend": 12.0,
    "annotation": 12.0,
    "weight": "normal",
}
FIGURE = {
    # Ширина и пропорции близки к образцу: квадратная рабочая область
    # плюс узкая вертикальная цветовая шкала.
    "single_column": (6.8, 6.3),
    "double_column": (8.3, 6.3),
    "dpi": 300,
    "formats": ("png", "pdf", "svg"),
    "cmap": "viridis",}

STYLE = {
    "single_size": FIGURE["single_column"],
    "wide_size": FIGURE["double_column"],
    "double_size": FIGURE["double_column"],
    "dpi": FIGURE["dpi"],
    "formats": FIGURE["formats"],
    "cmap": FIGURE["cmap"],
    "contour_levels": 18,
    "contour_lines": 10,
    "line_width": 1.6,
    "marker_size": 6.5,
    "optimum_color": "#C00000",
    "hot_color": "#B2182B",
    "cold_color": "#2166AC",
    "grid_alpha": 0.18,
    "legend_columns": 3,
}

TEXT = {
    "p1": r"Давление перед турбиной, $P_1$ (МПа)",
    "p2": r"Давление после турбины, $P_2$ (МПа)",
    "eta": r"КПД нетто АСММ, $\eta_{net}$ (%)",
    "turbine": "Электрическая мощность турбины, МВт",
    "compressor": "Электрическая мощность компрессора, МВт",
    "net_power": "Электрическая мощность нетто, МВт",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": FONT["family"],
        "font.weight": FONT["weight"],
        "font.size": FONT["tick"],
        "axes.titlesize": FONT["title"],
        "axes.labelsize": FONT["label"],
        "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"],
        "legend.fontsize": FONT["legend"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.linewidth": 0.9,
        "lines.linewidth": STYLE["line_width"],
        "lines.markersize": STYLE["marker_size"],
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.grid": False,
        "legend.frameon": True,
        "legend.framealpha": 0.96,
        "legend.facecolor": "white",
        "legend.edgecolor": "0.65",
        "legend.fancybox": True,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save(fig: plt.Figure, name: str, formats: Iterable[str] | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in formats or STYLE["formats"]:
        fig.savefig(OUT_DIR / f"{name}.{extension}", dpi=STYLE["dpi"],
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


def legend(ax, *, loc: str = "best", outside: bool = False,
           ncol: int = 1) -> None:
    kwargs = {"loc": loc, "ncol": ncol, "frameon": True,
              "framealpha": 0.96, "facecolor": "white",
              "edgecolor": "0.65", "fancybox": False}
    if outside:
        kwargs["bbox_to_anchor"] = (0.5, -0.18)
    ax.legend(**kwargs)


def style_axes(ax, *, square: bool = False, grid: bool = False) -> None:
    """Единое оформление осей для всех статичных рисунков."""
    ax.tick_params(which="major", length=4.0, width=0.8, pad=3)
    ax.tick_params(which="minor", length=2.1, width=0.65)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if square:
        ax.set_box_aspect(1)
    if grid:
        ax.grid(alpha=STYLE["grid_alpha"], lw=0.6)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (DATA_FILE, PROFILE_FILE, STATES_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл: {path}")

    modes = pd.read_csv(DATA_FILE, low_memory=False)
    profile = pd.read_csv(PROFILE_FILE)
    states = pd.read_csv(STATES_FILE)
    modes["valid"] = modes["valid"].astype(str).str.lower().eq("true")

    required = {
        "P1_MPa", "P2_MPa", "T6_C", "valid",
        "eta_asmm_net_initial_heat", "N_T_el_W",
        "N_K_el_consumed_W", "N_el_net_W",
    }
    missing = sorted(required.difference(modes.columns))
    if missing:
        raise KeyError(f"В pressure_sweep_full.csv отсутствуют столбцы: {missing}")
    return modes, profile, states


def valid_modes(modes: pd.DataFrame) -> pd.DataFrame:
    data = modes.loc[modes["valid"]].replace([np.inf, -np.inf], np.nan).copy()
    data = data.dropna(subset=["eta_asmm_net_initial_heat", "N_el_net_W"])
    if data.empty:
        raise RuntimeError("В таблице нет допустимых режимов.")
    return data


def global_optimum(data: pd.DataFrame) -> pd.Series:
    return data.loc[data["eta_asmm_net_initial_heat"].idxmax()]


def selected_slice(data: pd.DataFrame, optimum: pd.Series) -> tuple[pd.DataFrame, float]:
    requested = float(optimum["T6_C"] if T6_SELECT is None else T6_SELECT)
    available = np.sort(data["T6_C"].dropna().unique().astype(float))
    chosen = float(available[np.argmin(np.abs(available - requested))])
    return data.loc[np.isclose(data["T6_C"], chosen)].copy(), chosen


def optimum_in_slice(data: pd.DataFrame) -> pd.Series:
    return data.loc[data["eta_asmm_net_initial_heat"].idxmax()]


def pivot_grid(data: pd.DataFrame, column: str, scale: float = 1.0
               ) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    pivot = data.pivot_table(index="P1_MPa", columns="P2_MPa",
                             values=column, aggfunc="max")
    if pivot.empty:
        raise RuntimeError(f"Не удалось построить сетку для {column}.")
    x = pivot.columns.to_numpy(float)
    y = pivot.index.to_numpy(float)
    z = np.ma.masked_invalid(pivot.to_numpy(float) * scale)
    return x, y, z


def mark_optimum(ax, row: pd.Series, value: float, unit: str) -> None:
    x, y = float(row["P2_MPa"]), float(row["P1_MPa"])
    ax.plot(x, y, marker="*", ms=18, color=STYLE["optimum_color"],
            markeredgecolor="white", markeredgewidth=0.9,
            label="Оптимум по КПД", zorder=10)
    ax.annotate(
        f"Оптимум\n$P_1$ = {y:.2f} МПа\n$P_2$ = {x:.2f} МПа\n{value:.3f} {unit}",
        xy=(x, y), xytext=(0.60, 0.22), textcoords="axes fraction",
        fontsize=FONT["annotation"], ha="left", va="top",
        bbox={"boxstyle": "round,pad=0.28", "fc": "white",
              "ec": "0.45", "lw": 0.8, "alpha": 0.96},
    )


def contour_plot(data: pd.DataFrame, optimum: pd.Series, *, column: str,
                 scale: float, colorbar_label: str, title: str,
                 filename: str, optimum_unit: str,
                 figsize: tuple[float, float] | None = None) -> None:
    x, y, z = pivot_grid(data, column, scale)
    fig, ax = plt.subplots(figsize=figsize or STYLE["single_size"], layout="constrained")
    filled = ax.contourf(x, y, z, levels=STYLE["contour_levels"],
                         cmap=STYLE["cmap"], extend="both")
    lines = ax.contour(x, y, z, levels=STYLE["contour_lines"],
                       colors="black", linewidths=0.55, alpha=0.58)
    ax.clabel(lines, inline=True, fontsize=FONT["annotation"] - 2,
              fmt="%.2f", colors="black")
    colorbar = fig.colorbar(filled, ax=ax, pad=0.02, shrink=0.96, aspect=22)
    colorbar.set_label(colorbar_label)
    colorbar.ax.tick_params(labelsize=FONT["tick"], length=3.5)
    ax.set_xlabel(TEXT["p2"])
    ax.set_ylabel(TEXT["p1"])
    ax.set_title(title)
    style_axes(ax, square=True)
    mark_optimum(ax, optimum, float(optimum[column]) * scale, optimum_unit)
    legend(ax, loc="lower right")
    save(fig, filename)


def efficiency_surface(data: pd.DataFrame, optimum: pd.Series, t6: float) -> None:
    x, y, z = pivot_grid(data, "eta_asmm_net_initial_heat", 100.0)
    xx, yy = np.meshgrid(x, y)
    fig = plt.figure(figsize=STYLE["double_size"], layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(xx, yy, z, cmap=STYLE["cmap"],
                              linewidth=0, antialiased=True, alpha=0.94)
    value = 100.0 * float(optimum["eta_asmm_net_initial_heat"])
    ax.scatter(optimum["P2_MPa"], optimum["P1_MPa"], value,
               marker="*", s=90, color=STYLE["optimum_color"],
               edgecolor="white", linewidth=0.5, label="Оптимум по КПД")
    ax.text(optimum["P2_MPa"], optimum["P1_MPa"], value,
            f"  максимум = {value:.3f}%", fontsize=FONT["annotation"])
    ax.set_xlabel(r"$P_2$, МПа")
    ax.set_ylabel(r"$P_1$, МПа")
    ax.set_zlabel(TEXT["eta"])
    ax.set_title(f"Поверхность КПД нетто АСММ; $T_6$ = {t6:.0f} °C")
    fig.colorbar(surface, ax=ax, shrink=0.72, pad=0.10, label="КПД, %")
    legend(ax, loc="upper left")
    save(fig, "04_3d_поверхность_кпд")


def recuperator_temperature(profile: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=STYLE["wide_size"], constrained_layout=True)
    ax.plot(profile["x_L"], profile["T_hot_C"], color=STYLE["hot_color"],
            label="Горячий поток (2→3)")
    ax.plot(profile["x_L"], profile["T_cold_C"], color=STYLE["cold_color"],
            label="Холодный поток (6→5, противоток)")
    j = profile["delta_T_K"].astype(float).idxmin()
    row = profile.loc[j]
    ax.vlines(row["x_L"], row["T_cold_C"], row["T_hot_C"],
              color=STYLE["optimum_color"], ls="--", lw=1.1,
              label="Минимальный температурный напор")
    ax.annotate(
        fr"$\Delta T_{{min}}$ = {row['delta_T_K']:.2f} К" + "\n" +
        fr"$x/L$ = {row['x_L']:.3f}",
        xy=(row["x_L"], (row["T_hot_C"] + row["T_cold_C"]) / 2),
        xytext=(12, 5), textcoords="offset points",
        fontsize=FONT["annotation"],
        bbox={"boxstyle": "round,pad=0.28", "fc": "white",
              "ec": "0.45", "alpha": 0.96},
    )
    ax.set_xlabel("Относительная координата горячего потока, x/L")
    ax.set_ylabel("Температура, °C")
    ax.set_title("Температурный профиль рекуператора в оптимальном режиме")
    style_axes(ax, grid=True)
    legend(ax, loc="best")
    save(fig, "05_температурный_профиль_рекуператора")


def energy_balance(optimum: pd.Series) -> None:
    labels = [
        "Потери\nреактора", "Потери\nтрубопровода", "Отвод в\nохладителе",
        "Электромех.\nпотери", "Собственные\nнужды", "Мощность\nнетто",
    ]
    values = np.array([
        optimum["Q_R_loss_W"], optimum["Q_pipe_loss_W"], optimum["Q_cool_W"],
        optimum["N_electromech_loss_W"], optimum["N_aux_total_W"],
        optimum["N_el_net_W"],
    ], dtype=float) / 1e6
    fig, ax = plt.subplots(figsize=STYLE["wide_size"], constrained_layout=True)
    colors = mpl.colormaps["Blues"](np.linspace(0.35, 0.88, len(values)))
    colors[-1] = mpl.colors.to_rgba("#C00000")
    bars = ax.bar(np.arange(len(values)), values, color=colors,
                  edgecolor="0.25", linewidth=0.45)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values],
                 padding=3, fontsize=FONT["annotation"])
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("Мощность, МВт")
    ax.set_title("Распределение начальной тепловой мощности в оптимальном режиме")
    style_axes(ax)
    ax.grid(axis="y", alpha=STYLE["grid_alpha"], lw=0.6)
    eta = 100.0 * optimum["eta_asmm_net_initial_heat"]
    ax.text(
        0.99, 0.96,
        f"КПД нетто АСММ = {eta:.3f}%\n"
        f"$P_1$ = {optimum['P1_MPa']:.2f} МПа; $P_2$ = {optimum['P2_MPa']:.2f} МПа",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=FONT["annotation"],
        bbox={"boxstyle": "round,pad=0.3", "fc": "white",
              "ec": "0.45", "alpha": 0.96},
    )
    save(fig, "06_тепловой_баланс")


def pressure_ratio_analysis(data: pd.DataFrame, optimum: pd.Series) -> None:
    # --- Подготовка данных ---
    reduced_data = []
    for t6, group in data.groupby("T6_C"):
        reduced = group.groupby("pressure_ratio", as_index=False).agg(
            specific_turbine_work_kJ_kg=("specific_turbine_work_kJ_kg", "max"),
            specific_compressor_work_kJ_kg=("specific_compressor_work_kJ_kg", "min"),
            eta=("eta_asmm_net_initial_heat", "max"),
        ).sort_values("pressure_ratio")
        reduced["T6_C"] = t6
        reduced_data.append(reduced)

    full_reduced = pd.concat(reduced_data, ignore_index=True)

    # --- График 07a: Удельная работа турбины ---
    fig1, ax1 = plt.subplots(figsize=STYLE["wide_size"], constrained_layout=True)
    for t6, group in full_reduced.groupby("T6_C"):
        ax1.plot(group["pressure_ratio"], group["specific_turbine_work_kJ_kg"],
                 label=f"$T_6$ = {t6:.0f} °C")

    ax1.set_xlabel(r"Степень расширения, $P_1/P_2$")
    ax1.set_ylabel("Удельная работа турбины, кДж/кг")
    ax1.set_title("Удельная работа турбины")
    style_axes(ax1, grid=True)
    legend(ax1, loc="upper center", outside=True, ncol=STYLE["legend_columns"])
    save(fig1, "07a_удельная_работа_турбины")

    # --- График 07b: Влияние степени расширения на КПД ---
    fig2, ax2 = plt.subplots(figsize=STYLE["wide_size"], constrained_layout=True)
    for t6, group in full_reduced.groupby("T6_C"):
        ax2.plot(group["pressure_ratio"], 100.0 * group["eta"],
                 label=f"$T_6$ = {t6:.0f} °C")

    ax2.plot(optimum["pressure_ratio"], 100 * optimum["eta_asmm_net_initial_heat"],
             "*", ms=11, color=STYLE["optimum_color"],
             markeredgecolor="white", label="Глобальный оптимум")

    ax2.set_xlabel(r"Степень расширения, $P_1/P_2$")
    ax2.set_ylabel("КПД нетто АСММ, %")
    ax2.set_title("Влияние степени расширения на КПД")
    style_axes(ax2, grid=True)
    legend(ax2, loc="upper center", outside=True, ncol=STYLE["legend_columns"])
    save(fig2, "07b_влияние_степени_расширения_на_кпд")


def optimum_vs_t6(data: pd.DataFrame) -> None:
    best = data.loc[data.groupby("T6_C")["eta_asmm_net_initial_heat"].idxmax()].sort_values("T6_C")
    fig, axes = plt.subplots(1, 2, figsize=STYLE["double_size"], constrained_layout=True)
    axes[0].plot(best["T6_C"], 100 * best["eta_asmm_net_initial_heat"],
                 "o-", color="#1F4E79", label="Максимальный КПД")
    axes[1].plot(best["T6_C"], best["N_el_net_W"] / 1e6,
                 "s-", color="#C00000", label="Мощность нетто")
    axes[0].set(xlabel=r"Температура $T_6$, °C", ylabel="КПД нетто АСММ, %",
                title="Максимальный КПД при каждом $T_6$")
    axes[1].set(xlabel=r"Температура $T_6$, °C", ylabel="Мощность нетто, МВт",
                title="Мощность оптимального режима")
    for ax in axes:
        style_axes(ax, grid=True)
        legend(ax)
    save(fig, "08_оптимум_в_зависимости_от_T6")


def cycle_ts(states: pd.DataFrame) -> None:
    state_map = states.set_index("node")
    order = [0, 1, 2, 3, 4, 5, 6, 0]
    cycle = state_map.loc[order]
    fig, ax = plt.subplots(figsize=STYLE["single_size"], constrained_layout=True)
    ax.plot(cycle["s_j_kgk"] / 1e3, cycle["t_c"], "o-",
            color="#1F4E79", label="Цикл He–Xe")
    for node, row in state_map.iterrows():
        ax.annotate(str(int(node)), (row["s_j_kgk"] / 1e3, row["t_c"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=FONT["annotation"])
    ax.set_xlabel("Энтропия, кДж/(кг·К)")
    ax.set_ylabel("Температура, °C")
    ax.set_title("T–s-диаграмма оптимального режима")
    style_axes(ax, grid=True)
    legend(ax)
    save(fig, "09_Ts_диаграмма")


def main() -> None:
    configure_matplotlib()
    modes, profile, states = load_data()
    data = valid_modes(modes)
    optimum = global_optimum(data)
    slice_data, t6 = selected_slice(data, optimum)
    slice_optimum = optimum_in_slice(slice_data)

    contour_plot(
        slice_data, slice_optimum,
        column="eta_asmm_net_initial_heat", scale=100.0,
        colorbar_label=TEXT["eta"],
        title=f"Электрический КПД нетто АСММ; $T_6$ = {t6:.0f} °C",
        filename="01_контурная_карта_кпд", optimum_unit="%",
    )
    contour_plot(
        slice_data, slice_optimum,
        column="N_T_el_W", scale=1e-6,
        colorbar_label=TEXT["turbine"],
        title=f"Электрическая мощность турбины; $T_6$ = {t6:.0f} °C",
        filename="02_контурная_карта_турбины", optimum_unit="МВт",
    )
    contour_plot(
        slice_data, slice_optimum,
        column="N_K_el_consumed_W", scale=1e-6,
        colorbar_label=TEXT["compressor"],
        title=f"Электрическая мощность компрессора; $T_6$ = {t6:.0f} °C",
        filename="03_контурная_карта_компрессора", optimum_unit="МВт",
    )
    contour_plot(
        slice_data, slice_optimum,
        column="N_el_net_W", scale=1e-6,
        colorbar_label=TEXT["net_power"],
        title=f"Электрическая мощность нетто; $T_6$ = {t6:.0f} °C",
        filename="03b_контурная_карта_мощности_нетто", optimum_unit="МВт",
    )
    efficiency_surface(slice_data, slice_optimum, t6)
    recuperator_temperature(profile)
    energy_balance(optimum)
    pressure_ratio_analysis(data, optimum)
    optimum_vs_t6(data)
    cycle_ts(states)

    print(f"Графики сохранены: {OUT_DIR}")
    print(f"Использованный срез: T6 = {t6:.0f} °C")
    print(f"КПД глобального оптимума: {100 * optimum['eta_asmm_net_initial_heat']:.3f} %")


if __name__ == "__main__":
    main()
