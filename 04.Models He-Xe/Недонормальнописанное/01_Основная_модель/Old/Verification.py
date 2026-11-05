"""Сопоставление табличной модели свойств смеси He–Xe со справочными данными.

Состав смеси: x_Xe = 0.283, M_mix ≈ 40.02 г/моль.
Диапазон: P = 2.0 и 4.0 МПа; T = 350…1400 К.
Сплошные линии — расчётная модель, открытые маркеры — справочные данные.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator


ROOT = Path(__file__).resolve().parent
PROPS_FILE = ROOT.parent / "props.npy"
OUT_DIR = ROOT.parent / "res" / "verify"
X_XE = 0.283
PRESSURES_MPA = (2.0, 4.0)

FONT = {"family": "Times New Roman", "title": 14.0, "label": 12.0,
        "tick": 10.0, "legend": 10.5, "annotation": 10.0}
FIGURE = {"size": (5.0, 3.0), "dpi": 300, "formats": ("png", "pdf", "svg")}
COLORS, MARKERS = {2.0: "#1F77B4", 4.0: "#2CA02C"}, {2.0: "o", 4.0: "s"}

REFERENCE_DATA = {
    "viscosity": {"property": "Vis", "symbol": r"$\mu$", "unit": "мкПа·с",
        "title": "Динамическая вязкость смеси He–Xe",
        "T": np.array([370, 380, 500, 550, 1000, 1050, 1300, 1350, 1400]),
        2.0: np.array([31.8339, 32.5211, 40.5103, 43.6828, 68.5243, 70.9777, 82.5614, 84.7677, 86.9385]),
        4.0: np.array([32.9327, 33.5541, 41.1615, 44.2596, 68.8317, 71.2769, 82.8050, 85.0080, 87.1822])},
    "heat_capacity": {"property": "Cp", "symbol": r"$c_p$", "unit": "Дж/(кг·К)",
        "title": "Изобарная теплоёмкость смеси He–Xe",
        "T": np.array([350, 400, 500, 600, 1000, 1100, 1300, 1400]),
        2.0: np.array([542.32, 534.15, 527.10, 524.22, 520.78, 520.50, 520.13, 520.04]),
        4.0: np.array([577.50, 553.18, 535.82, 529.42, 522.27, 521.62, 520.88, 520.60])},
    "density": {"property": "D", "symbol": r"$\rho$", "unit": "кг/м³",
        "title": "Плотность смеси He–Xe",
        "T": np.array([370, 380, 500, 550, 1000, 1050, 1300, 1350]),
        2.0: np.array([27.4382, 26.6050, 19.5953, 17.7115, 9.5990, 9.1389, 7.3772, 7.1038]),
        4.0: np.array([58.2439, 56.1030, 39.8845, 35.8187, 19.1390, 18.2143, 14.6967, 14.1532])},
}


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": FONT["family"], "font.size": FONT["tick"],
        "axes.titlesize": FONT["title"], "axes.labelsize": FONT["label"],
        "xtick.labelsize": FONT["tick"], "ytick.labelsize": FONT["tick"],
        "legend.fontsize": FONT["legend"], "mathtext.fontset": "stix",
        "axes.linewidth": 0.8, "lines.linewidth": 1.45,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "legend.frameon": True, "legend.framealpha": 0.96,
        "legend.facecolor": "white", "legend.edgecolor": "0.65",
        "legend.fancybox": False, "savefig.bbox": "standard", "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    })


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(which="major", length=3.5, width=0.75, pad=2.5)
    ax.tick_params(which="minor", length=1.8, width=0.6)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(alpha=0.18, lw=0.5)


def save_figure(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in FIGURE["formats"]:
        fig.savefig(OUT_DIR / f"{name}.{extension}", dpi=FIGURE["dpi"],
                    facecolor="white", bbox_inches=None)
    plt.close(fig)


def load_interpolators() -> dict[str, Callable]:
    if not PROPS_FILE.exists():
        raise FileNotFoundError(f"Не найден файл свойств: {PROPS_FILE}")
    names = ("D", "H", "S", "Cp", "Vis", "Tcx")
    return dict(zip(names, np.load(PROPS_FILE, allow_pickle=True)))


def evaluate(spec: dict, interpolators: dict[str, Callable]) -> pd.DataFrame:
    rows = []
    function = interpolators[spec["property"]]
    for pressure in PRESSURES_MPA:
        for temperature, reference in zip(spec["T"], spec[pressure]):
            calculated = float(function((X_XE, pressure * 1e6, temperature)))
            rows.append({"T_K": temperature, "P_MPa": pressure,
                         "reference_value": reference, "calculated_value": calculated,
                         "relative_error_pct": (calculated - reference) / reference * 100.0})
    return pd.DataFrame(rows)


def plot_comparison(spec: dict, table: pd.DataFrame,
                    interpolators: dict[str, Callable], name: str) -> None:
    fig, ax = plt.subplots(figsize=FIGURE["size"], layout="constrained")
    function = interpolators[spec["property"]]
    for pressure in PRESSURES_MPA:
        reference = table.loc[table["P_MPa"].eq(pressure)]
        t_line = np.linspace(reference["T_K"].min(), reference["T_K"].max(), 200)
        y_line = [float(function((X_XE, pressure * 1e6, temperature)))
                  for temperature in t_line]
        color = COLORS[pressure]
        ax.plot(t_line, y_line, color=color, label=f"Расчёт, {pressure:.0f} МПа")
        ax.scatter(reference["T_K"], reference["reference_value"], s=27,
                   marker=MARKERS[pressure], facecolors="white", edgecolors=color,
                   linewidths=1.0, label=f"Справочные данные, {pressure:.0f} МПа", zorder=4)
    ax.set(xlabel="Температура, К", ylabel=f"{spec['symbol']}, {spec['unit']}",
           title=spec["title"])
    style_axes(ax)
    ax.legend(loc="best")
    save_figure(fig, name)


def make_summary(name: str, spec: dict, table: pd.DataFrame) -> list[dict]:
    rows = []
    for pressure in PRESSURES_MPA:
        errors = table.loc[table["P_MPa"].eq(pressure), "relative_error_pct"].abs()
        rows.append({"property": name, "symbol": spec["symbol"], "unit": spec["unit"],
                     "P_MPa": pressure, "mean_absolute_error_pct": errors.mean(),
                     "max_absolute_error_pct": errors.max()})
    return rows


def main() -> None:
    configure_matplotlib()
    interpolators = load_interpolators()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, spec in REFERENCE_DATA.items():
        table = evaluate(spec, interpolators)
        table.to_csv(OUT_DIR / f"{name}_comparison.csv", index=False, encoding="utf-8-sig")
        table.to_excel(OUT_DIR / f"{name}_comparison.xlsx", index=False)
        plot_comparison(spec, table, interpolators, f"{name}_comparison")
        summary.extend(make_summary(name, spec, table))
    report = pd.DataFrame(summary)
    report.to_csv(OUT_DIR / "verification_summary.csv", index=False, encoding="utf-8-sig")
    report.to_excel(OUT_DIR / "verification_summary.xlsx", index=False)
    print("Сопоставление со справочными данными завершено:")
    print(report.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Результаты сохранены: {OUT_DIR}")


if __name__ == "__main__":
    main()
