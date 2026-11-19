"""Верификация табличной модели He--Xe по независимым литературным данным.

Запуск из каталога ``Old``:
    python Literature_Verification.py

Скрипт всегда воспроизводит базовое сопоставление с закрытыми точками НИКИЭТ
из ``Verification.py``. Для публикаций создайте рядом CSV ``literature_points.csv``
в формате, описанном функцией ``write_input_template``. Каждая строка должна
содержать значение, непосредственно снятое из таблицы/приложения статьи; не
подставляйте значения, вычисленные этой же моделью.
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT.parent / "props.npy"
OUT_DIR = ROOT / "literature_verification"
INPUT_FILE = ROOT / "literature_points.csv"
X_XE_MODEL = 0.283
PROPERTY_COLUMNS = {"D": "density", "Cp": "heat_capacity", "Vis": "viscosity"}

# Используйте ID ниже в колонке source_id файла literature_points.csv.
# DOI/идентификатор включены, чтобы результат был проверяемым спустя время.
SOURCES = {
    "TOURNIER_2008": {
        "citation": ("J.-M. Tournier, M. S. El-Genk, Properties of Noble Gases and "
                     "Binary Mixtures for Closed Brayton Cycle Applications, Energy "
                     "Conversion and Management 49 (2008), 469-492."),
        "doi": "10.1016/j.enconman.2007.06.050",
        "use": "Независимая корреляция/база свойств He--Xe; прежде всего Vis и Tc.",
    },
    "JOHNSON_2006": {
        "citation": ("P. K. Johnson, A Method for Calculating Viscosity and Thermal "
                     "Conductivity of a Helium-Xenon Gas Mixture, NASA/CR-2006-214394."),
        "url": "https://ntrs.nasa.gov/citations/20060056311",
        "use": "Vis и теплопроводность; 400--1200 K, смеси с M=20.183, 39.94, 83.8 kg/kmol.",
    },
    "DRAGUNOV_2019": {
        "citation": ("Yu. G. Dragunov et al., Recommendations on calculation of transport "
                     "coefficients and thermodynamic properties of helium-xenon gas mixtures, "
                     "Nuclear Engineering and Design 354 (2019) 110196."),
        "doi": "10.1016/j.nucengdes.2019.110196",
        "use": "D, Cp, Vis и теплопроводность; обзор экспериментальных и расчётных данных.",
    },
    "QIU_2026": {
        "citation": ("S. Qiu et al., Density Measurements for High-Temperature Helium-Xenon "
                     "Mixtures, Journal of Chemical & Engineering Data 71 (2026)."),
        "doi": "10.1021/acs.jced.6c00134",
        "use": "Экспериментальная плотность: 313--773 K, до 6 MPa; особенно важна для низких T.",
    },
}


def load_interpolators() -> dict[str, Callable]:
    names = ("D", "H", "S", "Cp", "Vis", "Tcx")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Не найден файл модели: {MODEL_FILE}")
    return dict(zip(names, np.load(MODEL_FILE, allow_pickle=True)))


def bounds(function: Callable) -> tuple[np.ndarray, np.ndarray]:
    grid = getattr(function, "grid", getattr(function, "_grid", None))
    if grid is None:
        raise TypeError("Интерполятор не содержит координатную сетку.")
    return (np.array([np.asarray(axis)[0] for axis in grid], float),
            np.array([np.asarray(axis)[-1] for axis in grid], float))


def import_internal_points() -> pd.DataFrame:
    """Берёт закрытые точки НИКИЭТ из прежнего скрипта без копирования чисел."""
    path = ROOT / "Verification.py"
    spec = importlib.util.spec_from_file_location("internal_verification", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось загрузить {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows: list[dict] = []
    for item in module.REFERENCE_DATA.values():
        for pressure in module.PRESSURES_MPA:
            for temperature, value in zip(item["T"], item[pressure]):
                rows.append({"source_id": "NIKIET_INTERNAL", "property": item["property"],
                             "x_xe": module.X_XE, "T_K": temperature, "P_MPa": pressure,
                             "reference_value": value, "uncertainty_pct": np.nan})
    return pd.DataFrame(rows)


def write_input_template() -> None:
    if INPUT_FILE.exists():
        return
    with INPUT_FILE.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "property", "x_xe", "T_K", "P_MPa", "reference_value", "uncertainty_pct"])
        writer.writeheader()
        writer.writerow({"source_id": "QIU_2026", "property": "D", "x_xe": "", "T_K": "", "P_MPa": "", "reference_value": "", "uncertainty_pct": ""})


def load_literature_points() -> pd.DataFrame:
    write_input_template()
    external = pd.read_csv(INPUT_FILE)
    required = ["source_id", "property", "x_xe", "T_K", "P_MPa", "reference_value"]
    missing = set(required) - set(external.columns)
    if missing:
        raise ValueError(f"В {INPUT_FILE.name} нет колонок: {sorted(missing)}")
    external = external.dropna(subset=required).copy()  # строка-пример остаётся незаполненной
    if external.empty:
        return external
    if not set(external["source_id"]).issubset(SOURCES):
        unknown = sorted(set(external["source_id"]) - set(SOURCES))
        raise ValueError(f"Неизвестный source_id: {unknown}. Добавьте его в SOURCES.")
    if not set(external["property"]).issubset(PROPERTY_COLUMNS):
        raise ValueError("property допускает только D, Cp или Vis.")
    return external


def calculate(points: pd.DataFrame, f: dict[str, Callable]) -> pd.DataFrame:
    rows = []
    for row in points.to_dict("records"):
        prop = str(row["property"])
        x, p, t = float(row["x_xe"]), float(row["P_MPa"]) * 1e6, float(row["T_K"])
        lo, hi = bounds(f[prop])
        inside = bool(np.all(np.array([x, p, t]) >= lo) and np.all(np.array([x, p, t]) <= hi))
        value = float(f[prop]((x, p, t))) if inside else np.nan
        ref = float(row["reference_value"])
        rows.append({**row, "in_model_domain": inside, "calculated_value": value,
                     "relative_error_pct": (value - ref) / ref * 100 if inside and ref else np.nan})
    return pd.DataFrame(rows)


def make_summary(table: pd.DataFrame) -> pd.DataFrame:
    usable = table.loc[table["in_model_domain"]].copy()
    if usable.empty:
        return pd.DataFrame(columns=["source_id", "property", "n", "MAPE_pct", "bias_pct", "max_abs_error_pct"])
    return usable.groupby(["source_id", "property"], as_index=False).agg(
        n=("relative_error_pct", "size"), MAPE_pct=("relative_error_pct", lambda x: x.abs().mean()),
        bias_pct=("relative_error_pct", "mean"), max_abs_error_pct=("relative_error_pct", lambda x: x.abs().max()))


def plot(table: pd.DataFrame) -> None:
    usable = table.loc[table["in_model_domain"] & table["calculated_value"].notna()]
    if usable.empty:
        return
    for prop, group in usable.groupby("property"):
        fig, ax = plt.subplots(figsize=(6.0, 4.0), layout="constrained")
        for source, part in group.groupby("source_id"):
            ax.scatter(part["reference_value"], part["calculated_value"], s=28, label=source)
        limits = np.array([usable.loc[usable.property.eq(prop), ["reference_value", "calculated_value"]].min().min(),
                           usable.loc[usable.property.eq(prop), ["reference_value", "calculated_value"]].max().max()])
        ax.plot(limits, limits, "k--", lw=1, label="y = x")
        ax.set(xlabel="Литературное значение", ylabel="Модель", title=f"Верификация {prop}")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.savefig(OUT_DIR / f"{PROPERTY_COLUMNS[prop]}_parity.png", dpi=300)
        plt.close(fig)


def write_sources() -> None:
    lines = ["# Источники для верификации He--Xe", "",
             "Внешние точки должны быть независимы от таблицы `props.npy`. "
             "Для каждой точки фиксируйте таблицу/рисунок и способ оцифровки в журнале расчёта.", ""]
    for key, item in SOURCES.items():
        link = f" DOI: https://doi.org/{item['doi']}" if "doi" in item else f" URL: {item['url']}"
        lines.extend([f"## {key}", item["citation"] + link, "", f"Назначение: {item['use']}", ""])
    (OUT_DIR / "sources.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_sources()
    points = pd.concat([import_internal_points(), load_literature_points()], ignore_index=True)
    result = calculate(points, load_interpolators())
    result.to_csv(OUT_DIR / "comparison_points.csv", index=False, encoding="utf-8-sig")
    summary = make_summary(result)
    summary.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    plot(result)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Результаты: {OUT_DIR}")
    print(f"Для добавления статей заполните: {INPUT_FILE}")


if __name__ == "__main__":
    main()
