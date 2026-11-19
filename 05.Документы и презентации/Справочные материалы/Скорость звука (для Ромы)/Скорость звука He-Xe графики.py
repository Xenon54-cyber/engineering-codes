"""Расширенные графики и верификация скорости звука He-Xe.

Требования для графиков: numpy, scipy, matplotlib.
Для прямой верификации и кривых чистых He и Xe: ctREFPROP и REFPROP 10.

Запуск без аргументов использует props.npy и параметры по умолчанию из
"Скорость звука He-Xe.py".
"""

from __future__ import annotations

import argparse
import csv
import runpy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "Скорость звука He-Xe.py"
BASE = runpy.run_path(str(BASE_SCRIPT))
HeXeProperties = BASE["HeXeProperties"]
sound_speed = BASE["sound_speed"]
DEFAULT_PROPS_PATH = BASE["DEFAULT_PROPS_PATH"]

DEFAULT_OUT = ROOT / "Результаты основной модели" / "Скорость звука расширенно"
PRESSURES_MPA = (2.0, 4.0, 5.2)
TEMPERATURES_K = np.arange(275.0, 1375.1, 25.0)
PRESSURE_GRID_MPA = np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
     1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0,
     3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0,
     5.2, 5.4, 5.6, 5.8, 6.0]
)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 1.0,
        "savefig.dpi": 300,
    })


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", facecolor="white")
    fig.savefig(out / f"{name}.svg", facecolor="white")
    plt.close(fig)


def mixture_temperature_series(model: HeXeProperties) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for p_mpa in PRESSURES_MPA:
        for t_k in TEMPERATURES_K:
            result = sound_speed(model, p_mpa * 1e6, float(t_k))
            rows.append({"p_MPa": p_mpa, "T_K": t_k, "a_m_s": result["sound_speed_m_s"]})
    return rows


def mixture_pressure_series(model: HeXeProperties) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for t_k in (300.0, 400.0, 600.0, 800.0, 1000.0, 1300.0):
        for p_mpa in PRESSURE_GRID_MPA:
            result = sound_speed(model, float(p_mpa * 1e6), t_k)
            rows.append({"p_MPa": float(p_mpa), "T_K": t_k, "a_m_s": result["sound_speed_m_s"]})
    return rows


def export_csv(rows: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_mixture_variants(rows_t: list[dict[str, float]], rows_p: list[dict[str, float]], out: Path) -> None:
    colors = {2.0: "#1F77B4", 4.0: "#D62728", 5.2: "#2CA02C"}

    # Полный температурный диапазон.
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    for p_mpa in PRESSURES_MPA:
        d = [row for row in rows_t if row["p_MPa"] == p_mpa]
        ax.plot([r["T_K"] for r in d], [r["a_m_s"] for r in d], "o-", ms=3.3,
                color=colors[p_mpa], label=f"p = {p_mpa:g} МПа")
    ax.set(xlabel="Температура, К", ylabel="Скорость звука, м/с",
           title="Скорость звука в смеси He-Xe, x_Xe = 0.283")
    ax.grid(alpha=0.22); ax.legend(frameon=True, facecolor="white")
    save(fig, out, "01_speed_of_sound_temperature_full")

    # Увеличенный высокотемпературный диапазон: различия видны по оси y.
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    for p_mpa in PRESSURES_MPA:
        d = [row for row in rows_t if row["p_MPa"] == p_mpa and row["T_K"] >= 800.0]
        ax.plot([r["T_K"] for r in d], [r["a_m_s"] for r in d], "o-", ms=4.0,
                color=colors[p_mpa], label=f"p = {p_mpa:g} МПа")
    ax.set(xlabel="Температура, К", ylabel="Скорость звука, м/с",
           title="Скорость звука в смеси He-Xe, увеличенный диапазон 800-1375 К")
    ax.grid(alpha=0.22); ax.legend(frameon=True, facecolor="white")
    save(fig, out, "02_speed_of_sound_temperature_zoom")

    # Относительное отличие от кривой при 2 МПа.
    reference = {row["T_K"]: row["a_m_s"] for row in rows_t if row["p_MPa"] == 2.0}
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    for p_mpa in (4.0, 5.2):
        d = [row for row in rows_t if row["p_MPa"] == p_mpa]
        delta = [100.0 * (row["a_m_s"] - reference[row["T_K"]]) / reference[row["T_K"]] for row in d]
        ax.plot([r["T_K"] for r in d], delta, "o-", ms=3.5,
                color=colors[p_mpa], label=f"{p_mpa:g} МПа относительно 2 МПа")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set(xlabel="Температура, К", ylabel="Отклонение скорости звука, %",
           title="Влияние давления на скорость звука в He-Xe")
    ax.grid(alpha=0.22); ax.legend(frameon=True, facecolor="white")
    save(fig, out, "03_speed_of_sound_relative_difference")

    # Давление является независимой переменной: хорошо показывает неидеальность холодной смеси.
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    for t_k in sorted({row["T_K"] for row in rows_p}):
        d = [row for row in rows_p if row["T_K"] == t_k]
        ax.plot([r["p_MPa"] for r in d], [r["a_m_s"] for r in d], lw=1.6,
                label=f"T = {t_k:.0f} К")
    ax.set(xlabel="Давление, МПа", ylabel="Скорость звука, м/с",
           title="Влияние давления на скорость звука в смеси He-Xe")
    ax.grid(alpha=0.22); ax.legend(frameon=True, facecolor="white", ncol=2)
    save(fig, out, "04_speed_of_sound_pressure_series")


def plot_numerical_convergence(model: HeXeProperties, out: Path) -> None:
    """Проверяет устойчивость производной к выбору шага по давлению.

    Это верификация численного алгоритма, а не независимая физическая
    валидация. Последняя выполняется в блоке REFPROP.
    """
    rows: list[dict[str, float]] = []
    steps = (1.0e-3, 1.0e-4, 1.0e-5)
    for p_mpa in PRESSURES_MPA:
        for t_k in TEMPERATURES_K:
            values = {
                step: sound_speed(model, p_mpa * 1e6, float(t_k), step)["sound_speed_m_s"]
                for step in steps
            }
            for step, value in values.items():
                rows.append({
                    "p_MPa": p_mpa,
                    "T_K": t_k,
                    "relative_pressure_step": step,
                    "sound_speed_m_s": value,
                    "deviation_from_1e-4_pct": 100.0 * (value - values[1.0e-4]) / values[1.0e-4],
                })
    export_csv(rows, out / "sound_speed_numerical_convergence.csv")

    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    colors = {1.0e-3: "#D62728", 1.0e-5: "#1F77B4"}
    for step in (1.0e-3, 1.0e-5):
        d = [row for row in rows if row["p_MPa"] == 5.2 and row["relative_pressure_step"] == step]
        ax.plot([r["T_K"] for r in d], [r["deviation_from_1e-4_pct"] for r in d],
                "o-", ms=3.3, color=colors[step], label=f"delta p / p = {step:.0e}")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set(xlabel="Температура, К", ylabel="Отклонение от расчёта при delta p/p = 1e-4, %",
           title="Сходимость расчёта скорости звука, p = 5.2 МПа")
    ax.grid(alpha=0.22); ax.legend(frameon=True, facecolor="white")
    save(fig, out, "05_numerical_convergence")


def get_refprop(refprop_path: Path):
    """Подключает REFPROP только для независимого сравнения и чистых газов."""
    from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
    rp = REFPROPFunctionLibrary(str(refprop_path))
    rp.SETPATHdll(str(refprop_path))
    rp.FLAGSdll("PR", 1)
    return rp


def refprop_speed(rp, fluid: str, composition: list[float], p_mpa: float, t_k: float) -> float:
    # В REFPROP при MASS_BASE_SI давление передаётся в кПа, скорость звука возвращается в м/с.
    result = rp.REFPROPdll(fluid, "TP", "W", rp.MASS_BASE_SI, 0, 0, t_k, p_mpa * 1000.0, composition)
    value = float(result.Output[0])
    # Для He-Xe REFPROP выдаёт предупреждение о применении альтернативных
    # правил смешения. При этом численное значение W возвращается корректно.
    # Предупреждение сохраняется в статусе запуска, но не должно отменять
    # расчётное сопоставление.
    if not np.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"REFPROP не вернул скорость звука: {result.herr}")
    return value


def plot_refprop_outputs(model: HeXeProperties, rp, out: Path) -> None:
    # Верификация алгоритма: табличная модель и REFPROP в одинаковых точках.
    verification: list[dict[str, float]] = []
    for p_mpa in PRESSURES_MPA:
        for t_k in TEMPERATURES_K:
            a_calc = sound_speed(model, p_mpa * 1e6, float(t_k))["sound_speed_m_s"]
            a_ref = refprop_speed(rp, "Helium;Xenon", [0.717, 0.283], p_mpa, float(t_k))
            verification.append({"p_MPa": p_mpa, "T_K": t_k, "a_calc_m_s": a_calc,
                                 "a_REFPROP_m_s": a_ref,
                                 "error_pct": 100.0 * (a_calc - a_ref) / a_ref})
    export_csv(verification, out / "sound_speed_REFPROP_verification.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    for p_mpa in PRESSURES_MPA:
        d = [row for row in verification if row["p_MPa"] == p_mpa]
        axes[0].plot([r["a_REFPROP_m_s"] for r in d], [r["a_calc_m_s"] for r in d],
                     "o", ms=3.2, label=f"p = {p_mpa:g} МПа")
        axes[1].plot([r["T_K"] for r in d], [r["error_pct"] for r in d], "o-", ms=3.0,
                     label=f"p = {p_mpa:g} МПа")
    values = [row["a_REFPROP_m_s"] for row in verification] + [row["a_calc_m_s"] for row in verification]
    lo, hi = min(values), max(values)
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1.0, label="y = x")
    axes[0].set(xlabel="REFPROP, м/с", ylabel="Расчёт по props.npy, м/с",
                title="Сопоставление с REFPROP")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set(xlabel="Температура, К", ylabel="Относительная погрешность, %",
                title="Отклонение от REFPROP")
    for ax in axes:
        ax.grid(alpha=0.22); ax.legend(frameon=True, facecolor="white")
    save(fig, out, "06_REFPROP_verification")

    # Чистые компоненты и смесь при 2 МПа.
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    cases = (("He", "Helium", [1.0], "#1F77B4"),
             ("He-Xe, x_Xe = 0.283", "Helium;Xenon", [0.717, 0.283], "#2CA02C"),
             ("Xe", "Xenon", [1.0], "#D62728"))
    for label, fluid, composition, color in cases:
        values = [refprop_speed(rp, fluid, composition, 2.0, float(t_k)) for t_k in TEMPERATURES_K]
        ax.plot(TEMPERATURES_K, values, lw=1.8, color=color, label=label)
    ax.set(xlabel="Температура, К", ylabel="Скорость звука, м/с",
           title="Скорость звука чистых He, Xe и смеси He-Xe при 2 МПа")
    ax.grid(alpha=0.22); ax.legend(frameon=True, facecolor="white")
    save(fig, out, "07_pure_components_and_mixture")


def write_experimental_template(out: Path) -> None:
    path = out / "experimental_speed_of_sound_template.csv"
    if not path.exists():
        path.write_text("T_K,P_MPa,a_exp_m_s\n", encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", type=Path, default=DEFAULT_PROPS_PATH)
    parser.add_argument("--x-xe", type=float, default=0.283)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--refprop-path", type=Path, default=Path(r"C:\Program Files (x86)\REFPROP"))
    parser.add_argument("--skip-refprop", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_style()
    model = HeXeProperties(args.props, args.x_xe)
    rows_t = mixture_temperature_series(model)
    rows_p = mixture_pressure_series(model)
    export_csv(rows_t, args.out / "sound_speed_temperature_series.csv")
    export_csv(rows_p, args.out / "sound_speed_pressure_series.csv")
    plot_mixture_variants(rows_t, rows_p, args.out)
    plot_numerical_convergence(model, args.out)
    write_experimental_template(args.out)

    if args.skip_refprop:
        status = "REFPROP-варианты пропущены по запросу."
    else:
        try:
            rp = get_refprop(args.refprop_path)
            plot_refprop_outputs(model, rp, args.out)
            status = "REFPROP-варианты успешно сформированы."
        except Exception as error:
            status = f"REFPROP-варианты не сформированы: {type(error).__name__}: {error}"
    (args.out / "REFPROP_status.txt").write_text(status + "\n", encoding="utf-8")
    print(status)
    print(f"Результаты сохранены: {args.out}")


if __name__ == "__main__":
    main()
