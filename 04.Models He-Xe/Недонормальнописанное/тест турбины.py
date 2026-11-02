"""Сравнение расчёта турбины и компрессора He–Xe.

Сравниваются две модели изоэнтропического состояния:
1) приближение T2s = T1*(P2/P1)**0.4;
2) решение S(P2,T2s,x) = S(P1,T1,x).
В обоих случаях реальное состояние рассчитывается через энтальпию и КПД.

Запуск: python HeXe_turbine_compressor_comparison.py
Требуется props.npy с интерполяторами свойств He–Xe в той же папке.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROPS_FILE = HERE / "props.npy"
OUT_CSV = HERE / "HeXe_turbine_compressor_comparison.csv"
OUT_TURBINE_PLOT = HERE / "HeXe_turbine_comparison.png"
OUT_COMPRESSOR_PLOT = HERE / "HeXe_compressor_comparison.png"

# Параметры модели Recuperation He-Xe
Q = 2.75e6  # подводимая теплота, Вт
x = 0.283  # мольная доля Xe
T_MAX = 1060 + 273.15  # K, после нагревателя
T_AFTER_PIPE = 1030 + 273.15  # K, вход турбины
T_BEFORE = 760 + 273.15  # K, узел 6; выбранный рабочий режим
T_MIN = 35 + 273.15  # K, вход компрессора
P_IN = 48e5  # Pa, вход турбины
D_P_FIX = 4.553e-2
D_P_HEAT = 0.0364e-2
D_P_COOL = 0.497e-2
ETA_TURBINE = 0.9267
ETA_COMPRESSOR = 0.8706
P_OUT_GRID = np.arange(10, 31, 1, dtype=float) * 1e5  # Pa

props = np.load(PROPS_FILE, allow_pickle=True)
H, S = props[1], props[2]
# Совместимость с некоторыми версиями SciPy и сериализованными интерполяторами
for f in (H, S):
    if not hasattr(f, "grid") and hasattr(f, "_grid"):
        f.grid = f._grid
    if not hasattr(f, "values") and hasattr(f, "_values"):
        f.values = f._values

T_GRID = np.asarray(H.grid[2], dtype=float)
T_LO, T_HI = float(T_GRID[0]), float(T_GRID[-1])


def prop(f, P, T):
    return float(np.asarray(f((x, float(P), float(T)))).squeeze())


def invert_property(f, target, P, T_guess):
    """Найти T из f(x,P,T)=target в пределах температурной сетки свойств."""

    def residual(T):
        return prop(f, P, T) - target

    lo, hi = T_LO, T_HI
    flo, fhi = residual(lo), residual(hi)
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo * fhi > 0:
        # Ищем соседние узлы сетки, между которыми меняется знак невязки.
        vals = np.asarray([residual(t) for t in T_GRID])
        idx = np.where(vals[:-1] * vals[1:] <= 0)[0]
        if len(idx) == 0:
            raise ValueError(f"Не удалось инвертировать свойство при P={P:g} Pa, target={target:g}")
        i = int(idx[np.argmin(np.abs(T_GRID[idx] - T_guess))])
        lo, hi = float(T_GRID[i]), float(T_GRID[i + 1])
    return float(brentq(residual, lo, hi, xtol=1e-8))


def calculate_expander(P1, T1, P2, eta, mode, G, machine):
    h1, s1 = prop(H, P1, T1), prop(S, P1, T1)
    if mode == "formula_0.4":
        T2s = T1 * (P2 / P1) ** 0.4
    elif mode == "entropy":
        T_guess = T1 * (P2 / P1) ** 0.4
        T2s = invert_property(S, s1, P2, T_guess)
    else:
        raise ValueError(mode)
    h2s = prop(H, P2, T2s)
    if machine == "turbine":
        h2 = h1 - eta * (h1 - h2s)
        power = G * (h1 - h2)  # положительная отдаваемая мощность
    elif machine == "compressor":
        h2 = h1 + (h2s - h1) / eta
        power = G * (h1 - h2)  # отрицательная мощность по принятому в исходной модели знаку
    else:
        raise ValueError(machine)
    T2 = invert_property(H, h2, P2, T2s)
    return T2s, T2, power


# Расход соответствует исходной модели и выбранному T_before.
P6 = P_IN * (1 + D_P_HEAT)
h0 = prop(H, P_IN, T_MAX)
h6 = prop(H, P6, T_BEFORE)
G = Q / (h0 - h6)

rows = []
for P_OUT in P_OUT_GRID:
    # Турбина: узел 1 -> узел 2
    P1_T = P_IN
    P2_T = P_OUT
    # Компрессор: узел 4 -> узел 5; давления строго по Brayton() исходного кода
    P4_C = P_OUT * (1 - D_P_FIX - D_P_COOL)
    P5_C = P_IN * (1 + D_P_FIX + D_P_HEAT)
    for machine, P1, T1, P2, eta in (
            ("turbine", P1_T, T_AFTER_PIPE, P2_T, ETA_TURBINE),
            ("compressor", P4_C, T_MIN, P5_C, ETA_COMPRESSOR),
    ):
        a = calculate_expander(P1, T1, P2, eta, "formula_0.4", G, machine)
        b = calculate_expander(P1, T1, P2, eta, "entropy", G, machine)
        rows.append({
            "machine": machine,
            "P_out_cycle_MPa": P_OUT / 1e6,
            "P1_component_MPa": P1 / 1e6,
            "P2_component_MPa": P2 / 1e6,
            "T1_K": T1,
            "T2s_formula_K": a[0],
            "T2s_entropy_K": b[0],
            "T2_formula_K": a[1],
            "T2_entropy_K": b[1],
            "Delta_T2_K": a[1] - b[1],
            "Power_formula_kW": a[2] / 1000,
            "Power_entropy_kW": b[2] / 1000,
            "Delta_power_kW": (a[2] - b[2]) / 1000,
            "Delta_power_abs_kW": abs(a[2] - b[2]) / 1000,
            "G_kg_s": G,
        })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

for machine, plot_path, label in (
        ("turbine", OUT_TURBINE_PLOT, "Турбина"),
        ("compressor", OUT_COMPRESSOR_PLOT, "Компрессор"),
):
    d = df[df["machine"] == machine]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(d["P_out_cycle_MPa"], d["T2_formula_K"], label="Формула с показателем 0,4")
    axes[0].plot(d["P_out_cycle_MPa"], d["T2_entropy_K"], label="Расчёт по энтропии")
    axes[0].set_ylabel("Реальная температура на выходе, K")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(d["P_out_cycle_MPa"], d["Delta_T2_K"], label="ΔT: формула − энтропия")
    axes[1].plot(d["P_out_cycle_MPa"], d["Delta_power_kW"], label="ΔN: формула − энтропия")
    axes[1].set_xlabel("P_out цикла, MPa")
    axes[1].set_ylabel("Разность, K / kW")
    axes[1].grid(True)
    axes[1].legend()
    fig.suptitle(f"{label}: сравнение приближения и расчёта по энтропии")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

print(f"Расход рабочего тела: {G:.6f} kg/s")
print(f"CSV: {OUT_CSV}")
print(f"График турбины: {OUT_TURBINE_PLOT}")
print(f"График компрессора: {OUT_COMPRESSOR_PLOT}")
for machine, name in (("turbine", "Турбина"), ("compressor", "Компрессор")):
    d = df[df["machine"] == machine]
    j = d["Delta_power_abs_kW"].idxmax()
    r = df.loc[j]
    print(f"{name}: max |ΔN| = {r['Delta_power_abs_kW']:.3f} kW at P_out={r['P_out_cycle_MPa']:.1f} MPa; ΔT={r['Delta_T2_K']:.3f} K")
