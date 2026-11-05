"""
Нумерация узлов:
0 — выход из реактора;
1 — вход в турбину;
2 — выход из турбины / вход горячей стороны рекуператора;
3 — выход горячей стороны рекуператора;
4 — выход охладителя / вход компрессора;
5 — выход компрессора / вход холодной стороны рекуператора;
6 — выход холодной стороны рекуператора / вход в реактор.

Принятые связи давлений:
P0 = P1 = P6;
P3 = P2 - dPh; P4 = P3;
P6 = P5 - dPc.

dPh_frac и dPc_frac — относительные потери давления, доли от давлений
на входах соответствующих сторон рекуператора:
dPh = dPh_frac * P2; dPc = dPc_frac * P1.
Поэтому P5 = P1 + dPc. Все давления внутри расчёта — Па,
температуры свойств — К, энтальпии — Дж/кг.

Требуются файлы props.npy и props_water.npy с интерполяторами свойств,
совместимые с исходной моделью. Интерполятор вызывается как f((x, P, T)).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq, newton
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# 1. Исходные данные
FLUID_HEXE = "He-Xe"
FLUID_WATER = "Water"
PROPERTY_NAMES = ["D", "H", "S", "Cp", "Vis", "Tcx"]

Q_REACTOR_NOM = 2.75e6       # Номинальная мощность, Вт
X_XE = 0.283                 # Мольная доля Xe в He-Xe

T0_C = 1060.0                # Выход из реактора, °C
T1_C = 1030.0                # Вход в турбину, °C
T4_C = 35.0                  # Выход охладителя, °C
T6_C = 800.0                 # Вход в реактор, °C

ETA_TURBINE = 0.9267
ETA_COMPRESSOR = 0.8706
ETA_GENERATOR = 0.98
ETA_TURBINE_MECH = 0.99       # КПД механической передачи турбины
ETA_COMPRESSOR_MECH = 0.99    # КПД механической передачи компрессора
ETA_WATER_PUMP = 0.80
ETA_WATER_MOTOR = 0.95

# Параметры водяного контура. Вспомогательная нагрузка насоса считается
# приближённо по исходной методике.
DP_WATER_FRAC = 0.05
P_WATER = 6.0e5              # Па
DT_WATER_C = 10.0            # °C

# Потери давления рекуператора задаются как доли входных давлений.
# Значения по умолчанию можно изменить под конкретный расчёт.
DP_HOT_FRAC = 0.04553
DP_COLD_FRAC = 0.04553

# Ограничение на минимальный температурный напор рекуператора.
DT_PINCH_MIN_C = 5.0
N_PINCH = 20

# Диапазоны перебора давлений (Па). P1 — вход турбины, P2 — выход турбины.
P1_GRID = np.arange(3.0, 6.01, 0.1) * 1e6
P2_GRID = np.arange(1.0, 4.01, 0.1) * 1e6

# Перебор температуры на выходе холодной стороны рекуператора.
# Проверьте, что диапазон входит в температурную сетку props.npy.
T6_GRID_C = np.arange(650.0, 851.0, 50.0)

# При необходимости можно расширить диапазоны:
# T6_GRID_C = np.arange(650, 851, 1)
# DT_PINCH_GRID_C = np.arange(5, 16, 1)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "res_hexe"
OUT_DIR.mkdir(exist_ok=True)

# 2. Загрузка свойств
def load_property_tables():
    """Загружает интерполяторы свойств из файлов исходной модели."""
    props_path = HERE / "props.npy"
    water_path = HERE / "props_water.npy"
    if not props_path.exists():
        raise FileNotFoundError("Не найден props.npy. Поместите файл с интерполяторами He-Xe " "в рабочую папку PyCharm.")
    if not water_path.exists():
        raise FileNotFoundError("Не найден props_water.npy. Он нужен для оценки мощности " "насоса водяного контура.")

    hexe_values = np.load(props_path, allow_pickle=True)
    water_values = np.load(water_path, allow_pickle=True)

    if len(hexe_values) != len(PROPERTY_NAMES):
        raise ValueError(f"В props.npy ожидалось {len(PROPERTY_NAMES)} интерполяторов, " f"получено {len(hexe_values)}.")
    if len(water_values) != len(PROPERTY_NAMES):
        raise ValueError(f"В props_water.npy ожидалось {len(PROPERTY_NAMES)} интерполяторов, " f"получено {len(water_values)}.")

    result = {FLUID_HEXE: dict(zip(PROPERTY_NAMES, hexe_values)),FLUID_WATER: dict(zip(PROPERTY_NAMES, water_values)),}

    for fluid_props in result.values():
        for f in fluid_props.values():
            if not hasattr(f, "grid") and hasattr(f, "_grid"):
                f.grid = f._grid
            if not hasattr(f, "values") and hasattr(f, "_values"):
                f.values = f._values
    return result

PROPS = None
T_GRID = None
T_LO = None
T_HI = None

def prop(fluid: str, name: str, x: float, p: float, t: float) -> float:
    """Значение свойства при аргументах (x, P, T)."""
    value = PROPS[fluid][name]((x, p, t))
    return float(np.asarray(value))

def temperature_from_property(fluid: str,property_name: str,target_value: float, x: float,p: float,t_guess: float,) -> float:
    """Инверсия H(P,T) или S(P,T) по температурной сетке props.npy.
    Алгоритм повторяет проверенный тест турбины/компрессора:
    сканирование реальных узлов T_GRID и brentq между узлами,
    в которых меняется знак невязки.
    """
    f = PROPS[fluid][property_name]

    def residual(t):
        return float(np.asarray(f((x, float(p), float(t)))).squeeze()) - target_value

    lo, hi = T_LO, T_HI
    flo, fhi = residual(lo), residual(hi)
    if flo == 0:
        return float(lo)
    if fhi == 0:
        return float(hi)

    if flo * fhi > 0:
        vals = np.asarray([residual(t) for t in T_GRID], dtype=float)
        idx = np.where(vals[:-1] * vals[1:] <= 0)[0]
        if len(idx) == 0:
            raise ValueError(f"Не удалось инвертировать {property_name} при P={p:g} Pa, " f"target={target_value:g}. Целевое значение вне диапазона "
                "свойств или корень не найден на температурной сетке.")
        i = int(idx[np.argmin(np.abs(T_GRID[idx] - t_guess))])
        lo, hi = float(T_GRID[i]), float(T_GRID[i + 1])
    return float(brentq(residual, lo, hi, xtol=1e-8))


@dataclass
class State:
    node: int
    T: float
    P: float
    x: float = X_XE
    G: float = np.nan
    H: float = np.nan
    S: float = np.nan
    D: float = np.nan
    Cp: float = np.nan
    Vis: float = np.nan
    Tcx: float = np.nan

    def calculate_properties(self):
        for name in PROPERTY_NAMES:
            setattr(self, name, prop(FLUID_HEXE, name, self.x, self.P, self.T))
        return self

def state_from_h(node: int, p: float, h: float, g: float, t_guess: float) -> State:
    t = temperature_from_property(FLUID_HEXE, "H", h, X_XE, p, t_guess)
    state = State(node=node, T=t, P=p, G=g)
    state.calculate_properties()
    return state

def turbomachine_outlet(inlet: State,outlet_node: int,p_out: float,eta: float,mode: str,) -> tuple[State, float]:
    """
    Расчёт турбины/компрессора по изоэнтропическому КПД.
    mode='turbine': h_out = h_in - eta*(h_in-h_out_s)
    mode='compressor': h_out = h_in + (h_out_s-h_in)/eta
    Возвращает состояние выхода и внутреннюю мощность:
    для турбины положительна, для компрессора отрицательна.
    """
    t_guess = inlet.T * (p_out / inlet.P) ** 0.4
    t_s = temperature_from_property(FLUID_HEXE, "S", inlet.S, inlet.x, p_out, t_guess)
    h_s = prop(FLUID_HEXE, "H", inlet.x, p_out, t_s)

    if mode == "turbine":
        h_out = inlet.H - eta * (inlet.H - h_s)
    elif mode == "compressor":
        h_out = inlet.H + (h_s - inlet.H) / eta
    else:
        raise ValueError("mode должен быть 'turbine' или 'compressor'")

    outlet = state_from_h(outlet_node, p_out, h_out, inlet.G, t_guess)
    power = inlet.G * (inlet.H - outlet.H)
    return outlet, power

def pinch_temperature(states: dict[int, State], n_points: int = N_PINCH) -> float:
    """
    Минимальный температурный напор рекуператора.
    Между узлами 2->3 и 5->6 интерполируются H и P; T восстанавливается
    из свойств при заданных H/P. Узел 5 — холодный вход, узел 6 — выход.
    """
    hot_in, hot_out = states[2], states[3]
    cold_in, cold_out = states[5], states[6]
    dts = []

    for i in range(n_points):
        f = i / (n_points - 1)
        h_hot = hot_in.H + f * (hot_out.H - hot_in.H)
        p_hot = hot_in.P + f * (hot_out.P - hot_in.P)
        t_hot_guess = hot_in.T + f * (hot_out.T - hot_in.T)
        t_hot = temperature_from_property(FLUID_HEXE, "H", h_hot, X_XE, p_hot, t_hot_guess)

        # Противоточный профиль: холодный вход 6 сопоставляется с горячим
        # выходом 3, холодный выход 5 — с горячим входом 2.
        fc = 1.0 - f
        h_cold = cold_in.H + fc * (cold_out.H - cold_in.H)
        p_cold = cold_in.P + fc * (cold_out.P - cold_in.P)
        t_cold_guess = cold_in.T + fc * (cold_out.T - cold_in.T)
        t_cold = temperature_from_property(FLUID_HEXE, "H", h_cold, X_XE, p_cold, t_cold_guess)
        dts.append(t_hot - t_cold)

    return float(np.min(dts))

def water_pump_power(q_cool: float, t_water_in_c: float) -> float:
    """Приближённая мощность насоса водяного контура, Вт."""
    if q_cool <= 0:
        return np.nan

    p_mean = P_WATER * (1.0 - 0.5 * DP_WATER_FRAC)
    t_mean = t_water_in_c + 273.15 + 0.5 * DT_WATER_C
    cp = prop(FLUID_WATER, "Cp", 1.0, p_mean, t_mean)
    rho = prop(FLUID_WATER, "D", 1.0, p_mean, t_mean)
    if cp <= 0 or rho <= 0:
        return np.nan

    return (q_cool * DP_WATER_FRAC * P_WATER/ (cp * rho * DT_WATER_C * ETA_WATER_PUMP * ETA_WATER_MOTOR))

# 3. Расчёт одного цикла по методике пользователя
def calculate_cycle(p1: float,p2: float,dp_hot_frac: float = DP_HOT_FRAC,dp_cold_frac: float = DP_COLD_FRAC,
    eta_t: float = ETA_TURBINE, eta_c: float = ETA_COMPRESSOR,
    dt_pinch_min_c: float = DT_PINCH_MIN_C, t6_c: float | None = None,) -> dict:
    """
    Расчёт одного режима.
    Принята схема пользователя:
      P0 = P1 = P6
      dPh = dp_hot_frac * P2; P3 = P2 - dPh; P4 = P3
      dPc = dp_cold_frac * P1; P5 = P1 + dPc; P6 = P5 - dPc = P1
    Температуры T0, T1, T4, T6 задаются. Потери давления в нагревателе,
    охладителе и трубопроводах не вводятся. Разность T0 и T1 сохраняется
    как заданное граничное условие; Q_pipe_loss рассчитывается отдельно.
    """
    if p1 <= p2:
        return {"valid": False, "reason": "P1 должно быть больше P2"}
    if not (0 < dp_hot_frac < 1 and 0 <= dp_cold_frac < 1):
        return {"valid": False, "reason": "Некорректные относительные потери"}

    # Давления по схеме пользователя.
    dp_h = dp_hot_frac * p2
    dp_c = dp_cold_frac * p1
    p0 = p1
    p1_turbine = p1
    p3 = p2 - dp_h
    p4 = p3
    p5 = p1 + dp_c
    p6 = p1  # заданное целевое условие P6=P1; P5=P6+dPc

    if min(p0, p1_turbine, p2, p3, p4, p5, p6) <= 0:
        return {"valid": False, "reason": "Получено неположительное давление"}

    t0 = T0_C + 273.15
    t1 = T1_C + 273.15
    t4 = T4_C + 273.15
    t6_value_c = T6_C if t6_c is None else float(t6_c)
    t6 = t6_value_c + 273.15

    # Расход из баланса реактора: Qnom = G*(h0-h6).
    h0 = prop(FLUID_HEXE, "H", X_XE, p0, t0)
    h6 = prop(FLUID_HEXE, "H", X_XE, p6, t6)
    delta_h_reactor = h0 - h6
    if not np.isfinite(delta_h_reactor) or delta_h_reactor <= 0:
        return {"valid": False, "reason": "h0-h6 должно быть положительным"}

    g = Q_REACTOR_NOM / delta_h_reactor

    states: dict[int, State] = {}
    states[0] = State(0, t0, p0, G=g).calculate_properties()
    states[1] = State(1, t1, p1_turbine, G=g).calculate_properties()

    # 1 -> 2: турбина.
    states[2], n_turb = turbomachine_outlet(states[1], 2, p2, eta_t, "turbine")
    # 4: выход охладителя / вход компрессора.
    states[4] = State(4, t4, p4, G=g).calculate_properties()
    # 4 -> 5: компрессор. P5 обеспечивает P6=P1 после компенсации dPc.
    states[5], n_comp_signed = turbomachine_outlet(states[4], 5, p5, eta_c, "compressor")
    n_comp_consumed = -n_comp_signed
    # Узел 6: выход холодной стороны рекуператора / вход в реактор.
    states[6] = State(6, t6, p6, G=g).calculate_properties()
    # 2 -> 3 / 5 -> 6: баланс рекуператора.
    # G*(h2-h3)=G*(h6-h5) => h3=h2-(h6-h5).
    h3 = states[2].H - (states[6].H - states[5].H)
    states[3] = state_from_h(3, p3, h3, g, states[5].T)
    # Тепловые потоки по принятому направлению знаков.
    q_reactor = g * (states[0].H - states[6].H)
    q_pipe_loss = g * (states[0].H - states[1].H)
    q_cool = g * (states[3].H - states[4].H)
    q_recup_hot = g * (states[2].H - states[3].H)
    q_recup_cold = g * (states[6].H - states[5].H)
    # Проверка рекуператора.
    recup_balance = q_recup_hot - q_recup_cold
    t_pinch = pinch_temperature(states)

    # Электромеханический баланс: рабочее тело -> вал -> электричество.
    n_turb_shaft = n_turb * ETA_TURBINE_MECH
    n_comp_shaft = n_comp_consumed / ETA_COMPRESSOR_MECH
    n_turb_mech_loss = n_turb - n_turb_shaft
    n_comp_mech_loss = n_comp_shaft - n_comp_consumed
    n_shaft_net = n_turb_shaft - n_comp_shaft
    n_electric_gross = n_shaft_net * ETA_GENERATOR
    n_generator_loss = n_shaft_net - n_electric_gross
    n_electromech_loss = n_turb_mech_loss + n_comp_mech_loss + n_generator_loss

    n_water_pump = water_pump_power(q_cool, T4_C)
    n_electric_net = n_electric_gross - n_water_pump
    n_net_fluid = n_turb - n_comp_consumed
    eta_gross = n_electric_gross / q_reactor if q_reactor > 0 else np.nan
    eta_net = (
        n_electric_net / q_reactor
        if q_reactor > 0 and np.isfinite(n_water_pump)
        else np.nan
    )
    # Баланс цикла для проверки: Qreactor + Qpipe_loss - Qcool
    # должен соответствовать работе турбины/компрессора с учётом принятой
    # знаковой системы и малых численных погрешностей.
    cycle_balance = q_reactor - q_pipe_loss - q_cool - n_net_fluid
    valid = (np.isfinite(t_pinch)
        and t_pinch >= dt_pinch_min_c
        and q_cool >= 0
        and q_recup_hot >= -1e-6
        and q_recup_cold >= -1e-6
        and np.isfinite(eta_net)
        and np.isfinite(n_electric_net)
        and n_electric_net > 0
        and n_shaft_net > 0)

    result = {
        "valid": bool(valid),
        "reason": "" if valid else "Не пройдены проверки pinch/тепловых потоков",
        "T6_C": t6_value_c,
        "P1_MPa": p1 / 1e6,
        "P2_MPa": p2 / 1e6,
        "dPh_Pa": dp_h,
        "dPc_Pa": dp_c,
        "P3_MPa": p3 / 1e6,
        "P4_MPa": p4 / 1e6,
        "P5_MPa": p5 / 1e6,
        "P6_MPa": p6 / 1e6,
        "G_kg_s": g,
        "T2_C": states[2].T - 273.15,
        "T3_C": states[3].T - 273.15,
        "T5_C": states[5].T - 273.15,
        "T_pinch_C": t_pinch,
        "Q_reactor_W": q_reactor,
        "Q_pipe_loss_W": q_pipe_loss,
        "Q_cool_W": q_cool,
        "Q_recup_hot_W": q_recup_hot,
        "Q_recup_cold_W": q_recup_cold,
        "recup_balance_W": recup_balance,
        "N_turbine_fluid_W": n_turb,
        "N_compressor_fluid_W": n_comp_consumed,
        "N_turbine_shaft_W": n_turb_shaft,
        "N_compressor_shaft_W": n_comp_shaft,
        "N_turbine_mech_loss_W": n_turb_mech_loss,
        "N_compressor_mech_loss_W": n_comp_mech_loss,
        "N_shaft_net_W": n_shaft_net,
        "N_generator_loss_W": n_generator_loss,
        "N_electromech_loss_W": n_electromech_loss,
        "N_electric_gross_W": n_electric_gross,
        "N_water_pump_electric_W": n_water_pump,
        "N_electric_net_W": n_electric_net,
        "N_net_fluid_W": n_net_fluid,
        "eta_gross": eta_gross,
        "eta_net": eta_net,
        "cycle_balance_W": cycle_balance,
    }
    for i, s in states.items():
        result[f"T{i}_K"] = s.T
        result[f"P{i}_Pa"] = s.P
        result[f"h{i}_J_kg"] = s.H
        result[f"s{i}_J_kgK"] = s.S
    return result

# 4. Перебор P1/P2 и сохранение результатов
def run_pressure_sweep() -> pd.DataFrame:
    rows = []
    total = len(T6_GRID_C) * len(P1_GRID) * len(P2_GRID)
    combos = product(T6_GRID_C, P1_GRID, P2_GRID)
    for k, (t6_c, p1, p2) in enumerate(combos, start=1):
        try:
            result = calculate_cycle(p1, p2, t6_c=float(t6_c))
        except Exception as exc:
            result = {
                "valid": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "T6_C": T6_C,
        "P1_MPa": p1 / 1e6,
                "P2_MPa": p2 / 1e6,}
        rows.append(result)

        if k % 100 == 0 or k == total:
            print(f"Рассчитано {k}/{total}")
    df = pd.DataFrame(rows)
    output = OUT_DIR / "pressure_sweep.xlsx"
    df.to_excel(output, index=False)
    print(f"Полная таблица сохранена: {output.resolve()}")

    valid_df = df[df["valid"] == True].copy()
    if valid_df.empty:
        print("Допустимых режимов нет. Проверьте диапазоны давлений, свойства и ограничения.")
        return df

    best_idx = valid_df["eta_net"].idxmax()
    best = valid_df.loc[best_idx]
    best_df = pd.DataFrame([best])
    best_path = OUT_DIR / "best_mode.xlsx"
    best_df.to_excel(best_path, index=False)

    print("\nОптимальный допустимый режим:")
    print(f"P1 = {best['P1_MPa']:.3f} МПа")
    print(f"P2 = {best['P2_MPa']:.3f} МПа")
    print(f"G = {best['G_kg_s']:.5f} кг/с")
    print(f"T2 = {best['T2_C']:.2f} °C")
    print(f"T3 = {best['T3_C']:.2f} °C")
    print(f"T5 = {best['T5_C']:.2f} °C")
    print(f"T_pinch = {best['T_pinch_C']:.2f} °C")
    print(f"T6 = {best['T6_C']:.1f} °C")
    print(f"N_electric_gross = {best['N_electric_gross_W']/1e6:.4f} МВт")
    print(f"N_electric_net = {best['N_electric_net_W']/1e6:.4f} МВт")
    print(f"N_electromech_loss = {best['N_electromech_loss_W']/1e3:.3f} кВт")
    print(f"eta_net = {best['eta_net']*100:.3f} %")
    print(f"Лучший режим сохранён: {best_path.resolve()}")

    make_pressure_plot(valid_df)
    return df


def make_pressure_plot(df: pd.DataFrame):
    """Строит карты электрического КПД и тренды по T6."""
    if df.empty or "eta_net" not in df.columns:
        return

    for t6_c, group in df.groupby("T6_C"):
        group = group[group["valid"] == True]
        if group.empty:
            continue
        pivot = group.pivot_table(
            index="P1_MPa", columns="P2_MPa", values="eta_net", aggfunc="max"
        ).sort_index().sort_index(axis=1)
        if pivot.empty or not np.isfinite(pivot.to_numpy(dtype=float)).any():
            continue
        fig, ax = plt.subplots(figsize=(9, 6.5))
        cf = ax.contourf(
            pivot.columns.to_numpy(dtype=float),
            pivot.index.to_numpy(dtype=float),
            np.ma.masked_invalid(pivot.to_numpy(dtype=float)),
            levels=20
        )
        ax.set_xlabel("P₂ — давление после турбины, МПа")
        ax.set_ylabel("P₁ — давление перед турбиной, МПа")
        ax.set_title(f"Электрический КПД нетто He–Xe, T₆ = {t6_c:g} °C")
        fig.colorbar(cf, ax=ax, label="Электрический КПД нетто, доля")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"eta_net_map_T6_{t6_c:g}C.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    valid_df = df[df["valid"] == True].copy()
    if valid_df.empty:
        print("Нет допустимых режимов для построения трендов по T6.")
        return

    best_eta = valid_df.loc[valid_df.groupby("T6_C")["eta_net"].idxmax()].sort_values("T6_C")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(best_eta["T6_C"], best_eta["eta_net"] * 100, marker="o")
    ax.set_xlabel("T₆, °C")
    ax.set_ylabel("Максимальный электрический КПД нетто, %")
    ax.set_title("Влияние T₆ на максимальный допустимый электрический КПД")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eta_net_best_vs_T6.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    for col, scale, ylabel, title, filename in [
        ("N_electric_net_W", 1e6, "Электрическая мощность нетто, МВт",
         "Максимальная электрическая мощность нетто по T₆", "N_electric_net_best_vs_T6.png"),
        ("N_electromech_loss_W", 1e3, "Электромеханические потери, кВт",
         "Электромеханические потери в оптимальных режимах по T₆", "losses_best_vs_T6.png"),
    ]:
        best = valid_df.loc[valid_df.groupby("T6_C")[col].idxmax()].sort_values("T6_C")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(best["T6_C"], best[col] / scale, marker="o")
        ax.set_xlabel("T₆, °C")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    print(f"Графики сохранены в: {OUT_DIR.resolve()}")


def main():
    global PROPS, T_GRID, T_LO, T_HI
    PROPS = load_property_tables()
    H = PROPS[FLUID_HEXE]["H"]
    # Как в тестовом скрипте: координаты интерполятора передаются через .grid.
    if not hasattr(H, "grid") and hasattr(H, "_grid"):
        H.grid = H._grid
    T_GRID = np.asarray(H.grid[2], dtype=float)
    T_LO, T_HI = float(T_GRID[0]), float(T_GRID[-1])
    run_pressure_sweep()
if __name__ == "__main__":
    main()