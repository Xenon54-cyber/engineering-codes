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
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter

# 1. Исходные данные
FLUID_HEXE = "He-Xe"
FLUID_WATER = "Water"
PROPERTY_NAMES = ["D", "H", "S", "Cp", "Vis", "Tcx"]

Q_REACTOR_NOM = 2.75e6       # Q_R_nom: тепловая мощность ядерного топлива, Вт
ETA_REACTOR = 0.99           # КПД передачи теплоты от топлива рабочему телу в реакторе
ETA_PIPELINE = 0.99          # КПД паропровода/газопровода между узлами 0 и 1
N_AUX_OTHER_W = 0.0           # Прочие электрические собственные нужды, Вт; задать по проекту
X_XE = 0.283                 # Мольная доля Xe в He-Xe
T0_C = 1060.0                # Выход из реактора, °C
T1_C = 1030.0                # Вход в турбину, °C
T4_C = 35.0                  # Выход охладителя, °C
T6_C = 800.0                 # Вход в реактор, °C
ETA_TURBINE = 0.9267
ETA_COMPRESSOR = 0.8706
ETA_GENERATOR = 0.98          # КПД электрогенератора
ETA_TURBINE_MECH = 0.99      # КПД механической передачи от турбины к валу
ETA_COMPRESSOR_MECH = 0.99   # КПД механической передачи к компрессору
ETA_COMPRESSOR_DRIVE = 0.98  # КПД привода компрессора, как eta_drive в модели CO2
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
N_PINCH = 50
# Диапазоны перебора давлений (Па). P1 — вход турбины, P2 — выход турбины.
P1_GRID = np.arange(3.0, 8.01, 0.05) * 1e6
P2_GRID = np.arange(0.5, 6.01, 0.05) * 1e6
# При необходимости можно расширить диапазоны:
T6_GRID_C = np.array([650, 700, 750, 800, 850], dtype=float)
# DT_PINCH_GRID_C = np.arange(5, 16, 1)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "res_hexe"
OUT_DIR.mkdir(exist_ok=True)

# 2. Загрузка свойств
def load_property_tables():
    props_path = HERE / "props.npy"
    if not props_path.exists():
        raise FileNotFoundError("Не найден props.npy. Поместите файл свойств He-Xe в папку со скриптом.")
    values = np.load(props_path, allow_pickle=True)
    if len(values) != len(PROPERTY_NAMES):
        raise ValueError(f"В props.npy ожидалось {len(PROPERTY_NAMES)} интерполяторов, получено {len(values)}.")
    result = {FLUID_HEXE: dict(zip(PROPERTY_NAMES, values))}
    for f in result[FLUID_HEXE].values():
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
    value = PROPS[fluid][name]((x, p, t))
    return float(np.asarray(value))

def temperature_from_property(fluid: str,property_name: str,target_value: float, x: float,p: float,t_guess: float,) -> float:
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
    if not np.isfinite(q_cool) or q_cool <= 0:
        return np.nan
    cp_water = 4180.0       # Дж/(кг К)
    rho_water = 997.0       # кг/м3, приближение для воды около 25–40 °C
    delta_p = DP_WATER_FRAC * P_WATER
    m_water = q_cool / (cp_water * DT_WATER_C)
    volume_flow = m_water / rho_water
    return delta_p * volume_flow / (ETA_WATER_PUMP * ETA_WATER_MOTOR)

# 3. Расчёт одного цикла по методике
def calculate_cycle(p1: float, p2: float, dp_hot_frac: float = DP_HOT_FRAC,
                    dp_cold_frac: float = DP_COLD_FRAC, eta_t: float = ETA_TURBINE,
                    eta_c: float = ETA_COMPRESSOR,
                    dt_pinch_min_c: float = DT_PINCH_MIN_C,
                    t6_c: float = T6_C) -> dict:

    if p1 <= p2:
        return {"valid": False, "reason": "P1 должно быть больше P2"}
    if not (0 < dp_hot_frac < 1 and 0 <= dp_cold_frac < 1):
        return {"valid": False, "reason": "Некорректные относительные потери давления"}
    if not (0 < ETA_REACTOR <= 1):
        return {"valid": False, "reason": "КПД реактора должен быть в интервале (0; 1]"}

    dp_h = dp_hot_frac * p2
    dp_c = dp_cold_frac * p1
    p0 = p1
    p1_turbine = p1
    p3 = p2 - dp_h
    p4 = p3
    p5 = p1 + dp_c
    p6 = p1
    if min(p0, p1_turbine, p2, p3, p4, p5, p6) <= 0:
        return {"valid": False, "reason": "Получено неположительное давление"}

    t0 = T0_C + 273.15
    t4 = T4_C + 273.15
    t6 = t6_c + 273.15
    t1 = T1_C + 273.15
    h0 = prop(FLUID_HEXE, "H", X_XE, p0, t0)
    h1_fixed = prop(FLUID_HEXE, "H", X_XE, p1_turbine, t1)
    h6 = prop(FLUID_HEXE, "H", X_XE, p6, t6)
    delta_h_reactor = h0 - h6
    if not np.isfinite(delta_h_reactor) or delta_h_reactor <= 0:
        return {"valid": False, "reason": "h0-h6 должно быть положительным"}

    q_reactor_fuel = Q_REACTOR_NOM * ETA_REACTOR
    g = q_reactor_fuel / delta_h_reactor

    states: dict[int, State] = {}
    states[0] = State(0, t0, p0, G=g).calculate_properties()
    states[1] = State(1, t1, p1_turbine, G=g).calculate_properties()
    states[2], n_turb = turbomachine_outlet(states[1], 2, p2, eta_t, "turbine")
    states[4] = State(4, t4, p4, G=g).calculate_properties()
    states[5], n_comp_signed = turbomachine_outlet(states[4], 5, p5, eta_c, "compressor")
    n_comp_consumed = -n_comp_signed
    states[6] = State(6, t6, p6, G=g).calculate_properties()

    h3 = states[2].H - (states[6].H - states[5].H)
    states[3] = state_from_h(3, p3, h3, g, states[5].T)

    q_reactor_fluid = g * (states[0].H - states[6].H)
    q_reactor_loss = Q_REACTOR_NOM - q_reactor_fluid
    q_pipe_loss = g * (states[0].H - states[1].H)
    q_cycle_in = g * (states[1].H - states[6].H)
    q_cool = g * (states[3].H - states[4].H)
    q_recup_hot = g * (states[2].H - states[3].H)
    q_recup_cold = g * (states[6].H - states[5].H)
    recup_balance = q_recup_hot - q_recup_cold
    t_pinch = pinch_temperature(states)

    n_turb_shaft = n_turb * ETA_TURBINE_MECH
    n_turb_electric = n_turb_shaft * ETA_GENERATOR

    n_comp_shaft = n_comp_consumed / ETA_COMPRESSOR_MECH
    n_comp_electric = n_comp_shaft / ETA_COMPRESSOR_DRIVE

    n_turb_mech_loss = n_turb - n_turb_shaft
    n_generator_loss = n_turb_shaft - n_turb_electric
    n_comp_mech_loss = n_comp_shaft - n_comp_consumed
    n_comp_drive_loss = n_comp_electric - n_comp_shaft

    n_shaft_net = n_turb_shaft - n_comp_shaft
    n_electric_gross = n_turb_electric - n_comp_electric
    n_electromech_loss = n_turb_mech_loss + n_generator_loss + n_comp_mech_loss + n_comp_drive_loss

    n_water_pump = water_pump_power(q_cool, T4_C)
    n_aux_total = n_water_pump + N_AUX_OTHER_W if np.isfinite(n_water_pump) else np.nan
    n_electric_net = n_electric_gross - n_aux_total if np.isfinite(n_aux_total) else np.nan

    eta_el_gross = n_electric_gross / q_reactor_fluid if q_reactor_fluid > 0 else np.nan
    eta_el_net_cycle = (n_electric_net / q_reactor_fluid
                        if q_reactor_fluid > 0 and np.isfinite(n_electric_net) else np.nan)
    eta_el_net_cycle_Q2 = (n_electric_net / q_cycle_in
                           if q_cycle_in > 0 and np.isfinite(n_electric_net) else np.nan)
    eta_el_net_Q1 = eta_el_net_cycle
    eta_net_plant = (n_electric_net / Q_REACTOR_NOM
                     if np.isfinite(n_electric_net) else np.nan)
    eta_el_net = eta_el_net_cycle

    # Тепловой баланс цикла: QR = Q_pipe_loss + Q_cool + (N_T - N_K).
    n_net_fluid = n_turb - n_comp_consumed
    cycle_balance = q_cycle_in - q_cool - n_net_fluid
    # Баланс по мощности топлива с учётом потери в реакторе и трубопроводе.
    plant_thermal_balance = Q_REACTOR_NOM - q_reactor_loss - q_pipe_loss - q_cool - n_net_fluid

    valid = (np.isfinite(t_pinch) and t_pinch >= dt_pinch_min_c
             and q_cool >= 0 and q_recup_hot >= -1e-6 and q_recup_cold >= -1e-6
             and np.isfinite(eta_el_net) and np.isfinite(eta_net_plant)
             and n_electric_net > 0 and n_shaft_net > 0
             and abs(recup_balance) < max(1.0, 1e-5 * max(abs(q_recup_hot), abs(q_recup_cold), 1.0)))
    result = {"valid": bool(valid), "reason": "" if valid else "Не пройдены проверки pinch/тепловых потоков/баланса",
        "T6_C": t6_c, "P1_MPa": p1 / 1e6, "P2_MPa": p2 / 1e6,"dPh_Pa": dp_h, "dPc_Pa": dp_c, "P3_MPa": p3 / 1e6, "P4_MPa": p4 / 1e6,
        "P5_MPa": p5 / 1e6, "P6_MPa": p6 / 1e6, "G_kg_s": g,"T1_C": states[1].T - 273.15, "T2_C": states[2].T - 273.15,
        "T3_C": states[3].T - 273.15, "T5_C": states[5].T - 273.15,"T_pinch_C": t_pinch,
        "Q_R_nom_W": Q_REACTOR_NOM, "eta_R": ETA_REACTOR,"Q_R_W": q_reactor_fluid, "Q_R_loss_W": q_reactor_loss,
        "Q_pipe_loss_W": q_pipe_loss, "Q_cycle_in_W": q_cycle_in,"Q_cool_W": q_cool, "Q_recup_hot_W": q_recup_hot,
        "Q_recup_cold_W": q_recup_cold, "recup_balance_W": recup_balance,"N_T_i_W": n_turb, "N_K_i_consumed_W": n_comp_consumed,
        "N_T_e_W": n_turb_shaft, "N_T_el_W": n_turb_electric,"N_K_e_consumed_W": n_comp_shaft, "N_K_el_consumed_W": n_comp_electric,
        "N_T_mech_loss_W": n_turb_mech_loss, "N_K_mech_loss_W": n_comp_mech_loss,"N_generator_loss_W": n_generator_loss, "N_K_drive_loss_W": n_comp_drive_loss,
        "N_shaft_net_W": n_shaft_net, "N_electromech_loss_W": n_electromech_loss,"N_el_gross_W": n_electric_gross, "N_aux_pump_W": n_water_pump,
        "N_aux_other_W": N_AUX_OTHER_W, "N_aux_total_W": n_aux_total,"N_el_net_W": n_electric_net, "N_net_fluid_W": n_net_fluid,
        "eta_el_gross": eta_el_gross,"eta_el_net": eta_el_net,"eta_el_net_cycle": eta_el_net_cycle,
        "eta_el_net_cycle_Q2": eta_el_net_cycle_Q2,"eta_el_net_Q1": eta_el_net_Q1,"eta_net_plant": eta_net_plant,
        "cycle_balance_W": cycle_balance, "plant_thermal_balance_W": plant_thermal_balance,}
    # Обратная совместимость со старыми именами: eta_net теперь КПД всей установки.
    result["eta_gross"] = eta_el_gross
    result["eta_net"] = eta_net_plant
    result["N_electric_gross_W"] = n_electric_gross
    result["N_electric_net_W"] = n_electric_net
    for i, st in states.items():
        result[f"T{i}_K"] = st.T
        result[f"P{i}_Pa"] = st.P
        result[f"h{i}_J_kg"] = st.H
        result[f"s{i}_J_kgK"] = st.S
    return result
# 3.1. Формирование функциональной Excel-таблицы с фильтрами
COLUMN_LABELS = {
    "valid": "Допустимый режим","reason": "Причина недопуска","P1_MPa": "P1 — давление перед турбиной, МПа",
    "T6_C": "T6 — температура на входе в реактор, °C",
    "P2_MPa": "P2 — давление после турбины, МПа","P3_MPa": "P3 — давление после горячей стороны рекуператора, МПа","P4_MPa": "P4 — давление на входе компрессора, МПа",
    "P5_MPa": "P5 — давление после компрессора, МПа","P6_MPa": "P6 — давление перед реактором, МПа",
    "G_kg_s": "Массовый расход He–Xe, кг/с","T1_C": "T1 — температура перед турбиной, °C","T2_C": "T2 — температура после турбины, °C","T_pinch_C": "Минимальный температурный напор, °C",
    "Q_R_nom_W": "Номинальная тепловая мощность реактора, Вт",
    "Q_R_W": "Теплота, переданная He–Xe, Вт",
    "Q_R_loss_W": "Потери теплоты в реакторе, Вт",
    "Q_pipe_loss_W": "Потери теплоты в трубопроводе, Вт",
    "Q_cycle_in_W": "Теплота на входе в цикл, Вт",
    "Q_cool_W": "Теплота, отводимая в охладителе, Вт",
    "Q_recup_hot_W": "Теплота горячей стороны рекуператора, Вт",
    "Q_recup_cold_W": "Теплота холодной стороны рекуператора, Вт",
    "recup_balance_W": "Невязка рекуператора, Вт",
    "N_T_i_W": "Мощность турбины по рабочему телу, Вт",
    "N_K_i_consumed_W": "Потребление компрессора по рабочему телу, Вт",
    "N_T_e_W": "Мощность на валу турбины, Вт",
    "N_T_el_W": "Электрическая мощность турбины, Вт",
    "N_K_e_consumed_W": "Потребление компрессора на валу, Вт",
    "N_K_el_consumed_W": "Электрическое потребление компрессора, Вт",
    "N_T_mech_loss_W": "Механические потери турбины, Вт",
    "N_K_mech_loss_W": "Механические потери компрессора, Вт",
    "N_generator_loss_W": "Потери генератора, Вт",
    "N_K_drive_loss_W": "Потери привода компрессора, Вт",
    "N_shaft_net_W": "Мощность на валу нетто, Вт",
    "N_electromech_loss_W": "Электромеханические потери, Вт",
    "N_el_gross_W": "Электрическая мощность брутто, Вт",
    "N_aux_pump_W": "Потребление насоса, Вт",
    "N_aux_other_W": "Прочие собственные нужды, Вт",
    "N_aux_total_W": "Суммарные собственные нужды, Вт",
    "N_el_net_W": "Электрическая мощность нетто, Вт",
    "N_net_fluid_W": "Чистая мощность рабочего тела, Вт",
    "eta_el_gross": "КПД электрический брутто цикла, доля",
    "eta_el_net": "КПД электрический нетто цикла, доля",
    "eta_el_net_cycle": "КПД электрический нетто цикла, доля",
    "eta_el_net_cycle_Q2": "КПД нетто относительно Q2, доля",
    "eta_el_net_Q1": "КПД нетто относительно Q1, доля",
    "eta_net_plant": "КПД нетто всей установки, доля",
    "cycle_balance_W": "Невязка теплового баланса цикла, Вт",
    "plant_thermal_balance_W": "Невязка баланса установки, Вт",
}

FOCUS_COLUMNS = ["valid", "reason", "T6_C", "P1_MPa", "P2_MPa", "G_kg_s","T2_C", "T3_C", "T5_C", "T_pinch_C","N_T_i_W", "N_K_i_consumed_W", "N_el_gross_W", "N_el_net_W",
    "N_electromech_loss_W", "N_aux_pump_W","eta_el_gross", "eta_el_net_cycle", "eta_el_net_cycle_Q2", "eta_net_plant","Q_R_W", "Q_pipe_loss_W", "Q_cycle_in_W", "Q_cool_W",
    "cycle_balance_W", "plant_thermal_balance_W",]

def _prepare_excel_view(frame: pd.DataFrame, columns: list[str] | None = None,
                        efficiency_percent: bool = False) -> pd.DataFrame:
    available = [c for c in (columns or frame.columns.tolist()) if c in frame.columns]
    view = frame.loc[:, available].copy()
    view = view.rename(columns=COLUMN_LABELS)
    if efficiency_percent:
        for col in [
            "КПД электрический брутто цикла, доля",
            "КПД электрический нетто цикла, доля",
            "КПД нетто относительно Q2, доля",
            "КПД нетто относительно Q1, доля",
            "КПД нетто всей установки, доля",
        ]:
            if col in view.columns:
                view[col] = pd.to_numeric(view[col], errors="coerce") * 100.0
                view.rename(columns={col: col.replace(", доля", ", %")}, inplace=True)
    return view

def create_filterable_workbook(df: pd.DataFrame, output_path: Path) -> None:
    """Создаёт Excel-книгу с фильтруемыми таблицами и всеми рассчитанными режимами."""
    focus = _prepare_excel_view(df, FOCUS_COLUMNS, efficiency_percent=True)
    valid_source = df[df.get("valid", False) == True].copy()
    valid = _prepare_excel_view(valid_source, FOCUS_COLUMNS, efficiency_percent=True)
    full = _prepare_excel_view(df, efficiency_percent=False)

    if not valid_source.empty and "eta_net_plant" in valid_source:
        best_source = valid_source.loc[[valid_source["eta_net_plant"].idxmax()]]
        best = _prepare_excel_view(best_source, FOCUS_COLUMNS, efficiency_percent=True)
    else:
        best = pd.DataFrame(columns=focus.columns)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        focus.to_excel(writer, sheet_name="Фильтр — основные", index=False)
        valid.to_excel(writer, sheet_name="Только допустимые", index=False)
        full.to_excel(writer, sheet_name="Полная таблица", index=False)
        best.to_excel(writer, sheet_name="Оптимальный режим", index=False)

    wb = load_workbook(output_path)
    for index, ws in enumerate(wb.worksheets, start=1):
        ws.freeze_panes = "A2"
        ws.sheet_view.showGridLines = False
        ws.row_dimensions[1].height = 38
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF", size=10)
            cell.fill = PatternFill("solid", fgColor="244062")
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        if ws.max_row >= 2 and ws.max_column >= 1:
            table = Table(displayName=f"PressureTable{index}", ref=ws.dimensions)
            table.tableStyleInfo = TableStyleInfo(
                name="TableStyleMedium2", showFirstColumn=False,
                showLastColumn=False, showRowStripes=True, showColumnStripes=False
            )
            ws.add_table(table)
        else:
            ws.auto_filter.ref = ws.dimensions

        headers = {cell.value: cell.column for cell in ws[1]}
        for col_cells in ws.iter_cols(1, ws.max_column):
            header = str(col_cells[0].value or "")
            letter = get_column_letter(col_cells[0].column)
            sample = [str(c.value) for c in col_cells[1:min(ws.max_row, 60)] if c.value is not None]
            max_sample = max((len(v) for v in sample), default=0)
            if header == "Причина недопуска":
                width = 44
            elif len(header) > 38:
                width = 27
            else:
                width = min(max(12, len(header) * 0.85, max_sample * 0.85), 23)
            ws.column_dimensions[letter].width = width

        for header, col_idx in headers.items():
            if not header:
                continue
            for r in range(2, ws.max_row + 1):
                cell = ws.cell(r, col_idx)
                if "КПД" in str(header) and "%" in str(header):
                    cell.number_format = "0.000"
                elif "МПа" in str(header):
                    cell.number_format = "0.000"
                elif "кг/с" in str(header):
                    cell.number_format = "0.00000"
                elif "°C" in str(header):
                    cell.number_format = "0.00"
                elif "КПД" in str(header) and "доля" in str(header):
                    cell.number_format = "0.0000"
                elif "Вт" in str(header):
                    cell.number_format = '#,##0.000;[Red]-#,##0.000'

        for target in [
            "КПД нетто всей установки, %",
            "КПД электрический нетто цикла, %",
            "Минимальный температурный напор, °C",
        ]:
            if target in headers and ws.max_row > 2:
                letter = get_column_letter(headers[target])
                ws.conditional_formatting.add(
                    f"{letter}2:{letter}{ws.max_row}",
                    ColorScaleRule(
                        start_type="min", start_color="F8696B",
                        mid_type="percentile", mid_value=50, mid_color="FFEB84",
                        end_type="max", end_color="63BE7B",
                    ),
                )

    intro = wb.create_sheet("Как пользоваться", 0)
    intro.sheet_view.showGridLines = False
    intro["A1"] = "Таблица режимов He–Xe — фильтрация и сравнение"
    intro["A1"].font = Font(bold=True, size=16, color="FFFFFF")
    intro["A1"].fill = PatternFill("solid", fgColor="244062")
    intro["A1"].alignment = Alignment(vertical="center", wrap_text=True)
    intro.row_dimensions[1].height = 34
    instructions = [
        ("Назначение", "Книга автоматически создаётся после перебора температуры T6 и давлений P1/P2."),
        ("Фильтрация", "Нажмите стрелку в заголовке столбца для выбора значений, диапазонов и условий фильтра."),
        ("Фильтр — основные", "Ключевые параметры, мощности, КПД и балансы для быстрого сравнения режимов."),
        ("Только допустимые", "Только режимы, прошедшие критерии модели."),
        ("Полная таблица", "Все расчётные режимы и исходные столбцы модели."),
        ("Оптимальный режим", "Строка с максимальным КПД нетто всей установки среди допустимых режимов по всем значениям T6."),
        ("КПД", "На основном листе КПД показаны в процентах; на листе «Полная таблица» сохранены исходные доли."),
    ]
    for r, (label, value) in enumerate(instructions, start=3):
        intro.cell(r, 1, label)
        intro.cell(r, 2, value)
        intro.cell(r, 1).font = Font(bold=True, color="244062")
        intro.cell(r, 1).fill = PatternFill("solid", fgColor="D9EAF7")
        intro.cell(r, 1).alignment = Alignment(vertical="top", wrap_text=True)
        intro.cell(r, 2).alignment = Alignment(vertical="top", wrap_text=True)
        intro.row_dimensions[r].height = 36
    intro.column_dimensions["A"].width = 25
    intro.column_dimensions["B"].width = 100
    intro.freeze_panes = "A3"
    wb.save(output_path)
    print(f"Функциональная таблица с фильтрами сохранена: {output_path.resolve()}")


# 4. Перебор T6/P1/P2 и сохранение результатов
def run_pressure_sweep() -> pd.DataFrame:
    rows = []
    total = len(T6_GRID_C) * len(P1_GRID) * len(P2_GRID)
    for k, (t6_c, p1, p2) in enumerate(product(T6_GRID_C, P1_GRID, P2_GRID), start=1):
        try:
            result = calculate_cycle(p1, p2, t6_c=float(t6_c))
        except Exception as exc:
            result = {
                "valid": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "T6_C": float(t6_c),
                "P1_MPa": p1 / 1e6,
                "P2_MPa": p2 / 1e6,}
        rows.append(result)

        if k % 100 == 0 or k == total:
            print(f"Рассчитано {k}/{total}")
    df = pd.DataFrame(rows)
    output = OUT_DIR / "pressure_sweep.xlsx"
    create_filterable_workbook(df, output)
    df.to_csv(OUT_DIR / "pressure_sweep_full.csv", index=False, encoding="utf-8-sig")

    valid_df = df[df["valid"] == True].copy()
    if valid_df.empty:
        print("Допустимых режимов нет. Проверьте диапазоны давлений, температуры, свойства и ограничения.")
        return df

    best_idx = valid_df["eta_net_plant"].idxmax()
    best = valid_df.loc[best_idx]
    best_df = pd.DataFrame([best])
    best_path = OUT_DIR / "best_mode.xlsx"
    best_df.to_excel(best_path, index=False)

    best_per_t6 = valid_df.loc[valid_df.groupby("T6_C")["eta_net_plant"].idxmax()].sort_values("T6_C")
    best_per_t6_path = OUT_DIR / "best_mode_by_T6.xlsx"
    best_per_t6.to_excel(best_per_t6_path, index=False)

    print("\nОптимальный допустимый режим по всем значениям T6:")
    print(f"T6 = {best['T6_C']:.1f} °C")
    print(f"P1 = {best['P1_MPa']:.3f} МПа")
    print(f"P2 = {best['P2_MPa']:.3f} МПа")
    print(f"G = {best['G_kg_s']:.5f} кг/с")
    print(f"T2 = {best['T2_C']:.2f} °C")
    print(f"T3 = {best['T3_C']:.2f} °C")
    print(f"T5 = {best['T5_C']:.2f} °C")
    print(f"T_pinch = {best['T_pinch_C']:.2f} °C")
    print(f"N_T_i (мощность турбины по рабочему телу) = {best['N_T_i_W']/1e6:.4f} МВт")
    print(f"N_K_i (потребление компрессора по рабочему телу) = {best['N_K_i_consumed_W']/1e6:.4f} МВт")
    print(f"N_el_gross = {best['N_el_gross_W']/1e6:.4f} МВт")
    print(f"N_el_net = {best['N_el_net_W']/1e6:.4f} МВт")
    print(f"Электромеханические потери = {best['N_electromech_loss_W']/1e3:.3f} кВт")
    print(f"КПД электрический брутто цикла, N_el_gross/Q_R = {best['eta_el_gross']*100:.3f} %")
    print(f"КПД электрический нетто цикла, N_el_net/Q_R = {best['eta_el_net_cycle']*100:.3f} %")
    print(f"Дополнительно: N_el_net/Q2 = {best['eta_el_net_cycle_Q2']*100:.3f} %")
    print(f"КПД нетто всей установки, N_el_net/Q_R_nom = {best['eta_net_plant']*100:.3f} %")
    print(f"Теплота в охладителе = {best['Q_cool_W']/1e6:.4f} МВт")
    print(f"Потребление насоса = {best['N_aux_pump_W']/1e3:.3f} кВт")
    print(f"Тепловой баланс цикла = {best['cycle_balance_W']:.3f} Вт")
    print(f"Баланс установки = {best['plant_thermal_balance_W']:.3f} Вт")
    print(f"Лучший режим сохранён: {best_path.resolve()}")
    print(f"Оптимальные режимы для каждого T6 сохранены: {best_per_t6_path.resolve()}")

    for t6_c, group in valid_df.groupby("T6_C"):
        make_pressure_plot(group, suffix=f"_T6_{int(t6_c)}C")

    make_energy_balance_plot(best)
    return df


def make_pressure_plot(df: pd.DataFrame, suffix: str = ""):
    """Строит две карты: электрический КПД нетто цикла и всей установки для одного T6."""
    maps = [("eta_el_net_cycle", "КПД электрический нетто цикла He–Xe (N_el_net / Q_R)", f"eta_el_net_pressure_map{suffix}.png"),
        ("eta_net_plant", "КПД электрический нетто всей установки He–Xe (N_el_net / Q_R_nom)", f"eta_net_plant_pressure_map{suffix}.png"),]
    t6_label = f", T6={df['T6_C'].iloc[0]:.0f} °C" if "T6_C" in df.columns and not df.empty else ""
    for column, title, filename in maps:
        pivot = df.pivot_table(index="P1_MPa", columns="P2_MPa", values=column, aggfunc="max")
        if pivot.empty or not np.isfinite(pivot.to_numpy(dtype=float)).any():
            continue
        x = pivot.columns.to_numpy(dtype=float)
        y = pivot.index.to_numpy(dtype=float)
        z = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
        fig, ax = plt.subplots(figsize=(9, 6.5))
        cf = ax.contourf(x, y, z, levels=20)
        ax.set_xlabel("P2 — давление после турбины, МПа")
        ax.set_ylabel("P1 — давление перед турбиной, МПа")
        ax.set_title(title + t6_label)
        fig.colorbar(cf, ax=ax, label="КПД, доля")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = OUT_DIR / filename
        fig.savefig(path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"Карта сохранена: {path.resolve()}")


def make_energy_balance_plot(best: pd.Series) -> None:
    labels = [
        "Тепловая мощность топлива QR_nom",
        "Потери в реакторе",
        "Теплота передана He–Xe, QR",
        "Потери в трубопроводе",
        "Сброс теплоты в охладителе",
        "Внутренняя чистая мощность цикла",
        "Электромеханические потери",
        "Электрическая мощность брутто",
        "Насос + прочие собственные нужды",
        "Электрическая мощность нетто",]
    vals_mw = [best["Q_R_nom_W"] / 1e6,-best["Q_R_loss_W"] / 1e6, best["Q_R_W"] / 1e6,-best["Q_pipe_loss_W"] / 1e6,-best["Q_cool_W"] / 1e6,
        best["N_net_fluid_W"] / 1e6,-best["N_electromech_loss_W"] / 1e6,best["N_el_gross_W"] / 1e6,-best["N_aux_total_W"] / 1e6,best["N_el_net_W"] / 1e6,]
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(labels))
    ax.barh(y, vals_mw)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Мощность, МВт (знак минус — потери/отвод/потребление)")
    ax.set_title(f"Колесо баланса мощностей, P1={best['P1_MPa']:.2f} МПа, P2={best['P2_MPa']:.2f} МПа")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "energy_balance_best.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Таблица энергетического баланса оптимального режима в отдельный CSV.
    balance = pd.DataFrame({"Показатель": labels, "Мощность_МВт": vals_mw})
    balance.to_csv(OUT_DIR / "energy_balance_best.csv", index=False, encoding="utf-8-sig")

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

