import numpy as np
from CoolProp.CoolProp import PropsSI as prop
from data import nodes, blocks
from modules import Node, comp, turb, heat_source, heat_exch, split, mix
from scipy.optimize import root
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from datetime import datetime
import itertools
import multiprocessing as mp
import time
from tqdm import tqdm


# ФИКСИРОВАННАЯ ТОЧКА
P1_FIXED = 31.0e6
P7_FIXED = 8.5e6
X_FIXED = 0.7

TR1 = 730.0 + 273.15
TR2 = 550.0 + 273.15
T2_set = 715.0 + 273.15
T7 = 35.0 + 273.15
Q_HEAT = 2.75e6

KPDcomp_BASE = 0.80
KPDturb_BASE = 0.90
KPD_reactor = 0.99
KPD_RTO = 0.99

eta_turb_mech = 0.99
eta_generator = 0.99
eta_mc_mech = 0.97
eta_rc_mech = 0.97
eta_drive = 0.98

dTh_BASE = 5.0
dTl_BASE = 5.0
dP_HTHE_cold_BASE = 0.02
dP_HTHE_hot_BASE = 0.03
dP_LTHE_cold_BASE = 0.02
dP_LTHE_hot_BASE = 0.03
dP_COOL = 0.03

P_Na = 0.16e6
P_water = 0.50e6
dP_Na = 0.05
dP_water = 0.10
KPD_pump = 0.80
KPD_motor = 0.95
DT_water = 10.0
rho_Na = 850.0
cp_Na = 1270.0
rho_water = 1000.0
cp_water = 4180.0
N_SN_reactor = 0.0
eta_tr1 = 0.99
eta_tr2 = 0.99


# ТЕПЛОВЫЕ ПОТОКИ ДО ЦИКЛА CO2

Q_reactor = Q_HEAT * KPD_reactor
Q_tr1 = Q_reactor * eta_tr1
Q_CO2_RTO = Q_tr1 * KPD_RTO

Q_loss_reactor = Q_HEAT - Q_reactor
Q_loss_tr1 = Q_reactor - Q_tr1
Q_loss_RTO = Q_tr1 - Q_CO2_RTO

# ============================================================
# ДИАПАЗОНЫ
# ============================================================
ETA_COMP_RANGE = np.arange(70.0, 85.0 + 0.1, 1.0) / 100.0
ETA_TURB_RANGE = np.arange(75.0, 95.0 + 0.1, 1.0) / 100.0
DTH_RANGE = np.arange(5.0, 15.0 + 0.1, 0.5)
DTL_RANGE = np.arange(5.0, 15.0 + 0.1, 0.5)
DP_COLD_RANGE = np.arange(1.0, 15.0 + 0.01, 1.5) / 100.0
DP_HOT_RANGE = np.arange(1.0, 15.0 + 0.01, 1.5) / 100.0

ROOT_TOL = 1e-4

ROOT_GUESSES = [
    np.array([TR2 - 5.0, 450.0, 350.0, 300.0]),
    np.array([TR2 - 20.0, 500.0, 300.0, 250.0]),
    np.array([TR2 - 120.0, 400.0, 350.0, 300.0]),
]

def node_value(nodes_at, number, field):
    try:
        value = nodes_at.loc[str(number), field]
        if value is None:
            return np.nan
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan

def block_value(blocks_at, name, field):
    try:
        value = blocks_at.loc[name, field]
        if value is None:
            return np.nan
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan

def reset_node_pressure(nodes_at, number, pressure):
    T = node_value(nodes_at, number, "T")
    G = node_value(nodes_at, number, "G")
    if np.isfinite(T) and np.isfinite(G):
        Node(str(number), G=G, P=pressure, T=T, fluid="CO2")

def get_node_h(nodes_at, number):
    T = node_value(nodes_at, number, "T")
    P = node_value(nodes_at, number, "P")
    if not (np.isfinite(T) and np.isfinite(P)):
        return np.nan
    try:
        return prop("H", "T", T, "P", P, "CO2")
    except Exception:
        return np.nan

def clear_state():
    nodes.loc[:, :] = np.nan
    blocks.loc[:, :] = np.nan

def compute_point(eta_comp, eta_turb, dTh, dTl, dP_hot, dP_cold):
    P1 = P1_FIXED
    P7 = P7_FIXED
    x = X_FIXED
    nodes_at = nodes
    blocks_at = blocks
    clear_state()

    # Согласование давлений (как в главном коде)
    P12_target = P1 / (1.0 - dP_cold)
    P9_target = P12_target
    P8_target = P9_target / (1.0 - dP_cold)
    P7_target = P7
    P6_target = P7_target / (1.0 - dP_COOL)
    P5_target = P6_target
    P4_target = P5_target / (1.0 - dP_hot)
    P3_target = P4_target / (1.0 - dP_hot)

    result = {
        "P1_MPa": P1 / 1e6, "P7_MPa": P7 / 1e6, "x": x,
        "TR1_C": TR1 - 273.15, "TR2_C": TR2 - 273.15,
        "T2_set_C": T2_set - 273.15, "T7_C": T7 - 273.15,
        "KPDcomp_%": eta_comp * 100.0, "KPDturb_%": eta_turb * 100.0,
        "dT_HTHE_K": dTh, "dT_LTHE_K": dTl,
        "dP_hot_%": dP_hot * 100.0, "dP_cold_%": dP_cold * 100.0,
        "T1_C": np.nan, "T12_C": np.nan, "T5_C": np.nan, "T8_C": np.nan,
        "N_TURB_i_MW": np.nan, "N_TURB_el_MW": np.nan,
        "N_MCOMP_i_MW": np.nan, "N_MCOMP_el_MW": np.nan,
        "N_RCOMP_i_MW": np.nan, "N_RCOMP_el_MW": np.nan,
        "N_net_cycle_MW": np.nan, "N_el_gross_MW": np.nan, "N_el_net_MW": np.nan,
        "G_CO2_kg_s": np.nan, "G_Na_kg_s": np.nan, "G_water_kg_s": np.nan,
        "N_Na_pump_MW": np.nan, "N_water_pump_MW": np.nan,
        "N_pumps_MW": np.nan, "N_SN_MW": np.nan,
        "Q_HEAT_MW": np.nan, "Q_COOL_MW": np.nan, "Q_COOL_eff_MW": np.nan,
        "Q_loss_reactor_MW": Q_loss_reactor / 1e6,
        "Q_loss_tr1_MW": Q_loss_tr1 / 1e6,
        "Q_loss_RTO_MW": Q_loss_RTO / 1e6,
        "Q_transport_loss_CO2_MW": np.nan,
        "Q_EM_TURB_MW": np.nan, "Q_EM_MCOMP_MW": np.nan,
        "Q_EM_RCOMP_MW": np.nan, "Q_EM_pumps_MW": np.nan,
        "Q_EM_ASMM_MW": np.nan, "Q_recup_MW": np.nan,
        "Q_balance_out_MW": np.nan, "Q_balance_error_MW": np.nan,
        "eta_cycle_%": np.nan, "eta_el_gross_%": np.nan, "eta_el_net_%": np.nan,
        "TR2_minus_T1_K": np.nan,
        "P3_MPa": P3_target / 1e6, "P4_MPa": P4_target / 1e6,
        "P5_MPa": P5_target / 1e6, "P6_MPa": P6_target / 1e6,
        "P8_MPa": P8_target / 1e6, "P9_MPa": P9_target / 1e6,
        "P11_MPa": P12_target / 1e6, "P12_MPa": P12_target / 1e6,
        "root_residual_max": np.nan, "root_success": False,
        "physical_root": False, "message": "",
    }

    if P1 <= P3_target:
        result["message"] = f"Нет расширения: P1={P1/1e6:.3f} МПа, P3={P3_target/1e6:.3f} МПа"
        return result

    dT_Na = TR1 - TR2
    if dT_Na <= 0.0:
        result["message"] = "TR1 <= TR2"
        return result
    G_Na = Q_reactor / (cp_Na * dT_Na)
    if not np.isfinite(G_Na) or G_Na <= 0.0:
        result["message"] = "G_Na <= 0"
        return result

    def calc(v):
        T1, T12, T5, T8 = v
        try:
            H1 = prop("H", "T", T1, "P", P1, "CO2")
            T2_before_transport = T2_set
            H2_before_transport = prop("H", "T", T2_before_transport, "P", P1, "CO2")
            dH_heat = H2_before_transport - H1
            if not np.isfinite(dH_heat) or dH_heat <= 0.0:
                return np.full(4, 1e12)
            G1 = Q_CO2_RTO / dH_heat
            if not np.isfinite(G1) or G1 <= 0.0:
                return np.full(4, 1e12)

            Node("1", G=G1, P=P1, T=T1, fluid="CO2")
            heat_exch("HEAT", "1", "2", "R1", "R2", Q=Q_CO2_RTO, T21=TR1, T22=TR2,
                      hot_no_cp=True, fluid2="Na", G2=G_Na, P21=P_Na, P22=P_Na * (1.0 - dP_Na))

            H2_after_transport = H1 + eta_tr1 * dH_heat
            T2_after_transport = prop("T", "H", H2_after_transport, "P", P1, "CO2")
            Q_CO2_to_turbine = G1 * (H2_after_transport - H1)
            Q_transport_loss_CO2 = G1 * (H2_before_transport - H2_after_transport)

            if not np.isfinite(T2_after_transport):
                return np.full(4, 1e12)

            Node("2", G=G1, P=P1, T=T2_after_transport, fluid="CO2")
            turb("TURB", "2", "3", P3_target, eta_turb)

            Node("12", G=G1, P=P12_target, T=T12, fluid="CO2")
            heat_exch("HTHE", "3", "4", "12", "1", T22=T1,
                      dP1=dP_hot, dP2=dP_cold)
            reset_node_pressure(nodes_at, 4, P4_target)
            reset_node_pressure(nodes_at, 1, P1)

            Node("5", G=G1, P=P5_target, T=T5, fluid="CO2")
            Node("8", G=G1 * x, P=P8_target, T=T8, fluid="CO2")
            heat_exch("LTHE", "4", "5", "8", "9", T12=T5,
                      dP1=dP_hot, dP2=dP_cold)
            reset_node_pressure(nodes_at, 5, P5_target)
            reset_node_pressure(nodes_at, 8, P8_target)
            reset_node_pressure(nodes_at, 9, P9_target)
            split("SPLIT", "5", "6", "10", x)
            reset_node_pressure(nodes_at, 6, P6_target)
            reset_node_pressure(nodes_at, 10, P5_target)
            heat_source("COOL", "6", "7", T=T7)
            reset_node_pressure(nodes_at, 7, P7_target)
            comp("MCOMP", "7", "8", P8_target, eta_comp)
            comp("ACOMP", "10", "11", P12_target, eta_comp)
            reset_node_pressure(nodes_at, 8, P8_target)
            reset_node_pressure(nodes_at, 11, P12_target)
            reset_node_pressure(nodes_at, 9, P12_target)
            mix("MIX", "9", "11", "12")
            reset_node_pressure(nodes_at, 12, P12_target)

            eq1 = block_value(blocks_at, "HTHE", "dT") - dTh
            eq2 = node_value(nodes_at, 12, "T") - T12
            eq3 = node_value(nodes_at, 8, "T") - T8
            eq4 = block_value(blocks_at, "LTHE", "dT") - dTl
            residual = np.array([eq1, eq2, eq3, eq4], dtype=float)
            if not np.all(np.isfinite(residual)):
                return np.full(4, 1e12)
            return residual
        except Exception:
            return np.full(4, 1e12)

    best_solution = None
    best_norm = np.inf

    for guess in ROOT_GUESSES:
        try:
            solution = root(calc, x0=guess, method="hybr", options={"xtol": 1e-6, "maxfev": 150})
            if not np.all(np.isfinite(solution.x)):
                continue
            residual = calc(solution.x)
            if not np.all(np.isfinite(residual)):
                continue
            norm = np.max(np.abs(residual))
            if norm < best_norm:
                best_solution = solution
                best_norm = norm
            if not solution.success or norm >= ROOT_TOL:
                continue
            T1, T12, T5, T8 = solution.x
            if T1 >= TR2:
                continue
            if not all(np.isfinite(value) for value in [T1, T12, T5, T8]):
                continue
            final_residual = calc(solution.x)
            final_norm = np.max(np.abs(final_residual))
            if final_norm >= ROOT_TOL:
                continue

            H1 = prop("H", "T", T1, "P", P1, "CO2")
            H2_before_transport = prop("H", "T", T2_set, "P", P1, "CO2")
            dH_heat = H2_before_transport - H1
            G1 = Q_CO2_RTO / dH_heat

            H2_after_transport = H1 + eta_tr1 * dH_heat
            T2_after_transport = prop("T", "H", H2_after_transport, "P", P1, "CO2")
            Q_CO2_to_turbine = G1 * (H2_after_transport - H1)
            Q_transport_loss_CO2 = G1 * (H2_before_transport - H2_after_transport)

            # --- Считываем все энтальпии ---
            H2 = get_node_h(nodes_at, 2)
            H3 = get_node_h(nodes_at, 3)
            H4 = get_node_h(nodes_at, 4)
            H5 = get_node_h(nodes_at, 5)
            H6 = get_node_h(nodes_at, 6)
            H7 = get_node_h(nodes_at, 7)
            H8 = get_node_h(nodes_at, 8)
            H9 = get_node_h(nodes_at, 9)
            H10 = get_node_h(nodes_at, 10)
            H11 = get_node_h(nodes_at, 11)
            H12 = get_node_h(nodes_at, 12)
            G2 = node_value(nodes_at, 2, "G")
            G6 = node_value(nodes_at, 6, "G")
            G7 = node_value(nodes_at, 7, "G")
            G8 = node_value(nodes_at, 8, "G")
            G10 = node_value(nodes_at, 10, "G")

            required = [H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12,
                        G2, G6, G7, G8, G10]
            if not all(np.isfinite(v) for v in required):
                result["message"] = "Некорректные энтальпии/расходы"
                return result

            # --- Мощности (как в главном коде) ---
            N_TURB_i = G2 * (H2 - H3)
            N_TURB_e = N_TURB_i * eta_turb_mech
            N_TURB_el = N_TURB_e * eta_generator

            N_RCOMP_i = G7 * (H8 - H7)
            N_RCOMP_e = N_RCOMP_i / eta_rc_mech
            N_RCOMP_el = N_RCOMP_e / eta_drive

            N_MCOMP_i = G10 * (H11 - H10)
            N_MCOMP_e = N_MCOMP_i / eta_mc_mech
            N_MCOMP_el = N_MCOMP_e / eta_drive

            # --- ЭМ-потери по элементам ---
            Q_EM_TURB = N_TURB_i - N_TURB_el
            Q_EM_MCOMP = N_MCOMP_el - N_MCOMP_i
            Q_EM_RCOMP = N_RCOMP_el - N_RCOMP_i
            Q_EM_COMP = Q_EM_MCOMP + Q_EM_RCOMP
            Q_EM_ASMM = Q_EM_TURB + Q_EM_COMP

            # --- Q_COOL ---
            Q_COOL = G6 * (H6 - H7)
            if not np.isfinite(Q_COOL) or Q_COOL <= 0.0:
                result["message"] = "Q_COOL <= 0"
                return result

            # --- Теплообменники: невязка ---
            Q_HTHE_hot = G2 * (H3 - H4)
            Q_HTHE_cold = G1 * (H1 - H12)
            Q_LTHE_hot = G1 * (H4 - H5)
            Q_LTHE_cold = G8 * (H9 - H8)
            Q_recup = (Q_HTHE_hot - Q_HTHE_cold) + (Q_LTHE_hot - Q_LTHE_cold)

            Q_COOL_eff = Q_COOL + Q_recup
            if not np.isfinite(Q_COOL_eff) or Q_COOL_eff <= 0.0:
                result["message"] = "Q_COOL_eff <= 0"
                return result

            # --- Насосы ---
            deltaP_Na = P_Na * dP_Na
            N_Na_pump_hyd = (G_Na / rho_Na) * deltaP_Na
            N_Na_pump_el = N_Na_pump_hyd / (KPD_pump * KPD_motor)

            G_water = Q_COOL / (cp_water * DT_water)
            if not np.isfinite(G_water) or G_water <= 0.0:
                result["message"] = "G_water <= 0"
                return result
            deltaP_water = P_water * dP_water
            N_water_pump_hyd = (G_water / rho_water) * deltaP_water
            N_water_pump_el = N_water_pump_hyd / (KPD_pump * KPD_motor)

            N_pumps = N_Na_pump_el + N_water_pump_el
            Q_EM_pumps = N_pumps
            N_SN = N_pumps + N_SN_reactor

            # --- Электрические мощности ---
            N_el_gross = N_TURB_el - N_MCOMP_el - N_RCOMP_el
            N_el_net = N_el_gross - N_SN
            N_net_cycle = N_TURB_i - N_MCOMP_i - N_RCOMP_i

            eta_cycle = N_net_cycle / Q_HEAT
            eta_el_gross = N_el_gross / Q_HEAT
            eta_el_net = N_el_net / Q_HEAT

            # --- Баланс (как в главном коде) ---
            Q_balance_out = (Q_loss_reactor + Q_loss_tr1 + Q_loss_RTO
                             + Q_transport_loss_CO2
                             + Q_EM_TURB + Q_EM_MCOMP + Q_EM_RCOMP + Q_EM_pumps
                             + Q_COOL_eff
                             + N_el_net)
            Q_balance_error = Q_HEAT - Q_balance_out

            # --- Проверка физичности ---
            power_values = [N_TURB_el, N_MCOMP_el, N_RCOMP_el,
                            N_Na_pump_el, N_water_pump_el,
                            N_pumps, N_SN, N_el_gross, N_el_net,
                            eta_cycle, eta_el_gross, eta_el_net,
                            Q_transport_loss_CO2, Q_CO2_to_turbine,
                            Q_balance_out, Q_balance_error,
                            Q_HTHE_hot, Q_HTHE_cold, Q_LTHE_hot, Q_LTHE_cold,
                            Q_recup, Q_COOL_eff,
                            Q_EM_TURB, Q_EM_MCOMP, Q_EM_RCOMP, Q_EM_pumps]
            if not all(np.isfinite(value) for value in power_values):
                result["message"] = "Не все мощности конечны"
                return result
            if N_TURB_el <= 0.0:
                result["message"] = "N_TURB_el <= 0"
                return result
            if N_MCOMP_el < 0.0 or N_RCOMP_el < 0.0:
                result["message"] = "N_comp_el < 0"
                return result
            if N_pumps < 0.0:
                result["message"] = "N_pumps < 0"
                return result
            if N_el_net <= 0.0:
                result["message"] = "N_el_net <= 0"
                return result
            if eta_el_net <= 0.0:
                result["message"] = "eta_el_net <= 0"
                return result

            # --- Успех ---
            result["physical_root"] = True
            result["root_success"] = True
            result["T1_C"] = T1 - 273.15
            result["T12_C"] = T12 - 273.15
            result["T5_C"] = T5 - 273.15
            result["T8_C"] = T8 - 273.15
            result["TR2_minus_T1_K"] = TR2 - T1

            result["N_TURB_i_MW"] = N_TURB_i / 1e6
            result["N_TURB_el_MW"] = N_TURB_el / 1e6
            result["N_MCOMP_i_MW"] = N_MCOMP_i / 1e6
            result["N_MCOMP_el_MW"] = N_MCOMP_el / 1e6
            result["N_RCOMP_i_MW"] = N_RCOMP_i / 1e6
            result["N_RCOMP_el_MW"] = N_RCOMP_el / 1e6
            result["N_net_cycle_MW"] = N_net_cycle / 1e6
            result["N_el_gross_MW"] = N_el_gross / 1e6
            result["N_el_net_MW"] = N_el_net / 1e6

            result["G_CO2_kg_s"] = G1
            result["G_Na_kg_s"] = G_Na
            result["G_water_kg_s"] = G_water

            result["N_Na_pump_MW"] = N_Na_pump_el / 1e6
            result["N_water_pump_MW"] = N_water_pump_el / 1e6
            result["N_pumps_MW"] = N_pumps / 1e6
            result["N_SN_MW"] = N_SN / 1e6

            result["Q_HEAT_MW"] = Q_HEAT / 1e6
            result["Q_COOL_MW"] = Q_COOL / 1e6
            result["Q_COOL_eff_MW"] = Q_COOL_eff / 1e6

            result["Q_transport_loss_CO2_MW"] = Q_transport_loss_CO2 / 1e6

            result["Q_EM_TURB_MW"] = Q_EM_TURB / 1e6
            result["Q_EM_MCOMP_MW"] = Q_EM_MCOMP / 1e6
            result["Q_EM_RCOMP_MW"] = Q_EM_RCOMP / 1e6
            result["Q_EM_pumps_MW"] = Q_EM_pumps / 1e6
            result["Q_EM_ASMM_MW"] = Q_EM_ASMM / 1e6
            result["Q_recup_MW"] = Q_recup / 1e6

            result["Q_balance_out_MW"] = Q_balance_out / 1e6
            result["Q_balance_error_MW"] = Q_balance_error / 1e6

            result["eta_cycle_%"] = eta_cycle * 100.0
            result["eta_el_gross_%"] = eta_el_gross * 100.0
            result["eta_el_net_%"] = eta_el_net * 100.0

            result["message"] = "OK"
            return result
        except Exception as e:
            result["message"] = f"Ошибка: {type(e).__name__}: {e}"
            return result

    # Если ни один guess не дал решения
    result["root_residual_max"] = best_norm
    result["message"] = f"Корень не найден; min residual={best_norm:.3e}"
    return result


# ============================================================
# WORKER
# ============================================================
def worker(task):
    idx, sweep_name, params = task
    try:
        result = compute_point(
            eta_comp=params["eta_comp"],
            eta_turb=params["eta_turb"],
            dTh=params["dTh"],
            dTl=params["dTl"],
            dP_hot=params["dP_hot"],
            dP_cold=params["dP_cold"],
        )
        result["sweep"] = sweep_name
        result["var1"] = params.get("var1", np.nan)
        result["var2"] = params.get("var2", np.nan)
        return idx, result
    except Exception as e:
        return idx, {
            "sweep": sweep_name,
            "var1": params.get("var1", np.nan),
            "var2": params.get("var2", np.nan),
            "P1_MPa": P1_FIXED / 1e6,
            "P7_MPa": P7_FIXED / 1e6,
            "x": X_FIXED,
            "physical_root": False,
            "root_success": False,
            "message": f"Worker error: {type(e).__name__}: {e}",
        }


# ============================================================
# ФОРМИРОВАНИЕ ЗАДАЧ
# ============================================================
def make_tasks():
    tasks = []
    idx = 0

    # --- Группа 1: КПД машин ---
    start_eff = idx
    for eta_comp, eta_turb in itertools.product(ETA_COMP_RANGE, ETA_TURB_RANGE):
        tasks.append((
            idx, "efficiency",
            {
                "eta_comp": eta_comp, "eta_turb": eta_turb,
                "dTh": dTh_BASE, "dTl": dTl_BASE,
                "dP_hot": dP_HTHE_hot_BASE, "dP_cold": dP_HTHE_cold_BASE,
                "var1": eta_comp, "var2": eta_turb,
            },
        ))
        idx += 1
    n_eff = idx - start_eff

    # --- Группа 2: Недогревы ---
    start_pinch = idx
    for dTh, dTl in itertools.product(DTH_RANGE, DTL_RANGE):
        tasks.append((
            idx, "pinch",
            {
                "eta_comp": KPDcomp_BASE, "eta_turb": KPDturb_BASE,
                "dTh": dTh, "dTl": dTl,
                "dP_hot": dP_HTHE_hot_BASE, "dP_cold": dP_HTHE_cold_BASE,
                "var1": dTh, "var2": dTl,
            },
        ))
        idx += 1
    n_pinch = idx - start_pinch

    # --- Группа 3: Потери давления ---
    start_press = idx
    for dP_hot, dP_cold in itertools.product(DP_HOT_RANGE, DP_COLD_RANGE):
        tasks.append((
            idx, "pressure",
            {
                "eta_comp": KPDcomp_BASE, "eta_turb": KPDturb_BASE,
                "dTh": dTh_BASE, "dTl": dTl_BASE,
                "dP_hot": dP_hot, "dP_cold": dP_cold,
                "var1": dP_hot, "var2": dP_cold,
            },
        ))
        idx += 1
    n_press = idx - start_press

    print()
    print("Количество точек:")
    print(f"  КПД компрессора/турбины : {n_eff}")
    print(f"  Недогревы ВТО/НТО       : {n_pinch}")
    print(f"  Потери давления         : {n_press}")
    print(f"  Всего                   : {len(tasks)}")
    print()

    return tasks, n_eff, n_pinch, n_press


# ============================================================
# СОХРАНЕНИЕ В EXCEL
# ============================================================
def write_sheet(ws, results):
    if not results:
        ws.append(["Нет результатов"])
        return
    keys = list(results[0].keys())
    ws.append(keys)
    for result in results:
        ws.append([result.get(key, "") for key in keys])
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col) + 2
        ws.column_dimensions[letter].width = min(width, 28)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def save_excel(results_eff, results_pinch, results_pressure):
    filename = "Чувствительность Рекомпрессия.xlsx"
    wb = Workbook()

    ws = wb.active
    ws.title = "KPD_comp_turb"
    write_sheet(ws, results_eff)

    ws = wb.create_sheet("dT_HTHE_LTHE")
    write_sheet(ws, results_pinch)

    ws = wb.create_sheet("dP_hot_cold")
    write_sheet(ws, results_pressure)

    ws = wb.create_sheet("Parameters")
    params = [
        ("Параметр", "Значение"),
        ("P1, МПа", P1_FIXED / 1e6),
        ("P7, МПа", P7_FIXED / 1e6),
        ("x", X_FIXED),
        ("TR1, °C", TR1 - 273.15),
        ("TR2, °C", TR2 - 273.15),
        ("T2_set, °C", T2_set - 273.15),
        ("T7, °C", T7 - 273.15),
        ("Q_HEAT, МВт", Q_HEAT / 1e6),
        ("eta_turb_mech", eta_turb_mech),
        ("eta_generator", eta_generator),
        ("eta_mc_mech", eta_mc_mech),
        ("eta_rc_mech", eta_rc_mech),
        ("eta_drive", eta_drive),
        ("KPDcomp базовый", KPDcomp_BASE),
        ("KPDturb базовый", KPDturb_BASE),
        ("dT HTHE базовый, K", dTh_BASE),
        ("dT LTHE базовый, K", dTl_BASE),
        ("dP hot базовый", dP_HTHE_hot_BASE),
        ("dP cold базовый", dP_HTHE_cold_BASE),
        ("dP COOL фиксированный", dP_COOL),
        ("P_Na, МПа", P_Na / 1e6),
        ("P_water, МПа", P_water / 1e6),
        ("КПД насоса", KPD_pump),
        ("КПД двигателя", KPD_motor),
        ("dT воды, K", DT_water),
    ]
    for row in params:
        ws.append(row)
    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = 20

    wb.save(filename)
    return filename

if __name__ == "__main__":
    mp.freeze_support()
    start_time = time.time()

    print("=" * 70)
    print("ЧУВСТВИТЕЛЬНОСТЬ sCO2 ЦИКЛА С РЕКОМПРЕССИЕЙ")
    print("=" * 70)
    print(f"Фиксированная точка: P1={P1_FIXED/1e6:.1f} МПа, P7={P7_FIXED/1e6:.1f} МПа, x={X_FIXED:.2f}")
    print(f"TR1={TR1-273.15:.1f} °C, TR2={TR2-273.15:.1f} °C, T2_set={T2_set-273.15:.1f} °C, T7={T7-273.15:.1f} °C")
    print("Физическое ограничение: T1 < TR2")
    print("Модель соответствует главному коду (itertools.txt)")
    print()

    tasks, n_eff, n_pinch, n_press = make_tasks()
    total = len(tasks)
    nproc = min(14, mp.cpu_count())
    print(f"Процессов: {nproc}")
    print()

    results = [None] * total
    with mp.Pool(processes=nproc) as pool:
        for idx, result in tqdm(pool.imap_unordered(worker, tasks, chunksize=1),
                                total=total, desc="Расчёт", unit="точка", dynamic_ncols=True):
            results[idx] = result

    results_eff = results[:n_eff]
    results_pinch = results[n_eff:n_eff + n_pinch]
    results_pressure = results[n_eff + n_pinch:]

    filename = save_excel(results_eff, results_pinch, results_pressure)

    def count_physical(arr):
        return sum(bool(r.get("physical_root", False)) for r in arr if r is not None)

    def best_result(arr):
        good = [r for r in arr if r is not None and r.get("physical_root", False)
                and np.isfinite(r.get("eta_el_net_%", np.nan))]
        if not good:
            return None
        return max(good, key=lambda r: r["eta_el_net_%"])

    print()
    print("=" * 70)
    print("РЕЗУЛЬТАТ")
    print("=" * 70)
    print(f"КПД машин: {count_physical(results_eff)}/{len(results_eff)}")
    print(f"Недогревы: {count_physical(results_pinch)}/{len(results_pinch)}")
    print(f"Потери давления: {count_physical(results_pressure)}/{len(results_pressure)}")

    for name, arr in [("КПД машин", results_eff),
                      ("Недогревы", results_pinch),
                      ("Потери давления", results_pressure)]:
        b = best_result(arr)
        if b is not None:
            print(f"{name}: максимум eta_el_net = {b['eta_el_net_%']:.4f} %, "
                  f"var1={b['var1']}, var2={b['var2']}")
        else:
            print(f"{name}: физических решений нет")

    elapsed = time.time() - start_time
    print()
    print(f"Время расчёта: {elapsed/60:.2f} мин")
    print(f"Excel: {filename}")
    print(f"Текущее время: {datetime.now()}")