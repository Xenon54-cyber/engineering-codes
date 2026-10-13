import itertools
import multiprocessing as mp
import time
from datetime import datetime
import numpy as np
from CoolProp.CoolProp import PropsSI as prop
from scipy.optimize import root
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from tqdm import tqdm
from data import nodes, blocks
from modules import Node, comp, turb, heat_source, heat_exch, split, mix

TR2 = 550.0 + 273.15
TR1 = 730.0 + 273.15
T2_set = 715.0 + 273.15
T7 = 35.0 + 273.15
Q_HEAT = 2.75e6

KPDcomp = 0.80
KPDturb = 0.90
KPD_reactor = 0.99
KPD_RTO = 0.99

eta_turb_mech = 0.99
eta_generator = 0.99
eta_mc_mech = 0.97
eta_rc_mech = 0.97
eta_drive = 0.98

dTh = 5.0
dTl = 5.0

dP_HTHE_cold = 0.02
dP_HTHE_hot = 0.03
dP_LTHE_cold = 0.02
dP_LTHE_hot = 0.03
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

Q_reactor = Q_HEAT * KPD_reactor
Q_tr1 = Q_reactor * eta_tr1
Q_CO2_RTO = Q_tr1 * KPD_RTO

P1_range = np.arange(20.0, 35.0, 0.50) * 1e6
P7_range = np.arange(8.0, 13.0, 0.50) * 1e6
x_range = np.arange(0.4, 0.8, 0.05)

ROOT_GUESSES = [
    np.array([TR2 - 5.0, 450.0, 350.0, 300.0]),
    np.array([TR2 - 20.0, 500.0, 300.0, 250.0]),
    np.array([TR2 - 120.0, 400.0, 350.0, 300.0]),
]
ROOT_TOL = 1e-4


def node_value(nodes_at, number, field):
    try:
        value = nodes_at.loc[str(number), field]
        if value is None:
            return np.nan
        value = float(value)
        if not np.isfinite(value):
            return np.nan
        return value
    except Exception:
        return np.nan


def block_value(blocks_at, name, field):
    try:
        value = blocks_at.loc[name, field]
        if value is None:
            return np.nan
        value = float(value)
        if not np.isfinite(value):
            return np.nan
        return value
    except Exception:
        return np.nan


def reset_node_pressure(nodes_at, number, pressure):
    temperature = node_value(nodes_at, number, "T")
    mass_flow = node_value(nodes_at, number, "G")
    if np.isfinite(temperature) and np.isfinite(mass_flow):
        Node(str(number), G=mass_flow, P=pressure, T=temperature, fluid="CO2")


def get_node_h(nodes_at, number):
    temperature = node_value(nodes_at, number, "T")
    pressure = node_value(nodes_at, number, "P")
    if not np.isfinite(temperature) or not np.isfinite(pressure):
        return np.nan
    try:
        return prop("H", "T", temperature, "P", pressure, "CO2")
    except Exception:
        return np.nan


def clear_state():
    nodes.loc[:, :] = np.nan
    blocks.loc[:, :] = np.nan


def compute_point(P1, P7, x):
    nodes_at = nodes
    blocks_at = blocks
    clear_state()

    P12_target = P1 / (1.0 - dP_HTHE_cold)
    P9_target = P12_target
    P8_target = P9_target / (1.0 - dP_LTHE_cold)
    P7_target = P7
    P6_target = P7_target / (1.0 - dP_COOL)
    P5_target = P6_target
    P4_target = P5_target / (1.0 - dP_LTHE_hot)
    P3_target = P4_target / (1.0 - dP_HTHE_hot)

    dT_Na = TR1 - TR2
    if dT_Na <= 0.0:
        return None
    G_Na = Q_reactor / (cp_Na * dT_Na)
    if not np.isfinite(G_Na) or G_Na <= 0.0:
        return None

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
            if not np.isfinite(Q_transport_loss_CO2):
                return np.full(4, 1e12)
            if not np.isfinite(Q_CO2_to_turbine):
                return np.full(4, 1e12)

            Node("2", G=G1, P=P1, T=T2_after_transport, fluid="CO2")
            turb("TURB", "2", "3", P3_target, KPDturb)

            Node("12", G=G1, P=P12_target, T=T12, fluid="CO2")
            heat_exch("HTHE", "3", "4", "12", "1", T22=T1)
            reset_node_pressure(nodes_at, 4, P4_target)
            reset_node_pressure(nodes_at, 1, P1)

            Node("5", G=G1, P=P5_target, T=T5, fluid="CO2")
            Node("8", G=G1 * x, P=P8_target, T=T8, fluid="CO2")
            heat_exch("LTHE", "4", "5", "8", "9", T12=T5)
            reset_node_pressure(nodes_at, 5, P5_target)
            reset_node_pressure(nodes_at, 8, P8_target)
            reset_node_pressure(nodes_at, 9, P9_target)
            split("SPLIT", "5", "6", "10", x)
            reset_node_pressure(nodes_at, 6, P6_target)
            reset_node_pressure(nodes_at, 10, P5_target)
            heat_source("COOL", "6", "7", T=T7)
            reset_node_pressure(nodes_at, 7, P7_target)
            comp("MCOMP", "7", "8", P8_target, KPDcomp)
            comp("ACOMP", "10", "11", P12_target, KPDcomp)
            reset_node_pressure(nodes_at, 8, P8_target)
            reset_node_pressure(nodes_at, 11, P12_target)
            reset_node_pressure(nodes_at, 9, P12_target)
            mix("MIX", "9", "11", "12")
            reset_node_pressure(nodes_at, 12, P12_target)

            equation1 = block_value(blocks_at, "HTHE", "dT") - dTh
            equation2 = node_value(nodes_at, 12, "T") - T12
            equation3 = node_value(nodes_at, 8, "T") - T8
            equation4 = block_value(blocks_at, "LTHE", "dT") - dTl
            residual = np.array([equation1, equation2, equation3, equation4], dtype=float)
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
            T2_before_transport = T2_set
            H2_before_transport = prop("H", "T", T2_before_transport, "P", P1, "CO2")
            dH_heat = H2_before_transport - H1
            G1 = Q_CO2_RTO / dH_heat

            H2_after_transport = H1 + eta_tr1 * dH_heat
            T2_after_transport = prop("T", "H", H2_after_transport, "P", P1, "CO2")
            Q_CO2_to_turbine = G1 * (H2_after_transport - H1)
            Q_transport_loss_CO2 = G1 * (H2_before_transport - H2_after_transport)

            node_data = {}
            for number in range(1, 13):
                temperature = node_value(nodes_at, number, "T")
                pressure = node_value(nodes_at, number, "P")
                mass_flow = node_value(nodes_at, number, "G")
                node_data[f"T{number}_C"] = temperature - 273.15 if np.isfinite(temperature) else np.nan
                node_data[f"P{number}_MPa"] = pressure / 1e6 if np.isfinite(pressure) else np.nan
                node_data[f"G{number}_kg_s"] = mass_flow if np.isfinite(mass_flow) else np.nan

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
            required_values = [H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12,
                               G2, G6, G7, G8, G10]
            if not all(np.isfinite(value) for value in required_values):
                continue

            N_TURB_i = G2 * (H2 - H3)
            N_TURB_e = N_TURB_i * eta_turb_mech
            N_TURB_el = N_TURB_e * eta_generator

            N_RCOMP_i = G7 * (H8 - H7)
            N_RCOMP_e = N_RCOMP_i / eta_rc_mech
            N_RCOMP_el = N_RCOMP_e / eta_drive

            N_MCOMP_i = G10 * (H11 - H10)
            N_MCOMP_e = N_MCOMP_i / eta_mc_mech
            N_MCOMP_el = N_MCOMP_e / eta_drive

            # Электромеханические потери по элементам
            Q_EM_TURB = N_TURB_i - N_TURB_el
            Q_EM_MCOMP = N_MCOMP_el - N_MCOMP_i
            Q_EM_RCOMP = N_RCOMP_el - N_RCOMP_i

            Q_EM_COMP = Q_EM_MCOMP + Q_EM_RCOMP
            Q_EM_ASMM = Q_EM_TURB + Q_EM_COMP

            Q_COOL = G6 * (H6 - H7)
            if not np.isfinite(Q_COOL) or Q_COOL <= 0.0:
                continue

            Q_COOL_diagram = Q_COOL - Q_EM_ASMM

            Q_HTHE_hot = G2 * (H3 - H4)
            Q_HTHE_cold = G1 * (H1 - H12)
            Q_LTHE_hot = G1 * (H4 - H5)
            Q_LTHE_cold = G8 * (H9 - H8)

            Q_recup = (Q_HTHE_hot - Q_HTHE_cold) + (Q_LTHE_hot - Q_LTHE_cold)

            Q_COOL_eff = Q_COOL + Q_recup
            if not np.isfinite(Q_COOL_eff) or Q_COOL_eff <= 0.0:
                continue

            deltaP_Na = P_Na * dP_Na
            N_Na_pump_hyd = (G_Na / rho_Na) * deltaP_Na
            N_Na_pump_el = N_Na_pump_hyd / (KPD_pump * KPD_motor)

            if DT_water <= 0.0:
                continue
            G_water = Q_COOL / (cp_water * DT_water)
            if not np.isfinite(G_water) or G_water <= 0.0:
                continue
            deltaP_water = P_water * dP_water
            N_water_pump_hyd = (G_water / rho_water) * deltaP_water
            N_water_pump_el = N_water_pump_hyd / (KPD_pump * KPD_motor)

            N_pumps = N_Na_pump_el + N_water_pump_el
            Q_EM_pumps = N_pumps  # потери насосов = их электрическая мощность
            N_SN = N_pumps + N_SN_reactor

            N_el_gross = N_TURB_el - N_MCOMP_el - N_RCOMP_el
            N_el_net = N_el_gross - N_SN
            N_net_cycle = N_TURB_i - N_MCOMP_i - N_RCOMP_i

            eta_cycle = N_net_cycle / Q_HEAT
            eta_el_gross = N_el_gross / Q_HEAT
            eta_el_net = N_el_net / Q_HEAT

            Q_loss_reactor = Q_HEAT - Q_reactor
            Q_loss_tr1 = Q_reactor - Q_tr1
            Q_loss_RTO = Q_tr1 - Q_CO2_RTO

            # Баланс энергии с раздельными ЭМ-потерями
            Q_balance_out = (Q_loss_reactor + Q_loss_tr1 + Q_loss_RTO
                             + Q_transport_loss_CO2
                             + Q_EM_TURB + Q_EM_MCOMP + Q_EM_RCOMP + Q_EM_pumps
                             + Q_COOL_eff
                             + N_el_net)
            Q_balance_error = Q_HEAT - Q_balance_out

            power_values = [N_TURB_el, N_MCOMP_el, N_RCOMP_el, N_Na_pump_el, N_water_pump_el,
                            N_pumps, N_SN, N_el_gross, N_el_net, eta_cycle, eta_el_gross, eta_el_net,
                            Q_transport_loss_CO2, Q_CO2_to_turbine, Q_balance_out, Q_balance_error,
                            Q_HTHE_hot, Q_HTHE_cold, Q_LTHE_hot, Q_LTHE_cold, Q_recup,
                            Q_COOL_eff, Q_EM_TURB, Q_EM_MCOMP, Q_EM_RCOMP, Q_EM_pumps]
            if not all(np.isfinite(value) for value in power_values):
                continue
            if N_TURB_el <= 0.0:
                continue
            if N_MCOMP_el < 0.0 or N_RCOMP_el < 0.0:
                continue
            if N_pumps < 0.0:
                continue
            if N_el_net <= 0.0:
                continue
            if eta_el_net <= 0.0:
                continue

            result = {
                "P1_MPa": P1 / 1e6,
                "P7_MPa": P7 / 1e6,
                "x": x,
                "T2_set_C": T2_set - 273.15,
                "T2_after_transport_C": T2_after_transport - 273.15,
                "N_TURB_i_MW": N_TURB_i / 1e6,
                "N_TURB_el_MW": N_TURB_el / 1e6,
                "N_MCOMP_i_MW": N_MCOMP_i / 1e6,
                "N_MCOMP_el_MW": N_MCOMP_el / 1e6,
                "N_RCOMP_i_MW": N_RCOMP_i / 1e6,
                "N_RCOMP_el_MW": N_RCOMP_el / 1e6,
                "N_Na_pump_hyd": N_Na_pump_hyd / 1e6,
                "N_water_pump_hyd": N_water_pump_hyd / 1e6,
                "N_pump_Na_MW": N_Na_pump_el / 1e6,
                "N_pump_water_MW": N_water_pump_el / 1e6,
                "N_pumps_MW": N_pumps / 1e6,
                "N_SN_reactor_MW": N_SN_reactor / 1e6,
                "N_SN_MW": N_SN / 1e6,
                "N_net_cycle_MW": N_net_cycle / 1e6,
                "N_el_gross_MW": N_el_gross / 1e6,
                "N_el_net_MW": N_el_net / 1e6,
                "Q_HEAT_MW": Q_HEAT / 1e6,
                "Q_reactor_MW": Q_reactor / 1e6,
                "Q_CO2_RTO_MW": Q_CO2_RTO / 1e6,
                "Q_CO2_to_turbine_MW": Q_CO2_to_turbine / 1e6,
                "Q_transport_loss_CO2_MW": Q_transport_loss_CO2 / 1e6,
                "Q_COOL_MW": Q_COOL / 1e6,
                "Q_COOL_eff_MW": Q_COOL_eff / 1e6,
                "Q_COOL_diagram_MW": Q_COOL_diagram / 1e6,
                "Q_loss_reactor_MW": Q_loss_reactor / 1e6,
                "Q_loss_tr1_MW": Q_loss_tr1 / 1e6,
                "Q_loss_RTO_MW": Q_loss_RTO / 1e6,
                # Раздельные ЭМ-потери
                "Q_EM_TURB_MW": Q_EM_TURB / 1e6,
                "Q_EM_MCOMP_MW": Q_EM_MCOMP / 1e6,
                "Q_EM_RCOMP_MW": Q_EM_RCOMP / 1e6,
                "Q_EM_COMP_MW": Q_EM_COMP / 1e6,
                "Q_EM_pumps_MW": Q_EM_pumps / 1e6,
                "Q_EM_ASMM_MW": Q_EM_ASMM / 1e6,
                "Q_HTHE_hot_MW": Q_HTHE_hot / 1e6,
                "Q_HTHE_cold_MW": Q_HTHE_cold / 1e6,
                "Q_LTHE_hot_MW": Q_LTHE_hot / 1e6,
                "Q_LTHE_cold_MW": Q_LTHE_cold / 1e6,
                "Q_recup_MW": Q_recup / 1e6,
                "Q_balance_out_MW": Q_balance_out / 1e6,
                "Q_balance_error_MW": Q_balance_error / 1e6,
                "eta_cycle_%": eta_cycle * 100.0,
                "eta_el_gross_%": eta_el_gross * 100.0,
                "eta_el_net_%": eta_el_net * 100.0,
                "G_Na_kg_s": G_Na,
                "G_water_kg_s": G_water,
                "dP_Na_MPa": deltaP_Na / 1e6,
                "dP_water_MPa": deltaP_water / 1e6,
                "dP_HTHE_cold_%": dP_HTHE_cold * 100.0,
                "dP_HTHE_hot_%": dP_HTHE_hot * 100.0,
                "dP_LTHE_cold_%": dP_LTHE_cold * 100.0,
                "dP_LTHE_hot_%": dP_LTHE_hot * 100.0,
                "dP_COOL_%": dP_COOL * 100.0,
            }
            result.update(node_data)
            return result
        except Exception:
            continue
    return None


def worker(task):
    index, P1, P7, x = task
    try:
        result = compute_point(P1, P7, x)
        return index, result
    except Exception:
        return index, None


# ============================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================

def _write_header(worksheet, columns):
    worksheet.append(columns)
    for column in worksheet.columns:
        letter = get_column_letter(column[0].column)
        width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column) + 2
        worksheet.column_dimensions[letter].width = min(width, 30)
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions


def _write_rows(worksheet, results, columns):
    for result in results:
        worksheet.append([result.get(column, "") for column in columns])


def _write_best_block_sheet(worksheet, best):
    worksheet.title = "Best block"

    # Таблица по элементам схемы
    headers = ["Параметр", "Турбина", "ОК", "РК", "ВТО", "НТО", "РТО", "Холодильник", "Насосы"]
    worksheet.append(headers)

    def add_row(name, *values):
        worksheet.append([name] + list(values))

    add_row("N_i, МВт",
            best.get("N_TURB_i_MW"),
            best.get("N_MCOMP_i_MW"),
            best.get("N_RCOMP_i_MW"),
            "", "", "", "", "")
    add_row("N_el, МВт",
            best.get("N_TURB_el_MW"),
            best.get("N_MCOMP_el_MW"),
            best.get("N_RCOMP_el_MW"),
            "", "", "", "",
            best.get("N_pumps_MW"))
    add_row("Q_EM, МВт",
            best.get("Q_EM_TURB_MW"),
            best.get("Q_EM_MCOMP_MW"),
            best.get("Q_EM_RCOMP_MW"),
            "", "", "", "",
            best.get("Q_EM_pumps_MW"))
    add_row("Q, МВт",
            "", "", "",
            best.get("Q_HTHE_hot_MW"),
            best.get("Q_LTHE_hot_MW"),
            best.get("Q_CO2_RTO_MW"),
            best.get("Q_COOL_MW"),
            "")

    add_row("T вход, °C",
            best.get("T2_C"),
            best.get("T7_C"),
            best.get("T10_C"),
            best.get("T3_C"),
            best.get("T4_C"),
            "",
            best.get("T6_C"),
            "")
    add_row("T выход, °C",
            best.get("T3_C"),
            best.get("T8_C"),
            best.get("T11_C"),
            best.get("T4_C"),
            best.get("T5_C"),
            "",
            best.get("T7_C"),
            "")

    add_row("P вход, МПа",
            best.get("P2_MPa"),
            best.get("P7_MPa"),
            best.get("P10_MPa"),
            best.get("P3_MPa"),
            best.get("P4_MPa"),
            "",
            best.get("P6_MPa"),
            "")
    add_row("P выход, МПа",
            best.get("P3_MPa"),
            best.get("P8_MPa"),
            best.get("P11_MPa"),
            best.get("P4_MPa"),
            best.get("P5_MPa"),
            "",
            best.get("P7_MPa"),
            "")

    add_row("G, кг/с",
            best.get("G2_kg_s"),
            best.get("G7_kg_s"),
            best.get("G10_kg_s"),
            "", "", "",
            best.get("G6_kg_s"),
            "")

    add_row("T хол. вход, °C",
            "", "", "",
            best.get("T12_C"),
            best.get("T8_C"),
            "", "", "")
    add_row("T хол. выход, °C",
            "", "", "",
            best.get("T1_C"),
            best.get("T9_C"),
            "", "", "")
    add_row("P хол. вход, МПа",
            "", "", "",
            best.get("P12_MPa"),
            best.get("P8_MPa"),
            "", "", "")
    add_row("P хол. выход, МПа",
            "", "", "",
            best.get("P1_MPa"),
            best.get("P9_MPa"),
            "", "", "")

    # --- ЭНЕРГЕТИЧЕСКИЕ ПОКАЗАТЕЛИ ---
    worksheet.append([])
    worksheet.append(["ЭНЕРГЕТИЧЕСКИЕ ПОКАЗАТЕЛИ", "", "", "", "", "", "", "", ""])
    worksheet.append(["Q_HEAT, МВт", best.get("Q_HEAT_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["N_net_cycle, МВт", best.get("N_net_cycle_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["N_el_gross, МВт", best.get("N_el_gross_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["N_el_net, МВт", best.get("N_el_net_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["N_pumps, МВт", best.get("N_pumps_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["N_SN, МВт", best.get("N_SN_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["eta_cycle, %", best.get("eta_cycle_%"), "", "", "", "", "", "", ""])
    worksheet.append(["eta_el_gross, %", best.get("eta_el_gross_%"), "", "", "", "", "", "", ""])
    worksheet.append(["eta_el_net, %", best.get("eta_el_net_%"), "", "", "", "", "", "", ""])

    # --- ЭМ-ПОТЕРИ (раздельно) ---
    worksheet.append([])
    worksheet.append(["ЭЛЕКТРОМЕХАНИЧЕСКИЕ ПОТЕРИ", "", "", "", "", "", "", "", ""])
    worksheet.append(["Q_EM_TURB (турбина), МВт", best.get("Q_EM_TURB_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_EM_MCOMP (ОК), МВт", best.get("Q_EM_MCOMP_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_EM_RCOMP (РК), МВт", best.get("Q_EM_RCOMP_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_EM_pumps (насосы), МВт", best.get("Q_EM_pumps_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_EM_ASMM (сумма), МВт", best.get("Q_EM_ASMM_MW"), "", "", "", "", "", "", ""])

    # --- БАЛАНС ЭНЕРГИИ ---
    worksheet.append([])
    worksheet.append(["БАЛАНС ЭНЕРГИИ", "", "", "", "", "", "", "", ""])
    worksheet.append(["Q_loss_reactor, МВт", best.get("Q_loss_reactor_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_loss_tr1, МВт", best.get("Q_loss_tr1_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_loss_RTO, МВт", best.get("Q_loss_RTO_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_transport_loss_CO2, МВт", best.get("Q_transport_loss_CO2_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_EM_TURB, МВт", best.get("Q_EM_TURB_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_EM_MCOMP, МВт", best.get("Q_EM_MCOMP_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_EM_RCOMP, МВт", best.get("Q_EM_RCOMP_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_EM_pumps, МВт", best.get("Q_EM_pumps_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_COOL (модель), МВт", best.get("Q_COOL_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_recup (невязка), МВт", best.get("Q_recup_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_COOL_eff (баланс), МВт", best.get("Q_COOL_eff_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["N_el_net, МВт", best.get("N_el_net_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_balance_out, МВт", best.get("Q_balance_out_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_balance_error, МВт", best.get("Q_balance_error_MW"), "", "", "", "", "", "", ""])

    # --- ПРОВЕРКА ЦИКЛА CO2 ---
    worksheet.append([])
    worksheet.append(["ПРОВЕРКА ЦИКЛА CO2", "", "", "", "", "", "", "", ""])
    worksheet.append(["Q_CO2_to_turbine, МВт", best.get("Q_CO2_to_turbine_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_HTHE_hot, МВт", best.get("Q_HTHE_hot_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_HTHE_cold, МВт", best.get("Q_HTHE_cold_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_LTHE_hot, МВт", best.get("Q_LTHE_hot_MW"), "", "", "", "", "", "", ""])
    worksheet.append(["Q_LTHE_cold, МВт", best.get("Q_LTHE_cold_MW"), "", "", "", "", "", "", ""])

    for column in worksheet.columns:
        letter = get_column_letter(column[0].column)
        width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column) + 2
        worksheet.column_dimensions[letter].width = min(width, 40)


def save_excel(results):
    results = [result for result in results if result is not None]
    filename = "sCO2_TR2_550_transport_loss2.xlsx"
    workbook = Workbook()

    worksheet = workbook.active
    worksheet.title = "Results"
    if not results:
        worksheet.append(["Результатов, удовлетворяющих условию T1 < TR2, нет"])
        workbook.save(filename)
        return filename, 0

    columns = list(results[0].keys())
    _write_header(worksheet, columns)
    _write_rows(worksheet, results, columns)

    best = max(results, key=lambda r: r.get("eta_el_net_%", -np.inf))
    best_sheet = workbook.create_sheet("Best block")
    _write_best_block_sheet(best_sheet, best)

    workbook.save(filename)
    return filename, len(results)


if __name__ == "__main__":
    mp.freeze_support()
    start_time = time.time()
    tasks = [(index, P1, P7, x) for index, (P1, P7, x) in enumerate(itertools.product(P1_range, P7_range, x_range))]
    total = len(tasks)
    nproc = min(14, mp.cpu_count())
    print(f"TR2 = {TR2 - 273.15:.1f} °C")
    print(f"Ограничение: T1 < TR2 = {TR2 - 273.15:.1f} °C")
    print(f"T2 до транспортировки = {T2_set - 273.15:.1f} °C")
    print(f"КПД первого транспортного участка eta_tr1 = {eta_tr1:.3f}")
    print(f"КПД второго транспортного участка eta_tr2 = {eta_tr2:.3f}")
    print("Транспортный участок: H2_after = H1 + eta_tr1 * (H2_before - H1)")
    print("Потери давления: HTHE cold=2%, HTHE hot=3%, LTHE cold=2%, LTHE hot=3%, COOL=3%")
    print(f"Всего расчётных точек: {total}")
    print(f"Процессов: {nproc}")
    results = [None] * total
    with mp.Pool(processes=nproc) as pool:
        for index, result in tqdm(pool.imap_unordered(worker, tasks, chunksize=1),
                                  total=total, desc="Расчёт", unit="точка", dynamic_ncols=True):
            results[index] = result
    filename, n_success = save_excel(results)
    elapsed_minutes = (time.time() - start_time) / 60.0
    print(f"Успешных физических решений: {n_success}/{total}")
    print(f"Время расчёта: {elapsed_minutes:.2f} мин")
    print(f"Excel: {filename}")
    print(f"Текущее время: {datetime.now()}")