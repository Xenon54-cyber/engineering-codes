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

TR2 = 550.0 + 273.15  # Температура Na на входе в реактор, K
TR1 = 730.0 + 273.15  # Температура Na на выходе из реактора, K
T2_set = 715.0 + 273.15  # Расчётная температура CO2 после нагревателя,до первого транспортного участка, K
T7 = 35.0 + 273.15  # Температура CO2 на входе в главный компрессор, K
Q_HEAT = 2.75e6  # Номинальная тепловая мощность реактора, W

KPDcomp = 0.80  # Внутренний относительный КПД компрессоров
KPDturb = 0.90  # Внутренний относительный КПД турбины
KPD_reactor = 0.99  # КПД реактора
KPD_RTO = 0.99  # КПД РТО

eta_turb_mech = 0.99  # Механический КПД турбины
eta_generator = 0.99  # КПД электрогенератора
eta_mc_mech = 0.97  # Механический КПД главного компрессора
eta_rc_mech = 0.97  # Механический КПД рекомпрессора
eta_drive = 0.98  # КПД электропривода компрессоров

dTh = 5.0  # Минимальный температурный напор HTHE, K
dTl = 5.0  # Минимальный температурный напор LTHE, K

dP_HTHE_cold = 0.02
dP_HTHE_hot = 0.03
dP_LTHE_cold = 0.02
dP_LTHE_hot = 0.03
dP_COOL = 0.03

P_Na = 0.16e6  # Давление натрия, Pa
P_water = 0.50e6  # Давление воды, Pa
dP_Na = 0.05  # Относительные потери давления в натриевом контуре
dP_water = 0.10  # Относительные потери давления в водяном контуре
KPD_pump = 0.80
KPD_motor = 0.95
DT_water = 10.0  # Подогрев воды, K
rho_Na = 850.0  # Плотность натрия, kg/m3
cp_Na = 1270.0  # Теплоёмкость натрия, J/(kg K)
rho_water = 1000.0  # Плотность воды, kg/m3
cp_water = 4180.0  # Теплоёмкость воды, J/(kg K)
N_SN_reactor = 0.0  # Дополнительные собственные нужды реактора, W
eta_tr1 = 0.99  # КПД первого транспортного участка
eta_tr2 = 0.99  # КПД второго транспортного участка

Q_reactor = Q_HEAT * KPD_reactor
Q_tr1 = Q_reactor * eta_tr1
Q_CO2_RTO = Q_tr1 * KPD_RTO

P1_range = np.arange(24.0, 35.0, 1.0) * 1e6
P7_range = np.arange(8.0, 10.0, 0.50) * 1e6
x_range = np.arange(0.6, 0.7, 0.1)

# P1_range = np.array([25.0]) * 1e6
# P7_range = np.array([10.0]) * 1e6
# x_range = np.array([0.7])

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
            # Теплоподвод к CO2 в РТО до транспортировки
            dH_heat = H2_before_transport - H1
            if not np.isfinite(dH_heat) or dH_heat <= 0.0:
                return np.full(4, 1e12)
            G1 = Q_CO2_RTO / dH_heat
            if not np.isfinite(G1) or G1 <= 0.0:
                return np.full(4, 1e12)
            Node("1", G=G1, P=P1, T=T1, fluid="CO2")
            heat_exch("HEAT", "1", "2", "R1", "R2", Q=Q_CO2_RTO, T21=TR1, T22=TR2,
                      hot_no_cp=True, fluid2="Na", G2=G_Na, P21=P_Na, P22=P_Na * (1.0 - dP_Na))

            # ------------------------------------------------
            # Первый транспортный участок
            #
            # eta_tr1 учитывается как падение температуры:
            # T2_after = T1 + eta_tr1 * (T2_before - T1)
            # При eta_tr1 = 0.99 температура уменьшается
            # на 1% от температурного перепада T2_before - T1.
            # ------------------------------------------------
            T2_after_transport = T1 + eta_tr1 * (T2_before_transport - T1)
            # Приближённые тепловые потери CO2 на первом транспортном участке
            Q_transport_loss_CO2 = G1 * dH_heat * (1.0 - eta_tr1)
            # Тепловая мощность CO2 после транспортировки
            Q_CO2_to_turbine = G1 * dH_heat * eta_tr1

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
            T2_after_transport = T1 + eta_tr1 * (T2_before_transport - T1)
            H2_after_transport = prop("H", "T", T2_after_transport, "P", P1, "CO2")
            Q_transport_loss_CO2 = G1 * dH_heat * (1.0 - eta_tr1)

            Q_CO2_to_turbine = G1 * (H2_before_transport-H2_after_transport)

            node_data = {}
            for number in range(1, 13):
                temperature = node_value(nodes_at, number, "T")
                pressure = node_value(nodes_at, number, "P")
                mass_flow = node_value(nodes_at, number, "G")
                node_data[f"T{number}_C"] = temperature - 273.15 if np.isfinite(temperature) else np.nan
                node_data[f"P{number}_MPa"] = pressure / 1e6 if np.isfinite(pressure) else np.nan
                node_data[f"G{number}_kg_s"] = mass_flow if np.isfinite(mass_flow) else np.nan

            P9 = node_value(nodes_at, 9, "P")
            P11 = node_value(nodes_at, 11, "P")
            P12 = node_value(nodes_at, 12, "P")
            if not all(np.isfinite(value) for value in [P9, P11, P12]):
                continue
            mix_pressure_error = max(abs(P9 - P11), abs(P11 - P12), abs(P9 - P12))

            H2 = get_node_h(nodes_at, 2)
            H3 = get_node_h(nodes_at, 3)
            H6 = get_node_h(nodes_at, 6)
            H7 = get_node_h(nodes_at, 7)
            H8 = get_node_h(nodes_at, 8)
            H10 = get_node_h(nodes_at, 10)
            H11 = get_node_h(nodes_at, 11)
            G2 = node_value(nodes_at, 2, "G")
            G6 = node_value(nodes_at, 6, "G")
            G7 = node_value(nodes_at, 7, "G")
            G10 = node_value(nodes_at, 10, "G")
            required_values = [H2, H3, H6, H7, H8, H10, H11, G2, G6, G7, G10]
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

            Q_EM_TURB = N_TURB_i*(1- (eta_turb_mech*eta_generator))
            Q_EM_RCOMP = N_RCOMP_i*(1-(eta_rc_mech*eta_drive))
            Q_EM_MCOMP = N_MCOMP_i*(1-(eta_mc_mech*eta_drive))

            Q_EM_COMP = Q_EM_RCOMP + Q_EM_MCOMP
            Q_EM_ASMM = Q_EM_TURB + Q_EM_COMP

            Q_COOL = G6 * (H6 - H7)
            if not np.isfinite(Q_COOL) or Q_COOL <= 0.0:
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
            N_SN = N_pumps + N_SN_reactor

            N_el_gross = N_TURB_el - N_MCOMP_el - N_RCOMP_el
            N_el_net = N_el_gross - N_SN
            N_net_cycle = N_TURB_i - N_MCOMP_i - N_RCOMP_i

            eta_cycle = N_net_cycle / Q_HEAT
            eta_el_gross = N_el_gross / Q_HEAT
            eta_el_net = N_el_net / Q_HEAT

            Q_loss_r = Q_HEAT * (1-KPD_reactor)
            Q_loss_tr1 = Q_reactor - (Q_reactor * eta_tr1)
            Q_loss_rto = (Q_HEAT * KPD_reactor * eta_tr1) - (Q_HEAT * KPD_reactor * eta_tr1 * KPD_RTO)
            Q_tr_GTU = Q_transport_loss_CO2 + Q_loss_tr1

            Q_loss_reactor = Q_HEAT - Q_reactor
            Q_loss_tr1 = Q_reactor - Q_tr1
            Q_loss_RTO = Q_tr1 - Q_CO2_RTO

            Q_EM = Q_EM_TURB + Q_EM_COMP + N_pumps

            Q_loss_reactor = Q_HEAT - Q_reactor
            Q_loss_tr1 = Q_reactor - Q_tr1
            Q_loss_RTO = Q_tr1 - Q_CO2_RTO

            Q_loss_reactor = Q_HEAT - Q_reactor
            Q_loss_tr1 = Q_reactor - Q_tr1
            Q_loss_RTO = Q_tr1 - Q_CO2_RTO

            Q_balance_out = (Q_loss_reactor+ Q_loss_tr1+ Q_loss_RTO+ Q_COOL+ N_el_net)

            power_values = [N_TURB_el, N_MCOMP_el, N_RCOMP_el, N_Na_pump_el, N_water_pump_el,
                            N_pumps, N_SN, N_el_gross, N_el_net, eta_cycle, eta_el_gross, eta_el_net,
                            Q_transport_loss_CO2, Q_CO2_to_turbine, Q_tr_GTU]
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
                "Q_loss_reactor_MW": Q_loss_r / 1e6,
                "Q_loss_tr1_MW": Q_loss_tr1 / 1e6,
                "Q_loss_RTO_MW": Q_loss_rto / 1e6,
                "Q_EM_TURB_MW": Q_EM_TURB / 1e6,
                "Q_EM_COMP_MW": Q_EM_COMP / 1e6,
                "Q_EM_ASMM_MW": Q_EM_ASMM / 1e6,
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


def save_excel(results):
    results = [result for result in results if result is not None]
    filename = "sCO2_TR2_550_transport_loss.xlsx"
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Results"
    if not results:
        worksheet.append(["Результатов, удовлетворяющих условию T1 < TR2, нет"])
        workbook.save(filename)
        return filename, 0
    columns = list(results[0].keys())
    worksheet.append(columns)
    for result in results:
        worksheet.append([result.get(column, "") for column in columns])
    # Автоматическая ширина столбцов
    for column in worksheet.columns:
        letter = get_column_letter(column[0].column)
        width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column) + 2
        worksheet.column_dimensions[letter].width = min(width, 30)
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
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
    print("Учитывается падение температуры CO2 на первом транспортном участке")
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