import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
#                 0. НАСТРОЙКИ ГРАФИКОВ
# ============================================================

# Создаём папку для графиков
os.makedirs("plots", exist_ok=True)

# Настройки шрифта
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16

FIGSIZE = (12, 8)
DPI = 300

# ============================================================
#                 1. БАЗОВЫЕ ДАННЫЕ
# ============================================================

Q_REACTOR_MW = 2.75
Q_REACTOR_KW = Q_REACTOR_MW * 1000.0

# Температуры
T_NA1_IN_C = 730.0
T_NA1_OUT_C = 550.0
T_NA2_OUT_C = 550.0           # фиксирована (на выходе из ТО2)
T_WATER_IN_C = 40.0
T_WATER_OUT_C = 70.0

# Давления
P_NA1_MPA = 1.2
P_NA2_MPA = 1.0
P_WATER_MPA = 1.6

# Теплофизика
CP_NA_J_KG_K = 1300.0
RHO_NA_KG_M3 = 850.0
RHO_WATER_KG_M3 = 970.0
CP_WATER_J_KG_K = 4180.0

# КПД насосов
ETA_PUMP_NA1 = 0.75
ETA_PUMP_NA2 = 0.75
ETA_PUMP_WATER = 0.75

# ============================================================
#                 2. ПОТЕРИ ТЕПЛА В ТЕПЛООБМЕННИКАХ
# ============================================================

# Потери в реакторном теплообменнике (1-й ↔ 2-й контур), %
LOSS_TO_NA2_PERCENT = 2.0

# Потери в теплообменнике 2-й контур ↔ вода, %
LOSS_TO_WATER_PERCENT = 1.5

# ============================================================
#                 3. РАСХОДЫ ТЕПЛОНОСИТЕЛЕЙ (ПРИ ЗАДАННОЙ Q)
# ============================================================

# Расход первичного натрия (1-й контур)
delta_T_NA1 = T_NA1_IN_C - T_NA1_OUT_C
G_NA1 = (Q_REACTOR_KW * 1000.0) / (CP_NA_J_KG_K * delta_T_NA1)

# ============================================================
#                 4. ДИАПАЗОНЫ ПЕРЕБОРА
# ============================================================

T_NA2_IN_MIN_C = 450.0
T_NA2_IN_MAX_C = 540.0
T_NA2_STEP_C = 5.0

DP_MIN_PERCENT = 1.0
DP_MAX_PERCENT = 15.0
DP_STEP_PERCENT = 1.0

T_NA2_VALUES = np.arange(T_NA2_IN_MIN_C, T_NA2_IN_MAX_C + T_NA2_STEP_C, T_NA2_STEP_C)
DP_VALUES = np.arange(DP_MIN_PERCENT, DP_MAX_PERCENT + DP_STEP_PERCENT, DP_STEP_PERCENT)

# ============================================================
#                 5. ФУНКЦИЯ МОЩНОСТИ НАСОСА
# ============================================================

def pump_power_kW(G_kg_s, dP_MPa, rho_kg_m3, eta):
    if any(x <= 0 for x in [G_kg_s, dP_MPa, rho_kg_m3, eta]):
        return np.nan
    V_m3_s = G_kg_s / rho_kg_m3
    dP_Pa = dP_MPa * 1e6
    return (V_m3_s * dP_Pa / eta) / 1000.0

# ============================================================
#                 6. ОСНОВНОЙ РАСЧЁТ
# ============================================================

results = []

for T_NA2_IN_C in T_NA2_VALUES:

    delta_T_NA2 = abs(T_NA2_IN_C - T_NA2_OUT_C)
    if delta_T_NA2 <= 0:
        continue

    # Расход вторичного натрия (2-й контур) без учёта потерь
    G_NA2_ideal = (Q_REACTOR_KW * 1000.0) / (CP_NA_J_KG_K * delta_T_NA2)

    # Реальный расход с учётом потерь в реакторном теплообменнике
    loss_factor_na2 = 1.0 - LOSS_TO_NA2_PERCENT / 100.0
    G_NA2_real = G_NA2_ideal / loss_factor_na2

    # Тепло, переданное от Na2 к воде (с учётом потерь)
    Q_to_water_ideal = Q_REACTOR_KW * loss_factor_na2
    Q_to_water_real = Q_to_water_ideal * (1.0 - LOSS_TO_WATER_PERCENT / 100.0)

    # Расход воды
    delta_T_WATER = T_WATER_OUT_C - T_WATER_IN_C
    G_WATER = (Q_to_water_real * 1000.0) / (CP_WATER_J_KG_K * delta_T_WATER)

    for DP_percent in DP_VALUES:

        # Перепады давления (пропорционально текущему давлению)
        dP_NA1 = P_NA1_MPA * (DP_percent / 100.0)
        dP_NA2 = P_NA2_MPA * (DP_percent / 100.0)
        dP_WATER = P_WATER_MPA * (DP_percent / 100.0)

        # Мощность насосов
        N_NA1 = pump_power_kW(G_NA1, dP_NA1, RHO_NA_KG_M3, ETA_PUMP_NA1)
        N_NA2 = pump_power_kW(G_NA2_real, dP_NA2, RHO_NA_KG_M3, ETA_PUMP_NA2)
        N_WATER = pump_power_kW(G_WATER, dP_WATER, RHO_WATER_KG_M3, ETA_PUMP_WATER)

        N_SN_TOTAL = N_NA1 + N_NA2 + N_WATER

        # КПД брутто (по полезной теплоте у воды)
        eta_gross = (Q_to_water_real / Q_REACTOR_KW) * 100.0

        # Потеря КПД от собственных нужд (п.п.)
        eta_SN_loss = (N_SN_TOTAL / Q_REACTOR_KW) * 100.0

        # КПД нетто
        eta_net = eta_gross - eta_SN_loss

        results.append({
            "T_NA2_IN_C": T_NA2_IN_C,
            "DP_percent": DP_percent,
            "G_NA1_kg_s": G_NA1,
            "G_NA2_real_kg_s": G_NA2_real,
            "G_water_kg_s": G_WATER,
            "Q_to_water_kW": Q_to_water_real,
            "N_NA1_kW": N_NA1,
            "N_NA2_kW": N_NA2,
            "N_water_kW": N_WATER,
            "N_SN_total_kW": N_SN_TOTAL,
            "eta_gross_%": eta_gross,
            "eta_SN_loss_pp": eta_SN_loss,
            "eta_net_%": eta_net
        })

df = pd.DataFrame(results)
df.to_excel("heat_net_losses.xlsx", index=False)

print("Результаты сохранены в heat_net_losses.xlsx")
print(f"Макс КПД нетто: {df['eta_net_%'].max():.3f} %")
print(f"Мин КПД нетто: {df['eta_net_%'].min():.3f} %")

# ============================================================
#                 7. ГРАФИКИ
# ============================================================

# График 1: КПД нетто vs T_Na2_in
plt.figure(figsize=FIGSIZE)
for DP in DP_VALUES[::2]:
    data = df[df["DP_percent"] == DP].sort_values("T_NA2_IN_C")
    plt.plot(
        data["T_NA2_IN_C"],
        data["eta_net_%"],
        marker='o',
        markersize=6,
        linewidth=2.0,
        label=f"ΔP = {DP:.0f}%"
    )
plt.xlabel("Температура Na теплофикации на входе в РТО, °C")
plt.ylabel("КПД нетто, %")
plt.title("Зависимость КПД нетто от температуры входа Na в контуре теплофикации")
plt.grid(alpha=0.3)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join("plots", "eta_net_vs_T_Na2.png"), dpi=DPI, bbox_inches="tight")
plt.show()
plt.close()

# График 2: состав собственных нужд (при ΔP = 5%)
DP_FIX = 5.0
data = df[df["DP_percent"] == DP_FIX].sort_values("T_NA2_IN_C")

plt.figure(figsize=FIGSIZE)
plt.plot(
    data["T_NA2_IN_C"],
    data["N_NA1_kW"],
    marker='o',
    markersize=6,
    linewidth=2.0,
    label="Насос Na реактора"
)
plt.plot(
    data["T_NA2_IN_C"],
    data["N_NA2_kW"],
    marker='s',
    markersize=6,
    linewidth=2.0,
    label="Насос Na теплофикации"
)
plt.plot(
    data["T_NA2_IN_C"],
    data["N_water_kW"],
    marker='d',
    markersize=6,
    linewidth=2.0,
    label="Водяной насос"
)
plt.plot(
    data["T_NA2_IN_C"],
    data["N_SN_total_kW"],
    linewidth=3.0,
    linestyle='--',
    color='black',
    label="Суммарная потребляемая мощность"
)
plt.xlabel("Температура Na теплофикации на входе в РТО, °C")
plt.ylabel("Мощность, кВт")
plt.title(f"Состав собственных нужд при ΔP = {DP_FIX:.0f}%")
plt.grid(alpha=0.3)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join("plots", "SN_components_fixed_DP.png"), dpi=DPI, bbox_inches="tight")
plt.show()
plt.close()

print("\nГрафики сохранены в папку 'plots'")