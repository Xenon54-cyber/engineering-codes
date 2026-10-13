import numpy as np, pandas as pd, matplotlib.pyplot as plt

# ==== ФИКСИРОВАННЫЕ ПАРАМЕТРЫ ====
Q_REACTOR_MW = 2.75
T_NA1_IN_C, T_NA1_OUT_C = 550.0, 730.0
TR1_C, TR2_C, T1_C = 730.0, 550.0, 715.0
T_WATER_IN_C, T_WATER_OUT_C = 40.0, 70.0

ETA_REACTOR, ETA_TURB1, ETA_TURB2 = 0.99, 0.99, 0.99
ETA_RTO = 0.99

CP_NA, RHO_NA = 1300.0, 850.0
CP_WATER, RHO_WATER = 4180.0, 970.0
ETA_PUMP_NA1, ETA_PUMP_NA3, ETA_PUMP_WATER = 0.75, 0.75, 0.75
P_NA1_MPA, P_NA3_MPA, P_WATER_MPA = 1.2, 1.0, 1.6

# ==== ВАРЬИРУЕМЫЕ ПАРАМЕТРЫ ====
T2_MIN_C, T2_MAX_C, T2_STEP_C = 380.0, 545.0, 5.0
DP_MIN_PERCENT, DP_MAX_PERCENT, DP_STEP_PERCENT = 1.0, 15.0, 1.0
ETA_TO_MIN, ETA_TO_MAX, ETA_TO_STEP = 0.90, 0.99, 0.01

# ==== ГРАФИКИ ====
FONT_NAME, FONT_SIZE = "Times New Roman", 16
FIGSIZE, DPI = (10, 4), 500
CONTOUR_LEVELS, CONTOUR_LINES, CONTOUR_LINEWIDTH, CONTOUR_LABEL_SIZE = 25, 10, 1.5, 12
plt.rcParams["font.family"] = FONT_NAME
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["mathtext.fontset"] = "stix"

# ==== РАСХОД Na1 (фикс) ====
delta_T_NA1_C = T_NA1_OUT_C - T_NA1_IN_C
G_NA1_kg_s = Q_REACTOR_MW * 1e6 / (CP_NA * delta_T_NA1_C)
Q_РТО_МW = G_NA1_kg_s * CP_NA * (TR1_C - TR2_C) * 1e-6

def pump_power_kW(G, dP_MPa, rho, eta):
    return (G / rho) * (dP_MPa * 1e6) / eta / 1000.0

T2_VALUES = np.arange(T2_MIN_C, T2_MAX_C + T2_STEP_C/2, T2_STEP_C)
DP_VALUES = np.arange(DP_MIN_PERCENT, DP_MAX_PERCENT + DP_STEP_PERCENT/2, DP_STEP_PERCENT)
ETA_TO_VALUES = np.round(np.arange(ETA_TO_MIN, ETA_TO_MAX + ETA_TO_STEP/2, ETA_TO_STEP), 2)

def calculate_point(T2_C, DP_percent, ETA_TO):
    Q_ЯТ = Q_REACTOR_MW
    Q_Р   = Q_ЯТ  * ETA_REACTOR
    Q_ТР1 = Q_Р   * ETA_TURB1
    Q_РТО = Q_ТР1 * ETA_RTO
    Q_ТР2 = Q_РТО * ETA_TURB2
    Q_ТО  = Q_ТР2 * ETA_TO

    delta_T_TO_C = T1_C - T2_C
    if delta_T_TO_C <= 0:
        return None
    G_NA3_kg_s = Q_ТО * 1e6 / (CP_NA * delta_T_TO_C)

    delta_T_WATER_C = T_WATER_OUT_C - T_WATER_IN_C
    G_WATER_kg_s = Q_ТО * 1e6 / (CP_WATER * delta_T_WATER_C)

    dP_NA1   = P_NA1_MPA   * DP_percent / 100.0
    dP_NA3   = P_NA3_MPA   * DP_percent / 100.0
    dP_WATER = P_WATER_MPA * DP_percent / 100.0

    N_NA1   = pump_power_kW(G_NA1_kg_s,   dP_NA1,   RHO_NA,    ETA_PUMP_NA1)
    N_NA3   = pump_power_kW(G_NA3_kg_s,   dP_NA3,   RHO_NA,    ETA_PUMP_NA3)
    N_WATER = pump_power_kW(G_WATER_kg_s, dP_WATER, RHO_WATER, ETA_PUMP_WATER)
    N_SN_TOTAL = N_NA1 + N_NA3 + N_WATER
    N_SN_MW = N_SN_TOTAL / 1000.0

    eta_heat_percent = Q_ТО / Q_ЯТ * 100.0
    eta_net_percent  = (Q_ТО - N_SN_MW) / Q_ЯТ * 100.0

    return {"T2_C": T2_C, "DP_percent": DP_percent, "ETA_TO": ETA_TO,
            "delta_T_TO_C": delta_T_TO_C,
            "Q_ЯТ_MW": Q_ЯТ, "Q_ТО_MW": Q_ТО, "Q_РТО_MW": Q_РТО,
            "G_NA1_kg_s": G_NA1_kg_s, "G_NA3_kg_s": G_NA3_kg_s, "G_WATER_kg_s": G_WATER_kg_s,
            "N_NA1_kW": N_NA1, "N_NA3_kW": N_NA3, "N_WATER_kW": N_WATER,
            "N_SN_TOTAL_kW": N_SN_TOTAL, "N_SN_MW": N_SN_MW,
            "eta_heat_percent": eta_heat_percent, "eta_net_percent": eta_net_percent}

results = []
for T2 in T2_VALUES:
    for DP in DP_VALUES:
        for ETA_TO in ETA_TO_VALUES:
            r = calculate_point(T2, DP, ETA_TO)
            if r is not None:
                results.append(r)
df = pd.DataFrame(results)
df.to_excel("heat_supply_sensitivity.xlsx", index=False)

# ==== ОБЩИЕ НАСТРОЙКИ ГРАФИКОВ ====
def format_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE, labelpad=10)
    ax.set_title(title, fontsize=FONT_SIZE, pad=20)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.2)

def add_legend_outside(ax):
    ax.legend(fontsize=12, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True)

def build_map(column, i_col, j_col):
    I = np.sort(df[i_col].unique()); J = np.sort(df[j_col].unique())
    X, Y = np.meshgrid(J, I); Z = np.zeros((len(I), len(J)))
    for i, v in enumerate(I):
        for j, w in enumerate(J):
            row = df[np.isclose(df[i_col], v) & np.isclose(df[j_col], w)]
            if len(row) > 0:
                Z[i, j] = row[column].mean()
    return X, Y, Z

def plot_map(column, i_col, j_col, xlabel, ylabel, title, cbar_label, filename, fmt="%.3f"):
    X, Y, Z = build_map(column, i_col, j_col)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    cf = ax.contourf(X, Y, Z, levels=CONTOUR_LEVELS)
    cs = ax.contour(X, Y, Z, levels=CONTOUR_LINES, colors="black", linewidths=CONTOUR_LINEWIDTH)
    labels = ax.clabel(cs, inline=True, fontsize=CONTOUR_LABEL_SIZE, fmt=fmt)
    for l in labels:
        l.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    format_axes(ax, xlabel, ylabel, title)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(cbar_label, fontsize=FONT_SIZE, labelpad=10)
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.show()

# ==== 1. Суммарные СН от T2 при разных ΔP ====
fig, ax = plt.subplots(figsize=FIGSIZE)
for DP in [1, 5, 10, 15]:
    d = df[np.isclose(df["DP_percent"], DP) & np.isclose(df["ETA_TO"], 0.99)].sort_values("T2_C")
    ax.plot(d["T2_C"], d["N_SN_TOTAL_kW"], marker="o", ms=5, lw=1.8, label=f"ΔP = {DP:.0f} %")
format_axes(ax, "T₂ (вход РТО), °C", "Суммарные СН, кВт",
            "Электрические СН от T₂ при разных ΔP")
add_legend_outside(ax)
plt.tight_layout(); plt.savefig("SN_vs_T2.png", dpi=DPI, bbox_inches="tight"); plt.show()

# ==== 2. КПД нетто от T2 при разных ΔP ====
fig, ax = plt.subplots(figsize=FIGSIZE)
for DP in [1, 5, 10, 15]:
    d = df[np.isclose(df["DP_percent"], DP) & np.isclose(df["ETA_TO"], 0.99)].sort_values("T2_C")
    ax.plot(d["T2_C"], d["eta_net_percent"], marker="s", ms=5, lw=1.8, label=f"ΔP = {DP:.0f} %")
format_axes(ax, "T₂ (вход РТО), °C", r"$\eta_{нетто}$, %",
            "Тепловой КПД нетто от T₂ при разных ΔP")
add_legend_outside(ax)
plt.tight_layout(); plt.savefig("eta_net_vs_T2.png", dpi=DPI, bbox_inches="tight"); plt.show()

# ==== 3. КПД нетто от ΔP при разных T2 ====
fig, ax = plt.subplots(figsize=FIGSIZE)
for T2 in [390, 430, 470, 510]:
    d = df[np.isclose(df["T2_C"], T2) & np.isclose(df["ETA_TO"], 0.99)].sort_values("DP_percent")
    ax.plot(d["DP_percent"], d["eta_net_percent"], marker="o", ms=5, lw=1.8, label=f"T₂ = {T2} °C")
format_axes(ax, "Потери давления, %", r"$\eta_{нетто}$, %",
            "Тепловой КПД нетто от ΔP при разных T₂")
add_legend_outside(ax)
plt.tight_layout(); plt.savefig("eta_net_vs_DP.png", dpi=DPI, bbox_inches="tight"); plt.show()

# ==== 4. КПД нетто от η_ТО при разных ΔP ====
fig, ax = plt.subplots(figsize=FIGSIZE)
for DP in [1, 5, 10, 15]:
    d = df[np.isclose(df["DP_percent"], DP) & np.isclose(df["T2_C"], 430.0)].sort_values("ETA_TO")
    ax.plot(d["ETA_TO"], d["eta_net_percent"], marker="o", ms=5, lw=1.8, label=f"ΔP = {DP:.0f} %")
format_axes(ax, r"$\eta_{ТО}$", r"$\eta_{нетто}$, %",
            "Тепловой КПД нетто от $\\eta_{ТО}$ при разных ΔP")
add_legend_outside(ax)
plt.tight_layout(); plt.savefig("eta_net_vs_ETA_TO.png", dpi=DPI, bbox_inches="tight"); plt.show()

# ==== 5. Расход Na3 от T2 при разных ΔP ====
fig, ax = plt.subplots(figsize=FIGSIZE)
for DP in [1, 5, 10, 15]:
    d = df[np.isclose(df["DP_percent"], DP) & np.isclose(df["ETA_TO"], 0.99)].sort_values("T2_C")
    ax.plot(d["T2_C"], d["G_NA3_kg_s"], marker="o", ms=5, lw=1.8, label=f"ΔP = {DP:.0f} %")
format_axes(ax, "T₂ (вход РТО), °C", "Расход Na₃, кг/с",
            "Расход Na пром.контура от T₂")
add_legend_outside(ax)
plt.tight_layout(); plt.savefig("G_Na3_vs_T2.png", dpi=DPI, bbox_inches="tight"); plt.show()

# ==== 6. КПД нетто от T2 при разных η_ТО ====
fig, ax = plt.subplots(figsize=FIGSIZE)
for ETA in [0.90, 0.93, 0.96, 0.99]:
    d = df[np.isclose(df["ETA_TO"], ETA) & np.isclose(df["DP_percent"], 5.0)].sort_values("T2_C")
    ax.plot(d["T2_C"], d["eta_net_percent"], marker="o", ms=5, lw=1.8, label=f"$\\eta_{{ТО}}$ = {ETA:.2f}")
format_axes(ax, "T₂ (вход РТО), °C", r"$\eta_{нетто}$, %",
            "Тепловой КПД нетто от T₂ при разных $\\eta_{ТО}$ (ΔP = 5 %)")
add_legend_outside(ax)
plt.tight_layout(); plt.savefig("eta_net_vs_T2_ETA_TO.png", dpi=DPI, bbox_inches="tight"); plt.show()

# ==== КАРТЫ ====
plot_map("eta_net_percent", "T2_C", "DP_percent",
         "Потери давления, %", "T₂ (вход РТО), °C",
         "Тепловой КПД нетто при $\\eta_{ТО}$ = 0,99", r"$\eta_{нетто}$, %",
         "eta_net_map_T2_DP.png", "%.3f")

plot_map("eta_net_percent", "T2_C", "ETA_TO",
         r"$\eta_{ТО}$", "T₂ (вход РТО), °C",
         "Тепловой КПД нетто при ΔP = 5 %", r"$\eta_{нетто}$, %",
         "eta_net_map_T2_ETA.png", "%.3f")

plot_map("N_SN_TOTAL_kW", "T2_C", "DP_percent",
         "Потери давления, %", "T₂ (вход РТО), °C",
         "Суммарные электрические СН при $\\eta_{ТО}$ = 0,99", "N$_{СН}$, кВт",
         "SN_map_T2_DP.png", "%.2f")

plot_map("G_NA3_kg_s", "T2_C", "ETA_TO",
         r"$\eta_{ТО}$", "T₂ (вход РТО), °C",
         "Расход Na пром.контура при ΔP = 5 %", "G$_{Na3}$, кг/с",
         "G_Na3_map_T2_ETA.png", "%.3f")

plot_map("Q_ТО_MW", "T2_C", "ETA_TO",
         r"$\eta_{ТО}$", "T₂ (вход РТО), °C",
         "Q$_{ТО}$ при ΔP = 5 %", "Q$_{ТО}$, МВт",
         "Q_to_map_T2_ETA.png", "%.3f")

# ==== ПЕЧАТЬ ====
print("=" * 70)
print(f"Q реактора = {Q_REACTOR_MW:.5f} МВт")
print(f"TR1 = {TR1_C} °C, TR2 = {TR2_C} °C (fix), T1 = {T1_C} °C")
print(f"G_Na1 (fix) = {G_NA1_kg_s:.5f} кг/с")
print(f"Q_РТО (fix) = {Q_РТО_МW:.5f} МВт")
print(f"T₂: {T2_MIN_C:.1f}…{T2_MAX_C:.1f} °C (шаг {T2_STEP_C})")
print()
print(f"Диапазон η тепловой: {df['eta_heat_percent'].min():.4f}–{df['eta_heat_percent'].max():.4f} %")
print(f"Диапазон η нетто:    {df['eta_net_percent'].min():.4f}–{df['eta_net_percent'].max():.4f} %")
print(f"Диапазон N СН: {df['N_SN_TOTAL_kW'].min():.4f}–{df['N_SN_TOTAL_kW'].max():.4f} кВт")
print(f"Диапазон G_Na3: {df['G_NA3_kg_s'].min():.4f}–{df['G_NA3_kg_s'].max():.4f} кг/с")
print()
for T2 in [380, 460, 545]:
    r = df[np.isclose(df["T2_C"], T2) & np.isclose(df["DP_percent"], 5) & np.isclose(df["ETA_TO"], 0.99)].iloc[0]
    print(f"T₂ = {T2} °C, ΔP = 5 %, η_ТО = 0.99:")
    print(f"  ΔT_ТО = {r['delta_T_TO_C']:.1f} K, G_Na3 = {r['G_NA3_kg_s']:.4f} кг/с, "
          f"N_СН = {r['N_SN_TOTAL_kW']:.4f} кВт, η нетто = {r['eta_net_percent']:.4f} %")
print("=" * 70)