import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "recuperative_CO2_DTmin5K_recuperator_losses.xlsx"
SHEET_NAME = "Results"

# Фильтр по электрическому КПД нетто АСММ
ETA_MIN = 30.0
ETA_TICK_STEP = 1

FONT_NAME = "Times New Roman"
FONT_SIZE = 16
DPI = 250
LINE_WIDTH = 1.8
MARKER_SIZE = 4
EXTREME_MARKER_SIZE = 90
EXTREME_LABEL_SIZE = 11

# РАЗМЕРЫ ГРАФИКОВ

FIGSIZES = {
    "eta":        (10, 7),
    "N_turb":     (10, 7),
    "N_comp":     (10, 7),
    "N_internal": (10, 7),
    "N_el":       (10, 7),
    "Q":          (10, 7),
    "G":          (10, 7),
}

plt.rcParams["font.family"] = FONT_NAME
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["mathtext.fontset"] = "stix"

OUT_DIR = BASE_DIR / "Линейные графики Общие"
OUT_DIR.mkdir(exist_ok=True)

if not EXCEL_FILE.exists():
    raise FileNotFoundError(f"Не найден Excel-файл:\n{EXCEL_FILE}")

df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)

required_columns = [
    "P1_MPa", "P2_MPa", "physical_root",
    "eta_cycle_CO2_%", "eta_plant_%",
    "eta_el_net_CO2_%", "eta_el_net_plant_%",
    "N_TURB_i_MW", "N_MCOMP_i_MW", "N_RCOMP_i_MW", "N_NET_i_MW",
    "N_el_gross_MW", "N_el_net_MW",
    "Q_RECUP_MW", "Q_COOL_MW", "G_CO2_kg_s",
]

missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise ValueError("В таблице отсутствуют столбцы:\n" + "\n".join(missing))

d = df.copy()

if "physical_root" in d.columns:
    d = d[d["physical_root"] == True].copy()

d = d.dropna(subset=["P1_MPa", "P2_MPa"]).copy()
d["N_COMP_TOTAL_MW"] = d["N_MCOMP_i_MW"] + d["N_RCOMP_i_MW"]
d["N_INTERNAL_MW"] = d["N_TURB_i_MW"] - d["N_COMP_TOTAL_MW"]

d_work = d[d["eta_el_net_plant_%"] >= ETA_MIN].copy()

if len(d_work) < 3:
    raise ValueError(
        f"После ограничения eta_el_net_plant_% >= {ETA_MIN} "
        f"осталось недостаточно точек: {len(d_work)}"
    )

# ============================================================
# ЛИНЕЙНЫЕ ГРАФИКИ: параметр по P2 для разных значений P1
# ============================================================
def plot_lines(data, value_column, title, y_label, filename,
               xlabel, ylabel, extreme_type="max",
               contour_format="%.2f", cbar_format="%.2f",
               eta_map=False, figsize=(10, 7)):
    # Используются только исходные расчётные точки: интерполяции нет.
    P1 = data["P1_MPa"].to_numpy(dtype=float)
    P2 = data["P2_MPa"].to_numpy(dtype=float)
    Z_data = data[value_column].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    p1_values = np.sort(data["P1_MPa"].dropna().unique())
    cmap = plt.get_cmap("viridis", max(len(p1_values), 2))

    for i, p1_value in enumerate(p1_values):
        subset = data[np.isclose(data["P1_MPa"], p1_value)].sort_values("P2_MPa")
        if subset.empty:
            continue
        ax.plot(
            subset["P2_MPa"], subset[value_column],
            color=cmap(i), linewidth=LINE_WIDTH,
            marker="o", markersize=MARKER_SIZE,
            label=fr"$P_1$ = {p1_value:g} МПа"
        )

    index = None
    if extreme_type == "max":
        index = int(np.nanargmax(Z_data))
        extreme_name = "max"
    elif extreme_type == "min":
        index = int(np.nanargmin(Z_data))
        extreme_name = "min"

    if index is not None:
        P1_extreme = P1[index]
        P2_extreme = P2[index]
        Z_extreme = Z_data[index]
        ax.scatter(P2_extreme, Z_extreme, s=EXTREME_MARKER_SIZE,
                   marker="*", color="red", edgecolors="black",
                   linewidths=0.8, zorder=10, label=f"{extreme_name} = {contour_format % Z_extreme}")

    ax.set_xlabel(xlabel, fontsize=FONT_SIZE, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE, labelpad=8)
    ax.set_title(title, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis="both", labelsize=FONT_SIZE - 1, direction="in", top=True, right=True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(title=r"Параметр серии ($P_1$)", fontsize=FONT_SIZE - 4,
              title_fontsize=FONT_SIZE - 3, ncol=1,
              loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    plt.tight_layout()
    out_path = OUT_DIR / filename
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print()
    print("=" * 70)
    print(title.replace("\n", " "))
    print("=" * 70)
    print(f"Количество точек: {len(data)}")
    print(f"Диапазон P2: {P2.min():.3f} – {P2.max():.3f} МПа")
    print(f"Диапазон P1: {P1.min():.3f} – {P1.max():.3f} МПа")
    print(f"Диапазон {value_column}: {np.nanmin(Z_data):.3f} – {np.nanmax(Z_data):.3f}")
    if index is not None:
        print(f"{extreme_name} = {contour_format % Z_extreme}; P1 = {P1_extreme:.3f} МПа; P2 = {P2_extreme:.3f} МПа")
    print(f"Файл: {out_path}")

plot_lines(
    d_work, "eta_el_net_plant_%",
    "Зависимость электрического КПД нетто АСММ от $P_2$\n"
    "при различных значениях $P_1$",
    r"$\eta_{нетто}^{АСММ}$, %",
    "01_КПД_нетто_АСММ.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.0f", eta_map=True,
    figsize=FIGSIZES["eta"],
)

plot_lines(
    d_work, "eta_el_net_CO2_%",
    "Зависимость электрического КПД нетто CO$_2$-цикла от $P_2$\n"
    "при различных значениях $P_1$",
    r"$\eta_{нетто}^{CO_2}$, %",
    "02_КПД_нетто_CO2.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.0f", eta_map=True,
    figsize=FIGSIZES["eta"],
)

plot_lines(
    d_work, "eta_cycle_CO2_%",
    "Зависимость КПД цикла CO$_2$ от $P_2$\n"
    "при различных значениях $P_1$",
    r"$\eta_{цикл}^{CO_2}$, %",
    "03_КПД_цикла_CO2.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.0f", eta_map=True,
    figsize=FIGSIZES["eta"],
)

plot_lines(
    d_work, "eta_plant_%",
    "Зависимость КПД установки от $P_2$\n"
    "при различных значениях $P_1$",
    r"$\eta_{уст}$, %",
    "04_КПД_установки.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.0f", eta_map=True,
    figsize=FIGSIZES["eta"],
)

# ============================================================
# 2. Мощности
# ============================================================
plot_lines(
    d_work, "N_TURB_i_MW",
    "Внутренняя мощность турбины\n"
    "от $P_2$ при различных значениях $P_1$",
    r"$N_{т,вн}$, МВт",
    "05_Мощность_турбины.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.2f",
    figsize=FIGSIZES["N_turb"],
)

plot_lines(
    d_work, "N_COMP_TOTAL_MW",
    "Суммарная внутренняя мощность компрессоров\n"
    "от $P_2$ при различных значениях $P_1$",
    r"$N_{мc,вн} + N_{rc,вн}$, МВт",
    "06_Мощность_компрессоров.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="min", contour_format="%.2f",
    cbar_format="%.2f",
    figsize=FIGSIZES["N_comp"],
)

plot_lines(
    d_work, "N_INTERNAL_MW",
    "Внутренняя мощность цикла\n"
    "от $P_2$ при различных значениях $P_1$",
    r"$N_{вн}$, МВт",
    "07_Внутренняя_мощность_цикла.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.2f",
    figsize=FIGSIZES["N_internal"],
)

plot_lines(
    d_work, "N_el_gross_MW",
    "Электрическая мощность брутто\n"
    "от $P_2$ при различных значениях $P_1$",
    r"$N_{эл,брутто}$, МВт",
    "08_Мощность_электрическая_брутто.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.2f",
    figsize=FIGSIZES["N_el"],
)

plot_lines(
    d_work, "N_el_net_MW",
    "Электрическая мощность нетто\n"
    "от $P_2$ при различных значениях $P_1$",
    r"$N_{эл,нетто}$, МВт",
    "09_Мощность_электрическая_нетто.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.2f",
    figsize=FIGSIZES["N_el"],
)

# ============================================================
# 3. Тепловые потоки и расход
# ============================================================
plot_lines(
    d_work, "Q_RECUP_MW",
    "Тепловой поток в рекуператоре\n"
    "от $P_2$ при различных значениях $P_1$",
    r"$Q_{рекуп}$, МВт",
    "10_Тепловой_поток_рекуператора.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.2f",
    figsize=FIGSIZES["Q"],
)

plot_lines(
    d_work, "Q_COOL_MW",
    "Тепловой поток в охладителе\n"
    "от $P_2$ при различных значениях $P_1$",
    r"$Q_{охл}$, МВт",
    "11_Тепловой_поток_охладителя.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="min", contour_format="%.2f",
    cbar_format="%.2f",
    figsize=FIGSIZES["Q"],
)

plot_lines(
    d_work, "G_CO2_kg_s",
    "Расход CO$_2$ в цикле\n"
    "от $P_2$ при различных значениях $P_1$",
    r"$G_{CO_2}$, кг/с",
    "12_Расход_CO2.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.2f",
    figsize=FIGSIZES["G"],
)

print()
print("=" * 70)
print("ПОСТРОЕНИЕ ЛИНЕЙНЫХ ГРАФИКОВ ЗАВЕРШЕНО")
print("=" * 70)
print(f"Папка сохранения: {OUT_DIR}")