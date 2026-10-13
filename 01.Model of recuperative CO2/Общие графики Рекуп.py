import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.interpolate import griddata
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
GRID_SIZE = 300
CONTOUR_LEVELS = 25
CONTOUR_LINES = 15
CONTOUR_LINEWIDTH = 1.6
CONTOUR_LABEL_SIZE = 12
EXTREME_MARKER_SIZE = 140
EXTREME_LABEL_SIZE = 13

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

OUT_DIR = BASE_DIR / "Контурные карты Общие"
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
# КОНТУРНАЯ КАРТА
# ============================================================
def plot_contour(data, value_column, title, colorbar_label, filename,
                 xlabel, ylabel, extreme_type="max",
                 contour_format="%.2f", cbar_format="%.2f",
                 eta_map=False, figsize=(10, 7)):
    P1 = data["P1_MPa"].to_numpy(dtype=float)
    P2 = data["P2_MPa"].to_numpy(dtype=float)
    Z_data = data[value_column].to_numpy(dtype=float)

    xi = np.linspace(P2.min(), P2.max(), GRID_SIZE)
    yi = np.linspace(P1.min(), P1.max(), GRID_SIZE)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((P2, P1), Z_data, (X, Y), method="linear")

    if np.all(np.isnan(Z)):
        raise ValueError(f"Не удалось построить карту для {value_column}.")

    z_min = np.nanmin(Z)
    z_max = np.nanmax(Z)

    if np.isclose(z_min, z_max):
        levels_fill = np.linspace(z_min - 1e-6, z_max + 1e-6, CONTOUR_LEVELS)
        levels_lines = np.linspace(z_min - 1e-6, z_max + 1e-6, CONTOUR_LINES)
    else:
        levels_fill = np.linspace(z_min, z_max, CONTOUR_LEVELS)
        levels_lines = np.linspace(z_min, z_max, CONTOUR_LINES)

    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(X, Y, Z, levels=levels_fill)
    cs = ax.contour(X, Y, Z, levels=levels_lines, colors="black",
                    linewidths=CONTOUR_LINEWIDTH)
    labels = ax.clabel(cs, levels=levels_lines, inline=True, inline_spacing=8,
                       fontsize=CONTOUR_LABEL_SIZE, fmt=contour_format)
    for lab in labels:
        lab.set_bbox(dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=1.5))

    index = None
    if extreme_type == "max":
        index = np.argmax(Z_data)
        extreme_name = "max"
    elif extreme_type == "min":
        index = np.argmin(Z_data)
        extreme_name = "min"

    if index is not None:
        P1_extreme = P1[index]
        P2_extreme = P2[index]
        Z_extreme = Z_data[index]

        ax.scatter(P2_extreme, P1_extreme, s=EXTREME_MARKER_SIZE, marker="*",
                   color="white", edgecolors="black", linewidths=1.5, zorder=10)

        extreme_text = (f"{extreme_name} = {contour_format % Z_extreme}\n"
                        f"$P_1$ = {P1_extreme:.2f} МПа\n"
                        f"$P_2$ = {P2_extreme:.2f} МПа")
        ax.text(0.98, 0.05, extreme_text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=EXTREME_LABEL_SIZE,
                bbox=dict(facecolor="white", edgecolor="black",
                          alpha=0.9, pad=4), zorder=11)

    ax.set_xlabel(xlabel, fontsize=FONT_SIZE, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE, labelpad=8)
    ax.set_title(title, fontsize=FONT_SIZE, pad=18)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(colorbar_label, fontsize=FONT_SIZE, labelpad=10)
    cbar.ax.tick_params(labelsize=FONT_SIZE)

    if eta_map:
        cbar.ax.yaxis.set_major_locator(MultipleLocator(ETA_TICK_STEP))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(cbar_format))

    plt.tight_layout()
    out_path = OUT_DIR / filename
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.show()
    plt.close()

    print()
    print("=" * 70)
    print(title.replace("\n", " "))
    print("=" * 70)
    print(f"Количество точек: {len(data)}")
    print(f"Диапазон P2: {P2.min():.3f} – {P2.max():.3f} МПа")
    print(f"Диапазон P1: {P1.min():.3f} – {P1.max():.3f} МПа")
    print(f"Диапазон {value_column}: {z_min:.3f} – {z_max:.3f}")
    if index is not None:
        print()
        print(f"{extreme_name} = {contour_format % Z_extreme}")
        print(f"P1 = {P1_extreme:.3f} МПа")
        print(f"P2 = {P2_extreme:.3f} МПа")
    print(f"Файл: {out_path}")

plot_contour(
    d_work, "eta_el_net_plant_%",
    "Зависимость электрического КПД нетто АСММ\n"
    "от давлений $P_1$ и $P_2$",
    r"$\eta_{нетто}^{АСММ}$, %",
    "01_КПД_нетто_АСММ.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.0f", eta_map=True,
    figsize=FIGSIZES["eta"],
)

plot_contour(
    d_work, "eta_el_net_CO2_%",
    "Зависимость электрического КПД нетто CO$_2$-цикла\n"
    "от давлений $P_1$ и $P_2$",
    r"$\eta_{нетто}^{CO_2}$, %",
    "02_КПД_нетто_CO2.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.0f", eta_map=True,
    figsize=FIGSIZES["eta"],
)

plot_contour(
    d_work, "eta_cycle_CO2_%",
    "Зависимость КПД цикла CO$_2$\n"
    "от давлений $P_1$ и $P_2$",
    r"$\eta_{цикл}^{CO_2}$, %",
    "03_КПД_цикла_CO2.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.0f", eta_map=True,
    figsize=FIGSIZES["eta"],
)

plot_contour(
    d_work, "eta_plant_%",
    "Зависимость КПД установки\n"
    "от давлений $P_1$ и $P_2$",
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
plot_contour(
    d_work, "N_TURB_i_MW",
    "Внутренняя мощность турбины\n"
    "от давлений $P_1$ и $P_2$",
    r"$N_{т,вн}$, МВт",
    "05_Мощность_турбины.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.2f",
    figsize=FIGSIZES["N_turb"],
)

plot_contour(
    d_work, "N_COMP_TOTAL_MW",
    "Суммарная внутренняя мощность компрессоров\n"
    "от давлений $P_1$ и $P_2$",
    r"$N_{мc,вн} + N_{rc,вн}$, МВт",
    "06_Мощность_компрессоров.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="min", contour_format="%.2f",
    cbar_format="%.2f",
    figsize=FIGSIZES["N_comp"],
)

plot_contour(
    d_work, "N_INTERNAL_MW",
    "Внутренняя мощность цикла\n"
    "от давлений $P_1$ и $P_2$",
    r"$N_{вн}$, МВт",
    "07_Внутренняя_мощность_цикла.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.2f",
    figsize=FIGSIZES["N_internal"],
)

plot_contour(
    d_work, "N_el_gross_MW",
    "Электрическая мощность брутто\n"
    "от давлений $P_1$ и $P_2$",
    r"$N_{эл,брутто}$, МВт",
    "08_Мощность_электрическая_брутто.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.2f",
    figsize=FIGSIZES["N_el"],
)

plot_contour(
    d_work, "N_el_net_MW",
    "Электрическая мощность нетто\n"
    "от давлений $P_1$ и $P_2$",
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
plot_contour(
    d_work, "Q_RECUP_MW",
    "Тепловой поток в рекуператоре\n"
    "от давлений $P_1$ и $P_2$",
    r"$Q_{рекуп}$, МВт",
    "10_Тепловой_поток_рекуператора.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.2f",
    figsize=FIGSIZES["Q"],
)

plot_contour(
    d_work, "Q_COOL_MW",
    "Тепловой поток в охладителе\n"
    "от давлений $P_1$ и $P_2$",
    r"$Q_{охл}$, МВт",
    "11_Тепловой_поток_охладителя.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="min", contour_format="%.2f",
    cbar_format="%.2f",
    figsize=FIGSIZES["Q"],
)

plot_contour(
    d_work, "G_CO2_kg_s",
    "Расход CO$_2$ в цикле\n"
    "от давлений $P_1$ и $P_2$",
    r"$G_{CO_2}$, кг/с",
    "12_Расход_CO2.png",
    xlabel=r"$P_2$, МПа", ylabel=r"$P_1$, МПа",
    extreme_type="max", contour_format="%.2f",
    cbar_format="%.2f",
    figsize=FIGSIZES["G"],
)

print()
print("=" * 70)
print("ПОСТРОЕНИЕ КАРТ ЗАВЕРШЕНО")
print("=" * 70)
print(f"Папка сохранения: {OUT_DIR}")