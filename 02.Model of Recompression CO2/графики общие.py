import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.interpolate import griddata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "sCO2_TR2_550_transport_loss2.xlsx"
SHEET_NAME = "Results"
X_RECOMP = 0.7
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
SCATTER_SIZE = 35
SCATTER_ALPHA = 0.55
MEAN_MARKER_SIZE = 7
LINE_WIDTH = 2.5

# РАЗМЕРЫ ГРАФИКОВ
FIGSIZES = {
    "eta_net":       (10, 7),   # КПД нетто АСММ
    "eta_gross":     (10, 7),   # КПД брутто
    "N_turb":        (10, 5),   # мощность турбины
    "N_comp":        (10, 5),   # мощность компрессоров
    "N_internal":    (10, 5),   # внутренняя мощность цикла
    "N_el_gross":    (10, 5),   # эл. мощность брутто
    "N_el_net":      (10, 5),   # эл. мощность нетто
    "recompression": (10, 7),   # график степени рекомпрессии
}

plt.rcParams["font.family"] = FONT_NAME
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["mathtext.fontset"] = "stix"

OUT_DIR = BASE_DIR / "Контурные карты"
OUT_DIR.mkdir(exist_ok=True)

if not EXCEL_FILE.exists():
    raise FileNotFoundError(f"Не найден Excel-файл:\n{EXCEL_FILE}")

df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)

required_columns = ["P1_MPa", "P7_MPa", "x", "eta_cycle_%", "eta_el_gross_%", "eta_el_net_%", "N_TURB_i_MW", "N_MCOMP_i_MW", "N_RCOMP_i_MW", "N_el_gross_MW", "N_el_net_MW"]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError("В Excel отсутствуют столбцы:\n" + "\n".join(missing))
d = df.copy()
if "physical_root" in d.columns:
    d = d[d["physical_root"] == True].copy()

d = d.dropna(subset=required_columns).copy()
d["N_COMP_TOTAL_MW"] = d["N_MCOMP_i_MW"] + d["N_RCOMP_i_MW"]
d["N_INTERNAL_MW"] = d["N_TURB_i_MW"] - d["N_COMP_TOTAL_MW"]

d_fixed = d[np.isclose(d["x"], X_RECOMP, atol=1e-8)].copy()
d_work = d_fixed[d_fixed["eta_el_net_%"] >= ETA_MIN].copy()

if len(d_fixed) < 3:
    raise ValueError(f"Для x = {X_RECOMP} найдено недостаточно расчётных точек.")

if len(d_work) < 3:
    raise ValueError(f"Для x = {X_RECOMP} после ограничения ηэл,нетто >= {ETA_MIN}% осталось недостаточно точек.")

d_recomp = d[d["eta_el_net_%"] >= ETA_MIN].copy()

if len(d_recomp) < 3:
    raise ValueError(f"Для анализа степени рекомпрессии после ограничения ηэл,нетто >= {ETA_MIN}% осталось недостаточно точек.")

def plot_contour(data, value_column, title, colorbar_label, filename,
                 extreme_type=None, contour_format="%.2f",
                 eta_map=False, extreme_format=None, cbar_format="%.2f",
                 figsize=(10, 7)):
    if extreme_format is None:
        extreme_format = contour_format

    P1 = data["P1_MPa"].to_numpy(dtype=float)
    P7 = data["P7_MPa"].to_numpy(dtype=float)
    Z_data = data[value_column].to_numpy(dtype=float)
    xi = np.linspace(P7.min(), P7.max(), GRID_SIZE)
    yi = np.linspace(P1.min(), P1.max(), GRID_SIZE)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((P7, P1), Z_data, (X, Y), method="linear")

    if np.all(np.isnan(Z)):
        raise ValueError(f"Не удалось построить контурную карту для {value_column}.")

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
    cs = ax.contour(X, Y, Z, levels=levels_lines, colors="black", linewidths=CONTOUR_LINEWIDTH)
    labels = ax.clabel(cs, levels=levels_lines, inline=True, inline_spacing=8,
                       fontsize=CONTOUR_LABEL_SIZE, fmt=contour_format)

    for label in labels:
        label.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))

    index = None

    if extreme_type == "max":
        index = np.argmax(Z_data)
        extreme_name = "max"
    elif extreme_type == "min":
        index = np.argmin(Z_data)
        extreme_name = "min"

    if index is not None:
        P1_extreme = P1[index]
        P7_extreme = P7[index]
        Z_extreme = Z_data[index]
        ax.scatter(P7_extreme, P1_extreme, s=EXTREME_MARKER_SIZE, marker="*",
                   color="white", edgecolors="black", linewidths=1.5, zorder=10)
        extreme_text = (f"{extreme_name} = {extreme_format % Z_extreme}\n"
                        f"$P_1$ = {P1_extreme:.2f} МПа\n"
                        f"$P_2$ = {P7_extreme:.2f} МПа")
        ax.text(0.98, 0.05, extreme_text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=EXTREME_LABEL_SIZE,
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.9, pad=4),
                zorder=11)

    ax.set_xlabel(r"$P_2$, МПа", fontsize=FONT_SIZE, labelpad=8)
    ax.set_ylabel(r"$P_1$, МПа", fontsize=FONT_SIZE, labelpad=8)
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
    plt.savefig(OUT_DIR / filename, dpi=DPI, bbox_inches="tight")
    plt.show()
    plt.close()

    print(title)

    if index is not None:
        print(f"{extreme_name} = {extreme_format % Z_extreme}; "
              f"P1 = {P1_extreme:.2f} МПа; P2 = {P7_extreme:.2f} МПа")

    print(f"Диапазон: {z_min:.2f}–{z_max:.2f}")
    print(f"Файл: {OUT_DIR / filename}")

def plot_recompression_degree(data, filename, figsize=(11, 7)):
    data = data.copy()
    data["expansion_ratio"] = data["P1_MPa"] / data["P7_MPa"]

    stats = (data.groupby("x").agg(eta_mean=("eta_el_net_%", "mean"), eta_min=("eta_el_net_%", "min"), eta_max=("eta_el_net_%", "max"), eta_std=("eta_el_net_%", "std"), P1_min=("P1_MPa", "min"), P1_max=("P1_MPa", "max"), P7_min=("P7_MPa", "min"), P7_max=("P7_MPa", "max"), n=("eta_el_net_%", "count")).reset_index().sort_values("x"))

    ratio_min = data["expansion_ratio"].min()
    ratio_max = data["expansion_ratio"].max()
    norm = Normalize(vmin=ratio_min, vmax=ratio_max)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=figsize)
    colors = cmap(norm(data["expansion_ratio"]))

    ax.scatter(data["x"], data["eta_el_net_%"], c=colors, s=SCATTER_SIZE, alpha=SCATTER_ALPHA, edgecolors="none", zorder=2)
    ax.fill_between(stats["x"], stats["eta_min"], stats["eta_max"], alpha=0.15, label="Диапазон КПД")
    ax.plot(stats["x"], stats["eta_mean"], linewidth=LINE_WIDTH, marker="o", markersize=MEAN_MARKER_SIZE,
            label=r"Среднее значение $\eta_{нетто}^{АСММ}$", zorder=5)

    for _, row in stats.iterrows():
        ax.annotate(f"{row['eta_mean']:.2f}%", (row["x"], row["eta_mean"]), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=12, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2), zorder=6)

    imax = stats["eta_mean"].idxmax()
    best = stats.loc[imax]

    ax.scatter(best["x"], best["eta_mean"], s=EXTREME_MARKER_SIZE, marker="*", color="white", edgecolors="black", linewidths=1.5, zorder=10)

    best_text = (f"Максимум среднего КПД\n"
                 f"$x$ = {best['x']:.2f}\n"
                 f"$\\eta_{{нетто}}^{{АСММ}}$ = {best['eta_mean']:.2f} %\n"
                 f"$P_1$ = {best['P1_min']:.2f}–{best['P1_max']:.2f} МПа\n"
                 f"$P_2$ = {best['P7_min']:.2f}–{best['P7_max']:.2f} МПа")

    ax.text(0.98, 0.05, best_text, transform=ax.transAxes, ha="right", va="bottom", fontsize=12, bbox=dict(facecolor="white", edgecolor="black", alpha=0.9, pad=5), zorder=11)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"Степень расширения $\pi_t = P_1/P_2$", fontsize=FONT_SIZE, labelpad=10)
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.set_xlabel("Доля рекомпрессии $x$", fontsize=FONT_SIZE, labelpad=8)
    ax.set_ylabel(r"$\eta_{нетто}^{АСММ}$, %", fontsize=FONT_SIZE, labelpad=8)
    ax.set_title("Влияние доли рекомпрессии на электрический КПД нетто АСММ",
                 fontsize=FONT_SIZE, pad=18)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.yaxis.set_major_locator(MultipleLocator(ETA_TICK_STEP))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=12, loc="best")
    ax.set_xlim(stats["x"].min() - 0.03, stats["x"].max() + 0.03)

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=DPI, bbox_inches="tight")
    plt.show()
    plt.close()

    print()
    print("=" * 80)
    print("ВЛИЯНИЕ ДОЛИ РЕКОМПРЕССИИ НА КПД НЕТТО")
    print("=" * 80)
    print()
    print(stats[["x", "eta_mean", "eta_min", "eta_max", "eta_std", "P1_min", "P1_max", "P7_min", "P7_max", "n"]].rename(columns={"P7_min": "P2_min", "P7_max": "P2_max"}).to_string(index=False, formatters={"x": "{:.2f}".format, "eta_mean": "{:.2f}".format, "eta_min": "{:.2f}".format, "eta_max": "{:.2f}".format, "eta_std": "{:.2f}".format, "P1_min": "{:.2f}".format, "P1_max": "{:.2f}".format, "P2_min": "{:.2f}".format, "P2_max": "{:.2f}".format}))
    print()
    print("=" * 80)
    print("МАКСИМАЛЬНОЕ СРЕДНЕЕ ЗНАЧЕНИЕ")
    print("=" * 80)
    print()
    print(f"x = {best['x']:.2f}")
    print(f"Средний eta_нетто^АСММ = {best['eta_mean']:.2f} %")
    print(f"Минимальный eta_нетто^АСММ = {best['eta_min']:.2f} %")
    print(f"Максимальный eta_нетто^АСММ = {best['eta_max']:.2f} %")
    print()
    print(f"Диапазон P1: {best['P1_min']:.2f}–{best['P1_max']:.2f} МПа")
    print(f"Диапазон P2: {best['P7_min']:.2f}–{best['P7_max']:.2f} МПа")
    print()
    print(f"Диапазон степени расширения во всей выборке: {ratio_min:.2f}–{ratio_max:.2f}")
    print()
    print(f"Файл: {OUT_DIR / filename}")

# КПД — квадратный формат (10, 7)
plot_contour(d_work, "eta_el_net_%",
             "Зависимость электрического КПД нетто АСММ\nот давлений $P_1$ и $P_2$ при $x = 0{,}7$",
             r"$\eta_{нетто}^{АСММ}$, %", "01_КПД_нетто_АСММ.png",
             "max", "%.2f", True, cbar_format="%.0f",
             figsize=FIGSIZES["eta_net"])

plot_contour(d_work, "eta_el_gross_%",
             "Зависимость электрического КПД брутто\nот давлений $P_1$ и $P_2$ при $x = 0{,}7$",
             r"$\eta_{брутто}$, %", "02_КПД_брутто.png",
             "max", "%.2f", True, cbar_format="%.0f",
             figsize=FIGSIZES["eta_gross"])

# Мощности — широкий формат (12, 5)
plot_contour(d_work, "N_TURB_i_MW",
             "Внутренняя мощность турбины при $x = 0{,}7$",
             r"$N_{т,вн}$, МВт", "03_Мощность_турбины.png",
             "max", "%.2f", cbar_format="%.2f",
             figsize=FIGSIZES["N_turb"])

plot_contour(d_work, "N_COMP_TOTAL_MW",
             "Суммарная внутренняя мощность компрессоров при $x = 0{,}7$",
             r"$N_{мc,вн} + N_{rc,вн}$, МВт", "04_Мощность_компрессоров.png",
             "min", "%.2f", cbar_format="%.2f",
             figsize=FIGSIZES["N_comp"])

plot_contour(d_work, "N_INTERNAL_MW",
             "Внутренняя мощность цикла при $x = 0{,}7$",
             r"$N_{вн}$, МВт", "05_Внутренняя_мощность_цикла.png",
             "max", "%.2f", cbar_format="%.2f",
             figsize=FIGSIZES["N_internal"])

plot_contour(d_work, "N_el_gross_MW",
             "Электрическая мощность брутто при $x = 0{,}7$",
             r"$N_{эл,брутто}$, МВт", "06_Мощность_электрическая_брутто.png",
             "max", "%.2f", cbar_format="%.2f",
             figsize=FIGSIZES["N_el_gross"])

plot_contour(d_work, "N_el_net_MW",
             "Электрическая мощность нетто при $x = 0{,}7$",
             r"$N_{эл,нетто}$, МВт", "07_Мощность_электрическая_нетто.png",
             "max", "%.2f", cbar_format="%.2f",
             figsize=FIGSIZES["N_el_net"])

# График рекомпрессии — (11, 7)
plot_recompression_degree(d_recomp, "08_Степень_рекомпрессии.png",
                          figsize=FIGSIZES["recompression"])

print()
print(f"Готово. Карты сохранены в: {OUT_DIR}")