import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.interpolate import griddata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "sensitivity_CO2_recuperative_cycle.xlsx"

FONT_NAME = "Times New Roman"
FONT_SIZE = 16
FIGSIZE = (10, 7)
DPI = 250
GRID_SIZE = 300
CONTOUR_LEVELS = 25
CONTOUR_LINES = 15
CONTOUR_LINEWIDTH = 1.6
CONTOUR_LABEL_SIZE = 12
EXTREME_MARKER_SIZE = 140
EXTREME_LABEL_SIZE = 13
LINE_WIDTH = 2.5
MEAN_MARKER_SIZE = 7

plt.rcParams["font.family"] = FONT_NAME
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["mathtext.fontset"] = "stix"   # математика в стиле Times

OUT_DIR = BASE_DIR / "Контурные карты"
OUT_DIR.mkdir(exist_ok=True)

if not EXCEL_FILE.exists():
    raise FileNotFoundError(f"Не найден Excel-файл:\n{EXCEL_FILE}")

df_eta = pd.read_excel(EXCEL_FILE, sheet_name="Results_eta")
df_dp = pd.read_excel(EXCEL_FILE, sheet_name="Results_pressure")
df_dt = pd.read_excel(EXCEL_FILE, sheet_name="Results_dT")

df_eta = df_eta[df_eta["physical_root"] == True].copy()
df_dp = df_dp[df_dp["physical_root"] == True].copy()
df_dt = df_dt[df_dt["physical_root"] == True].copy()

def plot_contour(data, x_col, y_col, value_column, title,
                 colorbar_label, filename, xlabel, ylabel,
                 extreme_type="max", contour_format="%.2f",
                 extreme_format=None, cbar_format="%.2f"):
    if extreme_format is None:
        extreme_format = contour_format

    X_data = data[x_col].to_numpy(dtype=float)
    Y_data = data[y_col].to_numpy(dtype=float)
    Z_data = data[value_column].to_numpy(dtype=float)

    xi = np.linspace(X_data.min(), X_data.max(), GRID_SIZE)
    yi = np.linspace(Y_data.min(), Y_data.max(), GRID_SIZE)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((X_data, Y_data), Z_data, (X, Y), method="linear")

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

    fig, ax = plt.subplots(figsize=FIGSIZE)
    cf = ax.contourf(X, Y, Z, levels=levels_fill)
    cs = ax.contour(X, Y, Z, levels=levels_lines, colors="black",
                    linewidths=CONTOUR_LINEWIDTH)
    labels = ax.clabel(cs, levels=levels_lines, inline=True,
                       inline_spacing=8, fontsize=CONTOUR_LABEL_SIZE,
                       fmt=contour_format)

    for label in labels:
        label.set_bbox(dict(facecolor="white", edgecolor="none",
                            alpha=0.85, pad=1.5))

    index = None
    if extreme_type == "max":
        index = np.argmax(Z_data)
        extreme_name = "max"
    elif extreme_type == "min":
        index = np.argmin(Z_data)
        extreme_name = "min"

    if index is not None:
        x_extreme = X_data[index]
        y_extreme = Y_data[index]
        Z_extreme = Z_data[index]

        ax.scatter(x_extreme, y_extreme, s=EXTREME_MARKER_SIZE, marker="*",
                   color="white", edgecolors="black", linewidths=1.5, zorder=10)

        # Для подписи экстремума берём компактные обозначения
        x_name = xlabel.split(",")[0]
        y_name = ylabel.split(",")[0]
        extreme_text = (f"{extreme_name} = {extreme_format % Z_extreme}\n"
                        f"{x_name} = {x_extreme:.2f}\n"
                        f"{y_name} = {y_extreme:.2f}")
        ax.text(0.98, 0.05, extreme_text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=EXTREME_LABEL_SIZE,
                bbox=dict(facecolor="white", edgecolor="black",
                          alpha=0.9, pad=4), zorder=11)

    ax.set_xlabel(xlabel, fontsize=FONT_SIZE, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE, labelpad=8)

    # Заголовок с полным названием — поднят выше
    ax.set_title(title, fontsize=FONT_SIZE, pad=18)

    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.2)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(colorbar_label, fontsize=FONT_SIZE, labelpad=10)
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(cbar_format))

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=DPI, bbox_inches="tight")
    plt.show()
    plt.close()

    print(title)
    if index is not None:
        print(f"{extreme_name} = {extreme_format % Z_extreme}; "
              f"{x_name} = {x_extreme:.2f}; "
              f"{y_name} = {y_extreme:.2f}")
    print(f"Диапазон: {z_min:.2f}–{z_max:.2f}")
    print(f"Файл: {OUT_DIR / filename}")
    print()

# ============================================================
# ЛИНЕЙНЫЙ ГРАФИК ΔTMIN
# ============================================================

def plot_dTmin_linear(data, filename):
    data = data.sort_values("dTmin_target_K")

    stats = (data.groupby("dTmin_target_K").agg(
        eta_mean=("eta_el_net_plant_%", "mean"),
        eta_min=("eta_el_net_plant_%", "min"),
        eta_max=("eta_el_net_plant_%", "max"),
        n=("eta_el_net_plant_%", "count")
    ).reset_index().sort_values("dTmin_target_K"))

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.fill_between(stats["dTmin_target_K"], stats["eta_min"], stats["eta_max"],
                    alpha=0.15, label="Диапазон КПД")
    ax.plot(stats["dTmin_target_K"], stats["eta_mean"],
            linewidth=LINE_WIDTH, marker="o", markersize=MEAN_MARKER_SIZE,
            label=r"Среднее значение $\eta_{нетто}^{АСММ}$", zorder=5)

    imax = stats["eta_mean"].idxmax()
    best = stats.loc[imax]

    ax.scatter(best["dTmin_target_K"], best["eta_mean"],
               s=EXTREME_MARKER_SIZE, marker="*", color="white",
               edgecolors="black", linewidths=1.5, zorder=10)

    best_text = (f"Максимум среднего КПД\n"
                 f"$\\Delta T_{{min}}$ = {best['dTmin_target_K']:.1f} K\n"
                 f"$\\eta_{{нетто}}^{{АСММ}}$ = {best['eta_mean']:.2f} %\n"
                 f"min = {best['eta_min']:.2f} %\n"
                 f"max = {best['eta_max']:.2f} %")
    ax.text(0.98, 0.05, best_text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=12,
            bbox=dict(facecolor="white", edgecolor="black",
                      alpha=0.9, pad=5), zorder=11)

    ax.set_xlabel(r"$\Delta T_{min}$, K", fontsize=FONT_SIZE, labelpad=8)
    ax.set_ylabel(r"$\eta_{нетто}^{АСММ}$, %", fontsize=FONT_SIZE, labelpad=8)

    # Заголовок с полным названием — поднят выше
    ax.set_title("Влияние минимального температурного напора рекуператора\n"
                 "на электрический КПД нетто АСММ",
                 fontsize=FONT_SIZE, pad=18)

    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=12, loc="best")

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=DPI, bbox_inches="tight")
    plt.show()
    plt.close()

    print("Влияние минимального температурного напора на электрический КПД нетто АСММ")
    print(f"Максимум среднего: ΔTmin = {best['dTmin_target_K']:.1f} K; "
          f"η = {best['eta_mean']:.2f} %")
    print(f"Диапазон: {stats['eta_min'].min():.2f}–{stats['eta_max'].max():.2f}")
    print(f"Файл: {OUT_DIR / filename}")
    print()

# ============================================================
# ПОСТРОЕНИЕ
# ============================================================

# 1. Карта: η_нетто^АСММ от η_oi^T × η_oi^К
plot_contour(
    df_eta,
    x_col="eta_T_%",
    y_col="eta_C_%",
    value_column="eta_el_net_plant_%",
    title="Зависимость электрического КПД нетто АСММ\n"
          "от внутренних относительных КПД турбины и компрессора",
    colorbar_label=r"$\eta_{нетто}^{АСММ}$, %",
    filename="01_КПД_нетто_АСММ_etaT_etaC.png",
    xlabel=r"$\eta_{oi}^{T}$, %",
    ylabel=r"$\eta_{oi}^{К}$, %",
    extreme_type="max",
    contour_format="%.2f",
    cbar_format="%.0f"
)

# 2. Карта: η_нетто^АСММ от ΔP_гор × ΔP_хол
plot_contour(
    df_dp,
    x_col="DP_rec_hot_%",
    y_col="DP_rec_cold_%",
    value_column="eta_el_net_plant_%",
    title="Зависимость электрического КПД нетто АСММ\n"
          "от относительных потерь давления в рекуператорах",
    colorbar_label=r"$\eta_{нетто}^{АСММ}$, %",
    filename="02_КПД_нетто_АСММ_dP.png",
    xlabel=r"$\Delta P_{гор}$, %",
    ylabel=r"$\Delta P_{хол}$, %",
    extreme_type="max",
    contour_format="%.2f",
    cbar_format="%.0f"
)

# 3. Линейный график: η_нетто^АСММ от ΔT_min
plot_dTmin_linear(df_dt, "03_Влияние_dTmin.png")

print(f"Готово. Файлы сохранены в: {OUT_DIR}")