import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import griddata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "Графики чувствительности"
OUT_DIR.mkdir(exist_ok=True)

PATH = "Чувствительность Рекомпрессия.xlsx"

FONT_NAME = "Times New Roman"
FONT_SIZE = 16
DPI = 500
CONTOUR_LEVELS = 25
CONTOUR_LINES = 15
CONTOUR_LINEWIDTH = 1.6
CONTOUR_LABEL_SIZE = 12
MAX_MARKER_SIZE = 140
MAX_LABEL_SIZE = 13
ETA_MIN = None

# Шаг шкалы КПД (colorbar и ось Y)
ETA_TICK_STEP = 1   # 0.25, 0.5, 1, 2, 5

# ============================================================
# РАЗМЕРЫ ГРАФИКОВ

FIGSIZES = {
    "kpd_comp_turb":  (10, 7),   # карта КПД от η компрессора × η турбины
    "dt_contour":     (10, 7),   # карта КПД от ΔT_HTHE × ΔT_LTHE
    "dt_line":        (10, 5),   # одномерный график по ΔT
    "dt_family":      (12, 5),   # семейство кривых ΔT
    "dp_contour":     (10, 7),   # карта КПД от ΔP_cold × ΔP_hot
    "dp_line":        (10, 5),   # одномерный график по ΔP
}

plt.rcParams["font.family"] = FONT_NAME
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["mathtext.fontset"] = "stix"

# ============================================================
# СПИСОК СОЗДАННЫХ ФАЙЛОВ
# ============================================================
CREATED_FILES = []

# ============================================================
# ЧТЕНИЕ ФАЙЛА
# ============================================================
xls = pd.ExcelFile(PATH)
print()
print("=" * 70)
print("ЛИСТЫ EXCEL")
print("=" * 70)
print(xls.sheet_names)

# ============================================================
# ФИЛЬТРАЦИЯ ФИЗИЧНЫХ ТОЧЕК
# ============================================================
def filter_physical(d):
    if "physical_root" in d.columns:
        d = d[d["physical_root"] == True].copy()
    if "root_success" in d.columns:
        d = d[d["root_success"] == True].copy()
    if ETA_MIN is not None and "eta_el_net_%" in d.columns:
        d = d[d["eta_el_net_%"] >= ETA_MIN].copy()
    return d

# ============================================================
# ШКАЛА COLORBAR
# ============================================================
def set_colorbar_ticks(cbar, step=ETA_TICK_STEP):
    clim_lo, clim_hi = cbar.ax.get_ylim()
    tick_lo = np.ceil(clim_lo / step) * step
    tick_hi = np.floor(clim_hi / step) * step
    if tick_hi <= tick_lo:
        tick_lo = np.floor(clim_lo / step) * step
        tick_hi = np.ceil(clim_hi / step) * step
    ticks = np.arange(tick_lo, tick_hi + step * 0.5, step)
    if len(ticks) > 0:
        cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))

# ============================================================
# ШКАЛА ОСИ Y
# ============================================================
def set_axis_y_ticks(ax, step=ETA_TICK_STEP):
    y_lo, y_hi = ax.get_ylim()
    tick_lo = np.ceil(y_lo / step) * step
    tick_hi = np.floor(y_hi / step) * step
    if tick_hi <= tick_lo:
        tick_lo = np.floor(y_lo / step) * step
        tick_hi = np.ceil(y_hi / step) * step
    ticks = np.arange(tick_lo, tick_hi + step * 0.5, step)
    if len(ticks) > 0:
        ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))

# ============================================================
# КОНТУРНЫЙ ГРАФИК (2D)
# ============================================================
def plot_contour_map(df, x_column, y_column, z_column,
                     xlabel, ylabel, title, colorbar_label, filename,
                     show_max=True, figsize=(10, 7)):
    required = [x_column, y_column, z_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("В таблице отсутствуют столбцы:\n" + "\n".join(missing))

    d = filter_physical(df)
    d = d.dropna(subset=[x_column, y_column, z_column])
    if len(d) < 3:
        raise ValueError(f"Недостаточно точек для карты: {filename}")

    x = d[x_column].to_numpy(dtype=float)
    y = d[y_column].to_numpy(dtype=float)
    z = d[z_column].to_numpy(dtype=float)

    xi = np.linspace(x.min(), x.max(), 300)
    yi = np.linspace(y.min(), y.max(), 300)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method="linear")

    z_min = np.nanmin(Z)
    z_max = np.nanmax(Z)

    levels_fill = np.linspace(z_min, z_max, CONTOUR_LEVELS)
    levels_lines = np.linspace(z_min, z_max, CONTOUR_LINES)

    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(X, Y, Z, levels=levels_fill)
    cs = ax.contour(X, Y, Z, levels=levels_lines, colors="black",
                    linewidths=CONTOUR_LINEWIDTH)
    labels = ax.clabel(cs, levels=levels_lines, inline=True, inline_spacing=8,
                       fontsize=CONTOUR_LABEL_SIZE, fmt="%.2f")
    for lab in labels:
        lab.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))

    if show_max:
        imax = np.argmax(z)
        x_max = x[imax]
        y_max = y[imax]
        z_max_point = z[imax]
        ax.scatter(x_max, y_max, s=MAX_MARKER_SIZE, marker="*", color="white",
                   edgecolors="black", linewidths=1.5, zorder=10)
        ax.annotate(
            f"max = {z_max_point:.3f} %\n"
            f"{xlabel.split(',')[0]} = {x_max:.3f}\n"
            f"{ylabel.split(',')[0]} = {y_max:.3f}",
            xy=(0.98, 0.02), xycoords="axes fraction",
            ha="right", va="bottom",
            fontsize=MAX_LABEL_SIZE,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.9, pad=4),
            zorder=11,
        )

    ax.set_xlabel(xlabel, fontsize=FONT_SIZE, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE, labelpad=8)
    ax.set_title(title, fontsize=FONT_SIZE, pad=18)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.2)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(colorbar_label, fontsize=FONT_SIZE, labelpad=10)
    cbar.ax.tick_params(labelsize=FONT_SIZE)

    cbar.mappable.set_clim(z_min, z_max)
    set_colorbar_ticks(cbar, step=ETA_TICK_STEP)

    plt.tight_layout()
    out_path = OUT_DIR / filename
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.show()
    CREATED_FILES.append(out_path)

    print()
    print("=" * 70)
    print(title.replace("\n", " "))
    print("=" * 70)
    print(f"Количество точек: {len(d)}")
    print(f"Диапазон X: {x.min():.4f} – {x.max():.4f}")
    print(f"Диапазон Y: {y.min():.4f} – {y.max():.4f}")
    print(f"Диапазон Z: {z.min():.4f} – {z.max():.4f} %")
    if show_max:
        print()
        print(f"Максимум КПД = {z_max_point:.6f} %")
        print(f"{xlabel.split(',')[0]} = {x_max:.4f}")
        print(f"{ylabel.split(',')[0]} = {y_max:.4f}")
    print(f"Файл: {out_path}")

# ============================================================
# ОДНОМЕРНЫЙ ГРАФИК
# ============================================================
def plot_1d_sensitivity(df, x_column, z_column,
                        xlabel, ylabel, title, filename,
                        diagonal_column=None,
                        fixed_column=None, fixed_value=None,
                        show_max=True, marker="o", figsize=(10, 5)):
    required = [x_column, z_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("В таблице отсутствуют столбцы:\n" + "\n".join(missing))

    d = filter_physical(df)
    d = d.dropna(subset=[x_column, z_column])

    if diagonal_column is not None:
        d = d[np.isclose(d[x_column], d[diagonal_column])].copy()
        d = d.sort_values(by=x_column)
    elif fixed_column is not None and fixed_value is not None:
        d = d[np.isclose(d[fixed_column], fixed_value)].copy()
        d = d.sort_values(by=x_column)
    else:
        grouped = d.groupby(x_column)[z_column].mean().reset_index()
        d = grouped.sort_values(by=x_column)

    if len(d) < 3:
        raise ValueError(f"Недостаточно точек для графика: {filename}")

    x = d[x_column].to_numpy(dtype=float)
    z = d[z_column].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, z, marker=marker, linewidth=1.8, label=ylabel)

    if show_max:
        imax = int(np.argmax(z))
        x_max = x[imax]
        z_max_point = z[imax]
        ax.scatter(x_max, z_max_point, s=MAX_MARKER_SIZE, marker="*",
                   color="white", edgecolors="black", linewidths=1.5, zorder=10)
        ax.annotate(
            f"max = {z_max_point:.3f} %\n{xlabel.split(',')[0]} = {x_max:.3f}",
            xy=(0.98, 0.02), xycoords="axes fraction",
            ha="right", va="bottom",
            fontsize=MAX_LABEL_SIZE,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.9, pad=4),
            zorder=11,
        )

    ax.set_xlabel(xlabel, fontsize=FONT_SIZE, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE, labelpad=8)
    ax.set_title(title, fontsize=FONT_SIZE, pad=18)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE - 3)

    set_axis_y_ticks(ax, step=ETA_TICK_STEP)

    plt.tight_layout()
    out_path = OUT_DIR / filename
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.show()
    CREATED_FILES.append(out_path)

    print()
    print("=" * 70)
    print(title.replace("\n", " "))
    print("=" * 70)
    print(f"Количество точек: {len(d)}")
    print(f"Диапазон X: {x.min():.4f} – {x.max():.4f}")
    print(f"Диапазон Z: {z.min():.4f} – {z.max():.4f} %")
    if show_max:
        print()
        print(f"Максимум КПД = {z_max_point:.6f} %")
        print(f"{xlabel.split(',')[0]} = {x_max:.4f}")
    print(f"Файл: {out_path}")

# ============================================================
# 1. КПД компрессора / турбины
# ============================================================
df_kpd = pd.read_excel(PATH, sheet_name="KPD_comp_turb")
plot_contour_map(
    df=df_kpd,
    x_column="KPDcomp_%",
    y_column="KPDturb_%",
    z_column="eta_el_net_%",
    xlabel=r"$\eta_{oi}^{К}$, %",
    ylabel=r"$\eta_{oi}^{T}$, %",
    title="Зависимость электрического КПД нетто АСММ\n"
          "от внутренних относительных КПД компрессора и турбины",
    colorbar_label=r"$\eta_{нетто}^{АСММ}$, %",
    filename="KPD_comp_turb.png",
    figsize=FIGSIZES["kpd_comp_turb"],
)

# ============================================================
# 2. Недогрев HTHE / LTHE
# ============================================================
df_dt = pd.read_excel(PATH, sheet_name="dT_HTHE_LTHE")

# 2а. Контурный график
plot_contour_map(
    df=df_dt,
    x_column="dT_HTHE_K",
    y_column="dT_LTHE_K",
    z_column="eta_el_net_%",
    xlabel=r"$\Delta T_{HTHE}$, K",
    ylabel=r"$\Delta T_{LTHE}$, K",
    title="Зависимость электрического КПД нетто АСММ\n"
          "от недогревов высокотемпературного и низкотемпературного рекуператоров",
    colorbar_label=r"$\eta_{нетто}^{АСММ}$, %",
    filename="contour_eta_net_dT_HTHE_LTHE.png",
    figsize=FIGSIZES["dt_contour"],
)

# 2б. Одномерный график — по диагонали
plot_1d_sensitivity(
    df=df_dt,
    x_column="dT_HTHE_K",
    z_column="eta_el_net_%",
    diagonal_column="dT_LTHE_K",
    xlabel=r"$\Delta T_{min}$, K",
    ylabel=r"$\eta_{нетто}^{АСММ}$, %",
    title="Влияние минимального температурного напора рекуператора\n"
          "на электрический КПД нетто АСММ",
    filename="line_eta_net_dT_recuperator.png",
    figsize=FIGSIZES["dt_line"],
)

# 2в. Семейство кривых
fig, ax = plt.subplots(figsize=FIGSIZES["dt_family"])
for dtl in sorted(df_dt["dT_LTHE_K"].dropna().unique()):
    d_fixed = filter_physical(df_dt)
    d_fixed = d_fixed[np.isclose(d_fixed["dT_LTHE_K"], dtl)].sort_values(by="dT_HTHE_K")
    if len(d_fixed) < 3:
        continue
    ax.plot(d_fixed["dT_HTHE_K"], d_fixed["eta_el_net_%"],
            marker="o", linewidth=1.5, label=rf"$\Delta T_{{LTHE}}$ = {dtl:.0f} K")

ax.set_xlabel(r"$\Delta T_{HTHE}$, K", fontsize=FONT_SIZE, labelpad=8)
ax.set_ylabel(r"$\eta_{нетто}^{АСММ}$, %", fontsize=FONT_SIZE, labelpad=8)
ax.set_title("Семейство кривых электрического КПД нетто АСММ\n"
             "при различных недогревах низкотемпературного рекуператора",
             fontsize=FONT_SIZE, pad=18)
ax.tick_params(axis="both", labelsize=FONT_SIZE)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=FONT_SIZE - 4, ncol=2)

set_axis_y_ticks(ax, step=ETA_TICK_STEP)

plt.tight_layout()
out_path = OUT_DIR / "family_eta_net_dT_HTHE_LTHE.png"
plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
plt.show()
CREATED_FILES.append(out_path)

# ============================================================
# 3. Потери давления — в абсолютных МПа
# ============================================================
df_dp = pd.read_excel(PATH, sheet_name="dP_hot_cold")
df_dp = filter_physical(df_dp)

print()
print("=" * 70)
print("ПРОВЕРКА ЕДИНИЦ ПОТЕРЬ ДАВЛЕНИЯ")
print("=" * 70)
print(df_dp[["dP_hot_%", "dP_cold_%", "P3_MPa", "P12_MPa"]].head())
print()
print(f"Максимум dP_hot_%  = {df_dp['dP_hot_%'].max()}")
print(f"Максимум dP_cold_% = {df_dp['dP_cold_%'].max()}")

if df_dp["dP_hot_%"].max() > 1.0 or df_dp["dP_cold_%"].max() > 1.0:
    print("dP_% хранятся как ПРОЦЕНТЫ — делим на 100")
    df_dp["dP_hot_MPa"]  = df_dp["dP_hot_%"]  / 100.0 * df_dp["P3_MPa"]
    df_dp["dP_cold_MPa"] = df_dp["dP_cold_%"] / 100.0 * df_dp["P12_MPa"]
else:
    print("dP_% хранятся как ДОЛИ — не делим")
    df_dp["dP_hot_MPa"]  = df_dp["dP_hot_%"]  * df_dp["P3_MPa"]
    df_dp["dP_cold_MPa"] = df_dp["dP_cold_%"] * df_dp["P12_MPa"]

print()
print("После пересчёта:")
print(df_dp[["dP_hot_MPa", "dP_cold_MPa"]].describe())

plot_contour_map(
    df=df_dp,
    x_column="dP_cold_MPa",
    y_column="dP_hot_MPa",
    z_column="eta_el_net_%",
    xlabel=r"$\Delta P_{хол}$, МПа",
    ylabel=r"$\Delta P_{гор}$, МПа",
    title="Зависимость электрического КПД нетто АСММ\n"
          "от потерь давления в рекуператорах",
    colorbar_label=r"$\eta_{нетто}^{АСММ}$, %",
    filename="contour_eta_net_dP_hot_cold.png",
    figsize=FIGSIZES["dp_contour"],
)

if len(df_dp) > 0:
    dp_hot_min = df_dp["dP_hot_MPa"].min()
    plot_1d_sensitivity(
        df=df_dp,
        x_column="dP_cold_MPa",
        z_column="eta_el_net_%",
        fixed_column="dP_hot_MPa",
        fixed_value=dp_hot_min,
        xlabel=r"$\Delta P_{хол}$, МПа",
        ylabel=r"$\eta_{нетто}^{АСММ}$, %",
        title="Влияние потерь давления по холодной стороне\n"
              "на электрический КПД нетто АСММ",
        filename="line_eta_net_dP_cold.png",
        figsize=FIGSIZES["dp_line"],
    )

# ============================================================
# ИТОГ
# ============================================================
print()
print("=" * 70)
print("ПОСТРОЕНИЕ ГРАФИКОВ ЗАВЕРШЕНО")
print("=" * 70)
print()
print(f"Папка сохранения: {OUT_DIR}")
print()
print("Созданы файлы:")
for i, p in enumerate(CREATED_FILES, start=1):
    print(f"{i}. {p.name}")