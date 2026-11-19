"""Расчёт скорости звука в смеси He-Xe по табличной модели props.npy.

Скорость звука определяется из изоэнтропической производной:
    a = sqrt((dp / d rho)_(s, x_Xe))

Для двух близких значений давления температура подбирается так, чтобы
энтропия смеси сохранялась постоянной. Затем производная плотности по
давлению определяется центральной разностью. Такой расчёт согласован с
вириальной моделью свойств, по которой сформирован props.npy.

Пример:
python "Скорость звука He-Xe.py" --props "C:\\path\\props.npy" --x-xe 0.283 --p-mpa 5.2 --t-k 1303.15
"""

from __future__ import annotations

import argparse
import csv
import runpy
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq


PROPERTY_NAMES = ("D", "H", "S", "Cp", "Vis", "Tcx")

# Значения по умолчанию позволяют запускать файл кнопкой Run в PyCharm.
# При необходимости их можно изменить здесь либо передать через аргументы.
DEFAULT_PROPS_PATH = Path(
    r"C:\Users\KaluginaXF\Desktop\Рабочие материалы\Проекты\ЯУЭ 2026 этап 1\РНФ НИКИЭТ июль\Оптимизация код\engineering-codes\04.Models He-Xe\Все модели\01_Основная_модель\props.npy"
)
DEFAULT_X_XE = 0.283
DEFAULT_P_MPA = 5.2
DEFAULT_T_K = 1303.15
DEFAULT_SERIES_PRESSURES_MPA = (2.0, 4.0, 5.2)
DEFAULT_SERIES_T_MIN_K = 275.0
DEFAULT_SERIES_T_MAX_K = 1375.0
DEFAULT_SERIES_T_STEP_K = 25.0
DEFAULT_SERIES_OUT_DIR = Path(__file__).resolve().parent / "Результаты основной модели" / "Скорость звука"


class HeXeProperties:
    """Доступ к интерполяторам свойств, сохранённым в props.npy."""

    def __init__(self, props_path: Path, x_xe: float) -> None:
        raw = np.load(props_path, allow_pickle=True)
        if len(raw) != len(PROPERTY_NAMES):
            raise ValueError(
                f"Ожидалось {len(PROPERTY_NAMES)} интерполяторов, получено {len(raw)}."
            )
        self.x_xe = float(x_xe)
        self.functions: dict[str, Callable] = {}
        for name, legacy in zip(PROPERTY_NAMES, raw):
            # props.npy создан старой версией SciPy. Вместо исправления
            # внутренней структуры pickled-объекта создаём совместимый
            # интерполятор из сохранённых сетки и значений.
            grid = legacy.__dict__.get("grid", getattr(legacy, "_grid", None))
            values = legacy.__dict__.get("values", getattr(legacy, "_values", None))
            if grid is None or values is None:
                raise ValueError(f"В props.npy отсутствуют сетка или значения свойства {name}.")
            self.functions[name] = RegularGridInterpolator(
                grid, values, method="linear", bounds_error=True
            )

    def bounds(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        grid = self.functions[name].grid
        lower = np.array([float(axis[0]) for axis in grid])
        upper = np.array([float(axis[-1]) for axis in grid])
        return lower, upper

    def value(self, name: str, pressure_pa: float, temperature_k: float) -> float:
        lower, upper = self.bounds(name)
        point = np.array([self.x_xe, pressure_pa, temperature_k])
        if np.any(point < lower) or np.any(point > upper):
            raise ValueError(
                f"Точка x_Xe={self.x_xe:.6g}, p={pressure_pa:.6g} Па, "
                f"T={temperature_k:.6g} К вне таблицы {name}. "
                f"Допустимо: x_Xe={lower[0]:.6g}..{upper[0]:.6g}, "
                f"p={lower[1]:.6g}..{upper[1]:.6g} Па, "
                f"T={lower[2]:.6g}..{upper[2]:.6g} К."
            )
        result = float(np.asarray(self.functions[name]((self.x_xe, pressure_pa, temperature_k))).squeeze())
        if not np.isfinite(result):
            raise ValueError(f"Интерполятор {name} вернул некорректное значение.")
        return result

    def temperature_at_entropy(
        self,
        entropy_j_kgk: float,
        pressure_pa: float,
        guess_k: float,
    ) -> float:
        """Находит температуру при заданных давлении и энтропии."""
        lower, upper = self.bounds("S")
        if not lower[1] <= pressure_pa <= upper[1]:
            raise ValueError("Давление вне области интерполятора энтропии.")

        t_min, t_max = float(lower[2]), float(upper[2])

        def residual(temperature_k: float) -> float:
            return self.value("S", pressure_pa, temperature_k) - entropy_j_kgk

        f_min, f_max = residual(t_min), residual(t_max)
        if f_min == 0.0:
            return t_min
        if f_max == 0.0:
            return t_max
        if f_min * f_max < 0.0:
            return float(brentq(residual, t_min, t_max, xtol=1.0e-8))

        # Резервный поиск интервала смены знака для немонотонной таблицы.
        grid = np.asarray(self.functions["S"].grid[2], dtype=float)
        values = np.array([residual(temperature) for temperature in grid])
        candidates = np.flatnonzero(values[:-1] * values[1:] < 0.0)
        if not candidates.size:
            raise ValueError("Невозможно восстановить температуру по заданной энтропии.")
        index = int(candidates[np.argmin(np.abs(grid[candidates] - guess_k))])
        return float(brentq(residual, grid[index], grid[index + 1], xtol=1.0e-8))


def sound_speed(
    properties: HeXeProperties,
    pressure_pa: float,
    temperature_k: float,
    relative_pressure_step: float = 1.0e-4,
) -> dict[str, float]:
    """Возвращает скорость звука и промежуточные величины.

    Производная (d rho / d p)_s вычисляется центральной разностью. Шаг
    автоматически уменьшается у границы таблицы давления.
    """
    if relative_pressure_step <= 0.0:
        raise ValueError("Относительный шаг давления должен быть положительным.")

    entropy = properties.value("S", pressure_pa, temperature_k)
    density = properties.value("D", pressure_pa, temperature_k)
    lower, upper = properties.bounds("D")
    p_min, p_max = float(lower[1]), float(upper[1])

    requested_step = max(100.0, relative_pressure_step * pressure_pa)
    room_minus = pressure_pa - p_min
    room_plus = p_max - pressure_pa

    # Внутри таблицы используется центральная разность. На её границах,
    # которые входят в серию давлений, используется односторонняя разность.
    if room_minus > 0.0 and room_plus > 0.0:
        pressure_step = min(requested_step, 0.49 * room_minus, 0.49 * room_plus)
        p_minus, p_plus = pressure_pa - pressure_step, pressure_pa + pressure_step
        t_minus = properties.temperature_at_entropy(entropy, p_minus, temperature_k)
        t_plus = properties.temperature_at_entropy(entropy, p_plus, temperature_k)
        rho_minus = properties.value("D", p_minus, t_minus)
        rho_plus = properties.value("D", p_plus, t_plus)
        drho_dp_s = (rho_plus - rho_minus) / (p_plus - p_minus)
    elif room_plus > 0.0:
        pressure_step = min(requested_step, 0.49 * room_plus)
        p_plus = pressure_pa + pressure_step
        t_plus = properties.temperature_at_entropy(entropy, p_plus, temperature_k)
        rho_plus = properties.value("D", p_plus, t_plus)
        t_minus, rho_minus = np.nan, np.nan
        drho_dp_s = (rho_plus - density) / pressure_step
    elif room_minus > 0.0:
        pressure_step = min(requested_step, 0.49 * room_minus)
        p_minus = pressure_pa - pressure_step
        t_minus = properties.temperature_at_entropy(entropy, p_minus, temperature_k)
        rho_minus = properties.value("D", p_minus, t_minus)
        t_plus, rho_plus = np.nan, np.nan
        drho_dp_s = (density - rho_minus) / pressure_step
    else:
        raise ValueError("В таблице отсутствует интервал давления для вычисления производной.")
    if not np.isfinite(drho_dp_s) or drho_dp_s <= 0.0:
        raise ValueError("Получена некорректная изоэнтропическая сжимаемость.")

    speed_m_s = float(np.sqrt(1.0 / drho_dp_s))
    return {
        "x_Xe": properties.x_xe,
        "p_Pa": pressure_pa,
        "T_K": temperature_k,
        "rho_kg_m3": density,
        "s_J_kgK": entropy,
        "dp_Pa": pressure_step,
        "T_minus_K": t_minus,
        "T_plus_K": t_plus,
        "rho_minus_kg_m3": rho_minus,
        "rho_plus_kg_m3": rho_plus,
        "drho_dp_s_kg_m3_Pa": drho_dp_s,
        "sound_speed_m_s": speed_m_s,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("charts", "series", "point"), default="charts",
        help="charts: полный набор графиков; series: один температурный график; point: одно состояние",
    )
    parser.add_argument(
        "--props", type=Path, default=DEFAULT_PROPS_PATH,
        help="Путь к props.npy",
    )
    parser.add_argument("--x-xe", type=float, default=DEFAULT_X_XE, help="Мольная доля Xe")
    parser.add_argument("--p-mpa", type=float, default=DEFAULT_P_MPA, help="Давление, МПа")
    parser.add_argument("--t-k", type=float, default=DEFAULT_T_K, help="Температура, К")
    parser.add_argument(
        "--relative-pressure-step",
        type=float,
        default=1.0e-4,
        help="Относительный шаг центральной разности по давлению",
    )
    parser.add_argument("--out", type=Path, help="Необязательный путь для CSV-файла")
    parser.add_argument(
        "--pressures-mpa", type=str,
        default=",".join(str(value) for value in DEFAULT_SERIES_PRESSURES_MPA),
        help="Давления для серии, МПа, через запятую",
    )
    parser.add_argument("--t-min-k", type=float, default=DEFAULT_SERIES_T_MIN_K)
    parser.add_argument("--t-max-k", type=float, default=DEFAULT_SERIES_T_MAX_K)
    parser.add_argument("--t-step-k", type=float, default=DEFAULT_SERIES_T_STEP_K)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SERIES_OUT_DIR)
    return parser.parse_args()


def parse_pressure_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("Нужен непустой список положительных давлений.")
    return values


def calculate_series(
    model: HeXeProperties,
    pressures_mpa: list[float],
    t_min_k: float,
    t_max_k: float,
    t_step_k: float,
    relative_pressure_step: float,
) -> list[dict[str, float]]:
    if t_step_k <= 0.0 or t_max_k < t_min_k:
        raise ValueError("Некорректный диапазон температур для серии.")

    temperatures = np.arange(t_min_k, t_max_k + 0.5 * t_step_k, t_step_k)
    rows: list[dict[str, float]] = []
    for pressure_mpa in pressures_mpa:
        for temperature_k in temperatures:
            try:
                row = sound_speed(
                    model, pressure_mpa * 1.0e6, float(temperature_k), relative_pressure_step
                )
                row["p_MPa"] = pressure_mpa
                row["calculation_ok"] = 1.0
                rows.append(row)
            except ValueError as error:
                rows.append({
                    "x_Xe": model.x_xe,
                    "p_Pa": pressure_mpa * 1.0e6,
                    "p_MPa": pressure_mpa,
                    "T_K": float(temperature_k),
                    "sound_speed_m_s": np.nan,
                    "calculation_ok": 0.0,
                    "error": str(error),
                })
    return rows


def export_series(rows: list[dict[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sound_speed_temperature_series.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    pressures = sorted({float(row["p_MPa"]) for row in rows})
    series: dict[float, list[dict[str, float]]] = {}
    for pressure_mpa in pressures:
        selected = [row for row in rows if float(row["p_MPa"]) == pressure_mpa and np.isfinite(row.get("sound_speed_m_s", np.nan))]
        series[pressure_mpa] = selected

    # SVG строится без дополнительных библиотек и может быть вставлен в PowerPoint.
    write_svg_chart(series, out_dir / "sound_speed_temperature_series.svg")

    # Если matplotlib установлен в пользовательском Python, дополнительно
    # формируется PNG с разрешением 300 dpi.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен: создан SVG-график, PNG не сформирован.")
    else:
        plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
        fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
        for pressure_mpa, selected in series.items():
            if selected:
                ax.plot(
                    [row["T_K"] for row in selected],
                    [row["sound_speed_m_s"] for row in selected],
                    marker="o", markersize=3.6, linewidth=1.5,
                    label=f"p = {pressure_mpa:g} МПа",
                )
        ax.set(xlabel="Температура, К", ylabel="Скорость звука, м/с",
               title="Скорость звука в смеси He-Xe, x_Xe = 0.283")
        ax.grid(alpha=0.22, linewidth=0.55)
        ax.legend(frameon=True, facecolor="white", edgecolor="0.55")
        fig.savefig(out_dir / "sound_speed_temperature_series.png", dpi=300, facecolor="white")
        plt.close(fig)
    print(f"Серия сохранена: {csv_path}")
    print(f"Графики сохранены: {out_dir}")


def write_svg_chart(series: dict[float, list[dict[str, float]]], path: Path) -> None:
    """Создаёт простой векторный SVG-график без графических библиотек."""
    points = [row for rows in series.values() for row in rows]
    if not points:
        raise ValueError("Нет точек для построения графика скорости звука.")
    x_values = np.array([row["T_K"] for row in points], dtype=float)
    y_values = np.array([row["sound_speed_m_s"] for row in points], dtype=float)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    y_margin = max(5.0, 0.06 * (y_max - y_min))
    y_min, y_max = y_min - y_margin, y_max + y_margin

    width, height = 960, 640
    left, right, top, bottom = 110, 40, 70, 95
    plot_width, plot_height = width - left - right, height - top - bottom
    colors = ("#1F77B4", "#D62728", "#2CA02C", "#9467BD", "#FF7F0E")
    x_to_svg = lambda value: left + (value - x_min) / (x_max - x_min) * plot_width
    y_to_svg = lambda value: top + (y_max - value) / (y_max - y_min) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="480" y="35" text-anchor="middle" font-family="Times New Roman" font-size="22">Скорость звука в смеси He-Xe, x_Xe = 0.283</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
    ]
    for fraction in np.linspace(0.0, 1.0, 6):
        x = left + fraction * plot_width
        y = top + (1.0 - fraction) * plot_height
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_min + fraction * (y_max - y_min)
        lines.extend([
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#d9d9d9"/>',
            f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="Times New Roman" font-size="15">{x_value:.0f}</text>',
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#d9d9d9"/>',
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Times New Roman" font-size="15">{y_value:.0f}</text>',
        ])
    for index, (pressure_mpa, rows) in enumerate(series.items()):
        if not rows:
            continue
        color = colors[index % len(colors)]
        polyline = " ".join(f"{x_to_svg(row['T_K']):.1f},{y_to_svg(row['sound_speed_m_s']):.1f}" for row in rows)
        lines.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_y = top + 25 + 25 * index
        lines.extend([
            f'<line x1="{left + plot_width - 180}" y1="{legend_y}" x2="{left + plot_width - 150}" y2="{legend_y}" stroke="{color}" stroke-width="2.5"/>',
            f'<text x="{left + plot_width - 142}" y="{legend_y + 5}" font-family="Times New Roman" font-size="15">p = {pressure_mpa:g} МПа</text>',
        ])
    lines.extend([
        f'<text x="{left + plot_width / 2}" y="{height - 28}" text-anchor="middle" font-family="Times New Roman" font-size="18">Температура, К</text>',
        f'<text x="28" y="{top + plot_height / 2}" text-anchor="middle" font-family="Times New Roman" font-size="18" transform="rotate(-90 28 {top + plot_height / 2})">Скорость звука, м/с</text>',
        '</svg>',
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_arguments()

    # Обычный запуск из PyCharm формирует полный набор графиков.
    # Отдельный файл используется, чтобы не дублировать расчётные функции.
    if args.mode == "charts":
        chart_script = Path(__file__).with_name("Скорость звука He-Xe графики.py")
        if not chart_script.exists():
            raise FileNotFoundError(f"Не найден файл расширенных графиков: {chart_script}")
        original_argv = sys.argv
        try:
            sys.argv = [str(chart_script)]
            runpy.run_path(str(chart_script), run_name="__main__")
        finally:
            sys.argv = original_argv
        return

    model = HeXeProperties(args.props, args.x_xe)

    if args.mode == "series":
        rows = calculate_series(
            model=model,
            pressures_mpa=parse_pressure_list(args.pressures_mpa),
            t_min_k=args.t_min_k,
            t_max_k=args.t_max_k,
            t_step_k=args.t_step_k,
            relative_pressure_step=args.relative_pressure_step,
        )
        export_series(rows, args.out_dir)
        successful = sum(bool(row.get("calculation_ok")) for row in rows)
        print(f"Успешно рассчитано точек: {successful} из {len(rows)}")
        return

    result = sound_speed(
        model,
        pressure_pa=args.p_mpa * 1.0e6,
        temperature_k=args.t_k,
        relative_pressure_step=args.relative_pressure_step,
    )

    print(f"x_Xe = {result['x_Xe']:.6f}")
    print(f"p = {result['p_Pa'] / 1.0e6:.6f} МПа")
    print(f"T = {result['T_K']:.3f} К")
    print(f"rho = {result['rho_kg_m3']:.6f} кг/м3")
    print(f"a = {result['sound_speed_m_s']:.6f} м/с")
    print(f"dp = {result['dp_Pa']:.3f} Па")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=result.keys())
            writer.writeheader()
            writer.writerow(result)
        print(f"Результат сохранён: {args.out}")


if __name__ == "__main__":
    main()
