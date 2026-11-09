from __future__ import annotations

"""Анализ чувствительности, использующий актуальную основную модель He-Xe."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
MAIN_DIR = HERE.parent / "01_Основная_модель"
MAIN_SCRIPT = MAIN_DIR / "Test model.py"
PROPS_FILE = MAIN_DIR / "props.npy"
OUT_DIR = HERE / "Результаты_чувствительности_HeXe"
OUT_DIR.mkdir(exist_ok=True)
GENERAL_SWEEP_FILE = HERE.parent / "res_hexe" / "pressure_sweep_full.csv"

# Базовая точка чувствительности: изменяется при необходимости.
BASE = {"T6_C": 750.0, "P1_MPa": 5.3, "P2_MPa": 2.8}
ETA_T = np.array([0.85, 0.88, 0.90, 0.92, 0.9267, 0.94, 0.96, 0.98])
ETA_C = np.array([0.78, 0.80, 0.82, 0.85, 0.8706, 0.89, 0.91, 0.93])
DP = np.array([0.01, 0.02, 0.03, 0.04553, 0.06, 0.08, 0.10])


def load_core():
    if not MAIN_SCRIPT.exists() or not PROPS_FILE.exists():
        raise FileNotFoundError("Не найдены Test model.py или props.npy основной модели.")
    spec = importlib.util.spec_from_file_location("hexe_main", MAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось импортировать {MAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scenarios() -> list[dict]:
    rows: list[dict] = []
    rows += [
        {"group": "КПД турбины и компрессора", "parameter": "eta_turbine",
         "parameter_value": et, "eta_turbine": et, "eta_compressor": ec}
        for et in ETA_T for ec in ETA_C
    ]
    rows += [
        {"group": "Потери давления рекуператора", "parameter": "dp_hot_frac",
         "parameter_value": 100.0 * value, "dp_hot_frac": value, "dp_cold_frac": value}
        for value in DP
    ]
    return rows


def calculate(model, scenario: dict, is_base: bool = False) -> dict:
    result, _ = model.solve(BASE["P1_MPa"], BASE["P2_MPa"], BASE["T6_C"],
                            return_states=True, refine_pinch=True)
    row = BASE | scenario | result | {"is_base": is_base}
    for source, target, divisor in (
        ("N_T_el_W", "N_T_el_MW", 1e6),
        ("N_K_el_consumed_W", "N_K_el_consumed_MW", 1e6),
        ("N_el_gross_W", "N_el_gross_MW", 1e6),
        ("N_el_net_W", "N_el_net_MW", 1e6),
        ("N_aux_total_W", "N_aux_total_kW", 1e3),
        ("Q_pipe_loss_W", "Q_pipe_loss_kW", 1e3),
    ):
        if source in row:
            row[target] = row[source] / divisor
    return row


def make_changes(frame: pd.DataFrame, base: pd.Series) -> pd.DataFrame:
    metrics = {
        "eta_asmm_net_initial_heat": "КПД нетто АСММ",
        "N_el_net_MW": "Электрическая мощность нетто, МВт",
        "N_T_el_MW": "Электрическая мощность турбины, МВт",
        "N_K_el_consumed_MW": "Потребление компрессора, МВт",
        "Q_pipe_loss_kW": "Потери трубопровода, кВт",
        "T_pinch_C": "Минимальный температурный напор, К",
    }
    rows = []
    for _, scenario in frame.iterrows():
        for column, metric in metrics.items():
            value, ref = scenario.get(column, np.nan), base.get(column, np.nan)
            delta = ((value - ref) / abs(ref) * 100.0
                     if pd.notna(value) and pd.notna(ref) and ref != 0 else np.nan)
            rows.append({
                "group": scenario["group"], "parameter": scenario["parameter"],
                "parameter_value": scenario["parameter_value"], "metric": metric,
                "base_value": ref, "scenario_value": value,
                "relative_change_pct": delta, "valid": scenario.get("valid", False),
            })
    return pd.DataFrame(rows)


def pinch_dependence_from_sweep() -> pd.DataFrame:
    """Максимальный КПД для интервалов фактического pinch из общего перебора."""
    if not GENERAL_SWEEP_FILE.exists():
        raise FileNotFoundError(f"Не найдена общая таблица перебора: {GENERAL_SWEEP_FILE}")
    data = pd.read_csv(GENERAL_SWEEP_FILE, low_memory=False).replace([np.inf, -np.inf], np.nan)
    needed = {"T6_C", "T_pinch_C", "eta_asmm_net_initial_heat", "N_el_net_W", "Q_cool_W"}
    missing = needed.difference(data.columns)
    if missing:
        raise KeyError(f"В общей таблице нет столбцов: {sorted(missing)}")
    data = data.loc[
        data["T_pinch_C"].between(1.0, 15.0)
        & data["eta_asmm_net_initial_heat"].notna()
        & data["N_el_net_W"].gt(0)
        & data["Q_cool_W"].ge(0)
    ].copy()
    edges = np.arange(1.0, 15.5, 0.5)
    data["pinch_bin"] = pd.cut(data["T_pinch_C"], bins=edges, include_lowest=True)
    result = data.groupby(["T6_C", "pinch_bin"], observed=True).agg(
        T_pinch_C=("T_pinch_C", "mean"),
        eta_asmm_net_initial_heat=("eta_asmm_net_initial_heat", "max"),
        P1_MPa=("P1_MPa", "first"), P2_MPa=("P2_MPa", "first"),
        n_modes=("T_pinch_C", "size"),
    ).reset_index()
    return result


def export_excel(base: pd.DataFrame, frame: pd.DataFrame, changes: pd.DataFrame,
                 pinch: pd.DataFrame) -> None:
    with pd.ExcelWriter(OUT_DIR / "sensitivity_results.xlsx", engine="openpyxl") as writer:
        base.to_excel(writer, sheet_name="Базовый режим", index=False)
        frame.to_excel(writer, sheet_name="Сценарии", index=False)
        changes.to_excel(writer, sheet_name="Отклонения от базы", index=False)
        pinch.to_excel(writer, sheet_name="КПД от фактического pinch", index=False)


def main() -> None:
    core = load_core()
    base_cfg = core.CycleConfig()
    base_cfg.validate()
    props = core.PropertyModel(PROPS_FILE, base_cfg.x_xe)
    base_scenario = {"group": "Базовый режим", "parameter": "—", "parameter_value": np.nan}
    base = calculate(core.CycleModel(base_cfg, props), base_scenario, is_base=True)
    if not base.get("valid", False):
        raise RuntimeError(f"Базовый режим недопустим: {base.get('reason', '')}")

    rows = []
    allowed = {"eta_turbine", "eta_compressor", "dp_hot_frac", "dp_cold_frac"}
    for scenario in tqdm(scenarios(), desc="Расчёт чувствительности", unit="сценарий"):
        overrides = {key: value for key, value in scenario.items() if key in allowed}
        try:
            cfg = replace(base_cfg, **overrides)
            cfg.validate()
            rows.append(calculate(core.CycleModel(cfg, props), scenario))
        except Exception as exc:
            rows.append(BASE | scenario | {"valid": False, "is_base": False,
                                            "reason": f"{type(exc).__name__}: {exc}"})

    frame = pd.DataFrame(rows)
    base_frame = pd.DataFrame([base])
    changes = make_changes(frame, base_frame.iloc[0])
    pinch = pinch_dependence_from_sweep()
    base_frame.to_csv(OUT_DIR / "00_базовый_режим.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(OUT_DIR / "01_сценарии_чувствительности.csv", index=False, encoding="utf-8-sig")
    changes.to_csv(OUT_DIR / "02_отклонения_от_базы.csv", index=False, encoding="utf-8-sig")
    pinch.to_csv(OUT_DIR / "03_кпд_от_фактического_pinch.csv", index=False, encoding="utf-8-sig")
    export_excel(base_frame, frame, changes, pinch)
    print(f"Результаты сохранены: {OUT_DIR}")
    print(f"КПД нетто АСММ в базе: {100 * base['eta_asmm_net_initial_heat']:.3f} %")
    print(f"Допустимые сценарии: {int(frame['valid'].sum())}/{len(frame)}")


if __name__ == "__main__":
    main()
