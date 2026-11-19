from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.ticker import AutoMinorLocator
from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy.optimize import brentq, minimize_scalar
from tqdm.auto import tqdm


PROPERTY_NAMES = ("D", "H", "S", "Cp", "Vis", "Tcx")
FONT = {"family": "DejaVu Sans",
    "title": 10.0,
    "label": 9.0,
    "tick": 8.0,
    "legend": 8.0,
    "annotation": 8.0,
    "weight": "normal",}
FIGURE = { "single_column": (3.54, 2.75),
    "double_column": (7.48, 4.60),
    "dpi": 300,
    "formats": ("pdf", "svg"),
    "cmap": "viridis",}

def apply_plot_style() -> None:
    mpl.rcParams.update({
        "font.family": FONT["family"],
        "font.weight": FONT["weight"],
        "font.size": FONT["tick"],
        "axes.titlesize": FONT["title"],
        "axes.labelsize": FONT["label"],
        "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"],
        "legend.fontsize": FONT["legend"],
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.25,
        "lines.markersize": 4.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "axes.grid": False,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    "svg.fonttype": "none",
    })

@dataclass(frozen=True)
class CycleConfig:
    q_reactor_nom_w: float = 2.75e6
    eta_reactor: float = 0.99
    x_xe: float = 0.283
    t0_c: float = 1060.0
    t1_c: float = 1030.0
    t4_c: float = 35.0

    eta_turbine: float = 0.9267
    eta_compressor: float = 0.8706
    eta_turbine_mech: float = 0.99
    eta_compressor_mech: float = 0.99
    eta_generator: float = 0.98
    eta_compressor_drive: float = 0.98
    n_aux_other_w: float = 0.0

    dp_hot_frac: float = 0.04553
    dp_cold_frac: float = 0.04553
    dt_pinch_min_c: float = 5.0
    pinch_points: int = 32
    p_water_pa: float = 6.0e5
    dp_water_frac: float = 0.05
    dt_water_k: float = 10.0
    cp_water_j_kgk: float = 4180.0
    rho_water_kg_m3: float = 997.0
    eta_water_pump: float = 0.80
    eta_water_motor: float = 0.95

    t6_grid_c: tuple[float, ...] = (650.0, 700.0, 750.0, 800.0, 850.0)
    p1_grid_mpa: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.round(np.arange(3.0, 8.001, 0.05), 4)))
    p2_grid_mpa: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.round(np.arange(0.5, 6.001, 0.05), 4)))

    def validate(self) -> None:
        etas = {
            "eta_reactor": self.eta_reactor,
            "eta_turbine": self.eta_turbine, "eta_compressor": self.eta_compressor,
            "eta_turbine_mech": self.eta_turbine_mech,
            "eta_compressor_mech": self.eta_compressor_mech,
            "eta_generator": self.eta_generator,
            "eta_compressor_drive": self.eta_compressor_drive,
            "eta_water_pump": self.eta_water_pump,
            "eta_water_motor": self.eta_water_motor,}
        bad = {k: v for k, v in etas.items() if not 0.0 < v <= 1.0}
        if bad:
            raise ValueError(f"Efficiencies must be in (0, 1]: {bad}")
        if not (0.0 <= self.dp_hot_frac < 1.0 and 0.0 <= self.dp_cold_frac < 1.0):
            raise ValueError("Relative pressure losses must be in [0, 1).")
        if self.pinch_points < 3:
            raise ValueError("pinch_points must be at least 3.")
        if self.t0_c <= self.t1_c:
            raise ValueError("t0_c must be greater than the fixed turbine inlet temperature t1_c.")

class PropertyModel:
    def __init__(self, path: Path, x_xe: float):
        self.path = Path(path)
        raw = np.load(path, allow_pickle=True)
        if len(raw) != len(PROPERTY_NAMES):
            raise ValueError(f"Expected {len(PROPERTY_NAMES)} interpolators, got {len(raw)}.")
        self.x = x_xe
        self.f: dict[str, Callable] = dict(zip(PROPERTY_NAMES, raw))
        for interpolator in self.f.values():
            if not hasattr(interpolator, "grid") and hasattr(interpolator, "_grid"):
                interpolator.grid = interpolator._grid
            if not hasattr(interpolator, "values") and hasattr(interpolator, "_values"):
                interpolator.values = interpolator._values
        for name, interpolator in self.f.items():
            if not hasattr(interpolator, "grid"):
                raise TypeError(f"Interpolator {name!r} does not expose .grid or ._grid.")

    def bounds(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        grid = self.f[name].grid
        lo = np.array([np.asarray(axis, float)[0] for axis in grid])
        hi = np.array([np.asarray(axis, float)[-1] for axis in grid])
        return lo, hi

    def check_point(self, name: str, p: float, t: float) -> None:
        lo, hi = self.bounds(name)
        point = np.array([self.x, p, t])
        if np.any(point < lo) or np.any(point > hi):
            raise ValueError(
                f"{name}: point (x={self.x:g}, P={p:g} Pa, T={t:g} K) is outside "
                f"table bounds x={lo[0]:g}..{hi[0]:g}, P={lo[1]:g}..{hi[1]:g} Pa, "
                f"T={lo[2]:g}..{hi[2]:g} K.")

    def value(self, name: str, p: float, t: float) -> float:
        self.check_point(name, p, t)
        value = float(np.asarray(self.f[name]((self.x, p, t))).squeeze())
        if not np.isfinite(value):
            raise ValueError(f"{name} interpolator returned a non-finite value.")
        return value

    def temperature(self, name: str, target: float, p: float, guess: float) -> float:
        """Invert H(P,T) or S(P,T), using that property's own temperature grid."""
        interpolator = self.f[name]
        grid = np.asarray(interpolator.grid[2], float)
        p_lo, p_hi = self.bounds(name)[0][1], self.bounds(name)[1][1]
        if not p_lo <= p <= p_hi:
            raise ValueError(f"{name}: pressure {p:g} Pa is outside {p_lo:g}..{p_hi:g} Pa.")

        def residual(t: float) -> float:
            return float(np.asarray(interpolator((self.x, p, t))).squeeze()) - target

        lo, hi = float(grid[0]), float(grid[-1])
        f_lo, f_hi = residual(lo), residual(hi)
        if f_lo == 0.0:
            return lo
        if f_hi == 0.0:
            return hi
        if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi < 0.0:
            return float(brentq(residual, lo, hi, xtol=1e-7))

        values = np.array([residual(t) for t in grid])
        exact = np.flatnonzero(values == 0.0)
        if exact.size:
            return float(grid[exact[np.argmin(abs(grid[exact] - guess))]])
        brackets = np.flatnonzero(values[:-1] * values[1:] < 0.0)
        if not brackets.size:
            raise ValueError(f"Cannot invert {name} at P={p:g} Pa: target={target:g} is out of range.")
        i = int(brackets[np.argmin(abs(grid[brackets] - guess))])
        return float(brentq(residual, float(grid[i]), float(grid[i + 1]), xtol=1e-8))

@dataclass
class State:
    node: int
    t_k: float
    p_pa: float
    g_kg_s: float
    h_j_kg: float = np.nan
    s_j_kgk: float = np.nan
    rho_kg_m3: float = np.nan
    cp_j_kgk: float = np.nan
    mu_pa_s: float = np.nan
    k_w_mk: float = np.nan

    @property
    def t_c(self) -> float:
        return self.t_k - 273.15

class CycleModel:
    def __init__(self, cfg: CycleConfig, props: PropertyModel):
        self.c, self.p = cfg, props
    def state_pt(self, node: int, p: float, t: float, g: float) -> State:
        return State(
            node, t, p, g,
            h_j_kg=self.p.value("H", p, t),
            s_j_kgk=self.p.value("S", p, t),
            rho_kg_m3=self.p.value("D", p, t),
            cp_j_kgk=self.p.value("Cp", p, t),
            mu_pa_s=self.p.value("Vis", p, t),
            k_w_mk=self.p.value("Tcx", p, t),)
    def state_ph(self, node: int, p: float, h: float, g: float, guess: float) -> State:
        return self.state_pt(node, p, self.p.temperature("H", h, p, guess), g)

    def turbomachine(self, inlet: State, node: int, p_out: float,
                     eta: float, turbine: bool) -> tuple[State, float]:
        exponent_guess = 0.25
        guess = inlet.t_k * (p_out / inlet.p_pa) ** exponent_guess
        t_is = self.p.temperature("S", inlet.s_j_kgk, p_out, guess)
        h_is = self.p.value("H", p_out, t_is)
        if turbine:
            h_out = inlet.h_j_kg - eta * (inlet.h_j_kg - h_is)
        else:
            h_out = inlet.h_j_kg + (h_is - inlet.h_j_kg) / eta
        outlet = self.state_ph(node, p_out, h_out, inlet.g_kg_s, guess)
        signed_power = inlet.g_kg_s * (inlet.h_j_kg - outlet.h_j_kg)
        return outlet, signed_power

    def recuperator_profile(self, states: dict[int, State], n: int | None = None) -> pd.DataFrame:
        n = n or self.c.pinch_points
        fractions = np.linspace(0.0, 1.0, n)
        rows = []
        for f in fractions:
            h_h = states[2].h_j_kg + f * (states[3].h_j_kg - states[2].h_j_kg)
            p_h = states[2].p_pa + f * (states[3].p_pa - states[2].p_pa)
            gh = states[2].t_k + f * (states[3].t_k - states[2].t_k)
            fc = 1.0 - f
            h_c = states[5].h_j_kg + fc * (states[6].h_j_kg - states[5].h_j_kg)
            p_c = states[5].p_pa + fc * (states[6].p_pa - states[5].p_pa)
            gc = states[5].t_k + fc * (states[6].t_k - states[5].t_k)
            t_h = self.p.temperature("H", h_h, p_h, gh)
            t_c = self.p.temperature("H", h_c, p_c, gc)
            rows.append((f, t_h - 273.15, t_c - 273.15, t_h - t_c, p_h, p_c))
        return pd.DataFrame(rows, columns=(
            "x_L", "T_hot_C", "T_cold_C", "delta_T_K", "P_hot_Pa", "P_cold_Pa"))

    def fast_pinch(self, states: dict[int, State]) -> tuple[float, float]:
        minimum = (np.inf, 0.0)
        for f in np.linspace(0.0, 1.0, self.c.pinch_points):
            h_h = states[2].h_j_kg + f * (states[3].h_j_kg - states[2].h_j_kg)
            p_h = states[2].p_pa + f * (states[3].p_pa - states[2].p_pa)
            fc = 1.0 - f
            h_c = states[5].h_j_kg + fc * (states[6].h_j_kg - states[5].h_j_kg)
            p_c = states[5].p_pa + fc * (states[6].p_pa - states[5].p_pa)
            t_h = self.p.temperature("H", h_h, p_h, states[2].t_k + f * (states[3].t_k - states[2].t_k))
            t_c = self.p.temperature("H", h_c, p_c, states[5].t_k + fc * (states[6].t_k - states[5].t_k))
            minimum = min(minimum, (t_h - t_c, float(f)))
        return minimum

    def refined_pinch(self, states: dict[int, State]) -> tuple[float, float]:
        sampled = self.recuperator_profile(states)
        j = int(sampled["delta_T_K"].idxmin())
        x = sampled["x_L"].to_numpy()
        lo, hi = x[max(0, j - 1)], x[min(len(x) - 1, j + 1)]

        def delta_t(f: float) -> float:
            h_h = states[2].h_j_kg + f * (states[3].h_j_kg - states[2].h_j_kg)
            p_h = states[2].p_pa + f * (states[3].p_pa - states[2].p_pa)
            fc = 1.0 - f
            h_c = states[5].h_j_kg + fc * (states[6].h_j_kg - states[5].h_j_kg)
            p_c = states[5].p_pa + fc * (states[6].p_pa - states[5].p_pa)
            th = self.p.temperature("H", h_h, p_h, np.interp(f, [0, 1], [states[2].t_k, states[3].t_k]))
            tc = self.p.temperature("H", h_c, p_c, np.interp(fc, [0, 1], [states[5].t_k, states[6].t_k]))
            return th - tc

        solution = minimize_scalar(delta_t, bounds=(lo, hi), method="bounded",
                                   options={"xatol": 1e-7})
        candidates = [(delta_t(0.0), 0.0), (delta_t(1.0), 1.0),
                      (float(solution.fun), float(solution.x))]
        return min(candidates)

    def water_pump_power(self, q_cool: float) -> float:
        if not np.isfinite(q_cool) or q_cool <= 0.0:
            return np.nan
        m_w = q_cool / (self.c.cp_water_j_kgk * self.c.dt_water_k)
        volume_flow = m_w / self.c.rho_water_kg_m3
        dp = self.c.dp_water_frac * self.c.p_water_pa
        return dp * volume_flow / (self.c.eta_water_pump * self.c.eta_water_motor)

    def solve(self, p1_mpa: float, p2_mpa: float, t6_c: float,
              return_states: bool = False, refine_pinch: bool = False
              ) -> dict | tuple[dict, dict[int, State]]:
        c = self.c
        p1, p2 = p1_mpa * 1e6, p2_mpa * 1e6
        base = {"valid": False, "reason": "", "T6_C": t6_c,
                "P1_MPa": p1_mpa, "P2_MPa": p2_mpa}
        if p1 <= p2:
            base["reason"] = "P1 <= P2"
            return (base, {}) if return_states else base

        pressures = {0: p1, 1: p1, 2: p2,
                     3: p2 * (1.0 - c.dp_hot_frac),
                     4: p2 * (1.0 - c.dp_hot_frac),
                     5: p1 * (1.0 + c.dp_cold_frac), 6: p1}
        if min(pressures.values()) <= 0.0:
            base["reason"] = "non-positive pressure"
            return (base, {}) if return_states else base

        t0, t4, t6 = c.t0_c + 273.15, c.t4_c + 273.15, t6_c + 273.15
        h0 = self.p.value("H", pressures[0], t0)
        h6 = self.p.value("H", pressures[6], t6)
        dh_reactor = h0 - h6
        if dh_reactor <= 0.0:
            base["reason"] = "h0 <= h6"
            return (base, {}) if return_states else base

        q_reactor_fluid = c.q_reactor_nom_w * c.eta_reactor
        g = q_reactor_fluid / dh_reactor

        states: dict[int, State] = {
            0: self.state_pt(0, pressures[0], t0, g),
            1: self.state_pt(1, pressures[1], c.t1_c + 273.15, g),
            4: self.state_pt(4, pressures[4], t4, g),
            6: self.state_pt(6, pressures[6], t6, g),
        }
        states[2], n_turb = self.turbomachine(
            states[1], 2, pressures[2], c.eta_turbine, turbine=True)
        states[5], n_comp_signed = self.turbomachine(
            states[4], 5, pressures[5], c.eta_compressor, turbine=False)
        n_comp = -n_comp_signed

        # Адиабатический баланс рекуператора при одинаковом массовом расходе.
        h3 = states[2].h_j_kg - (states[6].h_j_kg - states[5].h_j_kg)
        states[3] = self.state_ph(3, pressures[3], h3, g, states[2].t_k)

        q_reactor_loss = c.q_reactor_nom_w - q_reactor_fluid
        q_pipe = g * (states[0].h_j_kg - states[1].h_j_kg)
        q_cycle = g * (states[1].h_j_kg - states[6].h_j_kg)
        eta_pipeline_actual = q_cycle / q_reactor_fluid
        q_cool = g * (states[3].h_j_kg - states[4].h_j_kg)
        q_rec_h = g * (states[2].h_j_kg - states[3].h_j_kg)
        q_rec_c = g * (states[6].h_j_kg - states[5].h_j_kg)
        rec_balance = q_rec_h - q_rec_c
        pinch, pinch_x = self.refined_pinch(states) if refine_pinch else self.fast_pinch(states)

        n_t_shaft = n_turb * c.eta_turbine_mech
        n_t_el = n_t_shaft * c.eta_generator
        n_c_shaft = n_comp / c.eta_compressor_mech
        n_c_el = n_c_shaft / c.eta_compressor_drive
        n_el_gross = n_t_el - n_c_el
        n_pump = self.water_pump_power(q_cool)
        n_aux = n_pump + c.n_aux_other_w if np.isfinite(n_pump) else np.nan
        n_el_net = n_el_gross - n_aux if np.isfinite(n_aux) else np.nan
        n_fluid_net = n_turb - n_comp
        n_shaft_net = n_t_shaft - n_c_shaft
        n_losses = (n_turb - n_t_shaft) + (n_t_shaft - n_t_el) + \
                   (n_c_shaft - n_comp) + (n_c_el - n_c_shaft)
        cycle_balance = q_cycle - q_cool - n_fluid_net
        plant_balance = c.q_reactor_nom_w - q_reactor_loss - q_pipe - q_cool - n_fluid_net
        tol_q = max(1.0, 1e-6 * c.q_reactor_nom_w)

        checks = {
            "pinch below limit": np.isfinite(pinch) and pinch >= c.dt_pinch_min_c,
            "negative cooler duty": q_cool >= 0.0,
            "negative recuperator duty": q_rec_h >= -tol_q and q_rec_c >= -tol_q,
            "non-positive net power": np.isfinite(n_el_net) and n_el_net > 0.0,
            "non-positive shaft power": n_shaft_net > 0.0,
            "recuperator imbalance": abs(rec_balance) <= tol_q,
            "cycle imbalance": abs(cycle_balance) <= tol_q,
            "plant imbalance": abs(plant_balance) <= tol_q,
        }
        failed = [name for name, passed in checks.items() if not passed]

        eta_asmm_net = n_el_net / c.q_reactor_nom_w
        eta_cycle_after_reactor = n_el_net / q_reactor_fluid
        eta_cycle_after_pipeline = n_el_net / q_cycle

        row = base | {
            "valid": not failed, "reason": "; ".join(failed),
            "pressure_ratio": p1 / p2, "compressor_pressure_ratio": pressures[5] / pressures[4],
            "P3_MPa": pressures[3] / 1e6, "P4_MPa": pressures[4] / 1e6,
            "P5_MPa": pressures[5] / 1e6, "P6_MPa": pressures[6] / 1e6,
            "G_kg_s": g, "T_pinch_C": pinch, "pinch_x_L": pinch_x,
            "Q_R_nom_W": c.q_reactor_nom_w, "Q_R_W": q_reactor_fluid,
            "Q_R_loss_W": q_reactor_loss, "Q_pipe_loss_W": q_pipe,
            "Q_cycle_in_W": q_cycle, "Q_cool_W": q_cool,
            "eta_pipeline_actual": eta_pipeline_actual,
            "Q_recup_hot_W": q_rec_h, "Q_recup_cold_W": q_rec_c,
            "recup_balance_W": rec_balance,
            "N_T_i_W": n_turb, "N_K_i_consumed_W": n_comp,
            "specific_turbine_work_kJ_kg": n_turb / g / 1e3,
            "specific_compressor_work_kJ_kg": n_comp / g / 1e3,
            "specific_net_work_kJ_kg": n_fluid_net / g / 1e3,
            "N_T_shaft_W": n_t_shaft, "N_T_el_W": n_t_el,
            "N_K_shaft_W": n_c_shaft, "N_K_el_consumed_W": n_c_el,
            "N_shaft_net_W": n_shaft_net, "N_electromech_loss_W": n_losses,
            "N_el_gross_W": n_el_gross, "N_aux_pump_W": n_pump,
            "N_aux_other_W": c.n_aux_other_w, "N_aux_total_W": n_aux,
            "N_el_net_W": n_el_net, "N_net_fluid_W": n_fluid_net,
            "eta_el_gross": n_el_gross / q_reactor_fluid,
            "eta_asmm_net_initial_heat": eta_asmm_net,
            "eta_cycle_net_after_reactor": eta_cycle_after_reactor,
            "eta_cycle_net_after_pipeline": eta_cycle_after_pipeline,
            "eta_el_net_cycle": eta_cycle_after_reactor,
            "eta_el_net_cycle_Q2": eta_cycle_after_pipeline,
            "eta_net_plant": eta_asmm_net,
            "cycle_balance_W": cycle_balance, "plant_thermal_balance_W": plant_balance,
        }
        for i, state in sorted(states.items()):
            row.update({f"T{i}_C": state.t_c, f"T{i}_K": state.t_k,
                        f"P{i}_Pa": state.p_pa, f"h{i}_J_kg": state.h_j_kg,
                        f"s{i}_J_kgK": state.s_j_kgk})
        return (row, states) if return_states else row


_WORKER_MODEL: CycleModel | None = None

def _init_worker(props_path: str, cfg: CycleConfig) -> None:
    global _WORKER_MODEL
    _WORKER_MODEL = CycleModel(cfg, PropertyModel(Path(props_path), cfg.x_xe))

def _solve_worker(task: tuple[float, float, float]) -> dict:
    t6, p1, p2 = task
    try:
        assert _WORKER_MODEL is not None
        return _WORKER_MODEL.solve(p1, p2, t6)
    except Exception as exc:
        return {"valid": False, "reason": f"{type(exc).__name__}: {exc}",
                "T6_C": t6, "P1_MPa": p1, "P2_MPa": p2}

def run_sweep(model: CycleModel, workers: int = 1) -> pd.DataFrame:
    c = model.c
    total = len(c.t6_grid_c) * len(c.p1_grid_mpa) * len(c.p2_grid_mpa)
    tasks = product(c.t6_grid_c, c.p1_grid_mpa, c.p2_grid_mpa)
    rows: list[dict] = []
    bar_options = dict(total=total, desc="Расчёт режимов", unit="точка",
                       dynamic_ncols=True, smoothing=0.1)
    if workers <= 1:
        for task in tqdm(tasks, **bar_options):
            t6, p1, p2 = task
            try:
                rows.append(model.solve(p1, p2, t6))
            except Exception as exc:
                rows.append({"valid": False, "reason": f"{type(exc).__name__}: {exc}",
                             "T6_C": t6, "P1_MPa": p1, "P2_MPa": p2})
    else:
        chunksize = max(1, total // (workers * 100))
        context = mp.get_context("spawn")
        with context.Pool(workers, initializer=_init_worker,
                          initargs=(str(model.p.path), c)) as pool:
            iterator = pool.imap_unordered(_solve_worker, tasks, chunksize=chunksize)
            rows.extend(tqdm(iterator, **bar_options))

    rows.sort(key=lambda row: (row.get("T6_C", np.inf), row.get("P1_MPa", np.inf),
                               row.get("P2_MPa", np.inf)))
    return pd.DataFrame(rows)

def best_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = df.loc[df["valid"].eq(True)].copy()
    if valid.empty:
        return valid, valid
    best = valid.loc[[valid["eta_asmm_net_initial_heat"].idxmax()]]
    by_t6 = valid.loc[
        valid.groupby("T6_C")["eta_asmm_net_initial_heat"].idxmax()
    ].sort_values("T6_C")
    return best, by_t6

def _format_worksheet(ws) -> None:
    ws.freeze_panes = "A2"
    ws.sheet_view.showGridLines = False
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF", size=9)
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 34
    if ws.max_row >= 2 and ws.max_column >= 1:
        ws.auto_filter.ref = ws.dimensions
    for cells in ws.iter_cols():
        header = str(cells[0].value or "")
        samples = [len(str(c.value)) for c in cells[1:50] if c.value is not None]
        ws.column_dimensions[get_column_letter(cells[0].column)].width = min(
            max(11, len(header) * 0.75, max(samples, default=0) * 0.7), 28)
        fmt = "0.0000"
        if header.endswith("_W") or header.endswith("_Pa"):
            fmt = '#,##0.000;[Red]-#,##0.000'
        elif "eta_" in header:
            fmt = "0.00000"
        if ws.max_row <= 5000:
            for cell in cells[1:]:
                if isinstance(cell.value, (int, float)):
                    cell.number_format = fmt

def export_excel(df: pd.DataFrame, profile: pd.DataFrame, states: pd.DataFrame,
                 cfg: CycleConfig, path: Path) -> None:
    safe_df = df.replace([np.inf, -np.inf], np.nan)
    best, by_t6 = best_rows(safe_df)
    valid = safe_df.loc[safe_df["valid"].eq(True)].copy()
    kpi = pd.DataFrame({
        "Показатель": [
            "КПД нетто АСММ (основной)",
            "КПД нетто после потерь реактора",
            "КПД нетто после потерь реактора и трубопровода",
        ],
        "Столбец": [
            "eta_asmm_net_initial_heat",
            "eta_cycle_net_after_reactor",
            "eta_cycle_net_after_pipeline",
        ],
        "Формула": [
            "N_el_net / Q_R_nom",
            "N_el_net / Q_R",
            "N_el_net / Q_cycle_in",
        ],
    })
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        safe_df.to_excel(writer, sheet_name="Все режимы", index=False)
        valid.to_excel(writer, sheet_name="Допустимые режимы", index=False)
        best.to_excel(writer, sheet_name="Оптимум", index=False)
        by_t6.to_excel(writer, sheet_name="Оптимум по T6", index=False)
        profile.to_excel(writer, sheet_name="Профиль рекуператора", index=False)
        states.to_excel(writer, sheet_name="Состояния оптимума", index=False)
        kpi.to_excel(writer, sheet_name="Определения КПД", index=False)
        pd.DataFrame({"parameter": list(asdict(cfg)),
                      "value": [json.dumps(v) if isinstance(v, tuple) else v
                                for v in asdict(cfg).values()]}).to_excel(
                                    writer, sheet_name="Исходные данные", index=False)
    wb = load_workbook(path)
    for ws in wb.worksheets:
        _format_worksheet(ws)
    for name in ("Все режимы", "Допустимые режимы"):
        ws = wb[name]
        headers = {cell.value: cell.column for cell in ws[1]}
        for column in ("eta_asmm_net_initial_heat", "N_el_net_W", "T_pinch_C"):
            if column in headers and ws.max_row > 2:
                letter = get_column_letter(headers[column])
                ws.conditional_formatting.add(
                    f"{letter}2:{letter}{ws.max_row}",
                    ColorScaleRule(start_type="min", start_color="F8696B",
                                   mid_type="percentile", mid_value=50, mid_color="FFEB84",
                                   end_type="max", end_color="63BE7B"))
    wb.save(path)


def export_interactive(df: pd.DataFrame, out_dir: Path) -> None:
    """Self-contained Plotly dashboard plus a filterable HTML data table."""
    valid = df.loc[df["valid"].eq(True)].copy()
    if valid.empty:
        return
    valid["eta_asmm_net_pct"] = 100.0 * valid["eta_asmm_net_initial_heat"]
    fig = px.scatter_3d(valid, x="P2_MPa", y="P1_MPa", z="eta_asmm_net_pct",
                        color="T6_C", size="N_el_net_W",
                        hover_data=["T_pinch_C", "G_kg_s"],
                        labels={"eta_asmm_net_pct": "КПД нетто АСММ, %",
                                "P1_MPa": "P1, МПа", "P2_MPa": "P2, МПа",
                                "T6_C": "T6, °C"},
                        title="Интерактивная карта режимов цикла He–Xe")
    fig.write_html(out_dir / "interactive_cycle_map.html", include_plotlyjs=True)

    compact = valid[[c for c in (
        "T6_C", "P1_MPa", "P2_MPa", "pressure_ratio", "G_kg_s", "T_pinch_C",
        "N_T_i_W", "N_K_i_consumed_W", "N_el_net_W", "eta_asmm_net_initial_heat") if c in valid]]
    table = compact.to_html(index=False, table_id="cycle", classes="display compact", float_format="%.6g")
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Таблица режимов цикла He-Xe</title>
<link rel='stylesheet' href='https://cdn.datatables.net/2.1.8/css/dataTables.dataTables.min.css'>
<style>body{{font-family:Arial,sans-serif;margin:20px;font-size:13px}} input{{width:95%;box-sizing:border-box}}</style>
</head><body><h2>Допустимые режимы цикла He-Xe</h2>{table}
<script src='https://code.jquery.com/jquery-3.7.1.min.js'></script>
<script src='https://cdn.datatables.net/2.1.8/js/dataTables.min.js'></script>
<script>$(document).ready(function(){{
$('#cycle thead tr').clone(true).appendTo('#cycle thead');
$('#cycle thead tr:eq(1) th').each(function(i){{$(this).html('<input type="text" placeholder="Filter"/>');
$('input',this).on('keyup change',function(){{if(table.column(i).search()!==this.value) table.column(i).search(this.value).draw();}});}});
var table=$('#cycle').DataTable({{orderCellsTop:true,pageLength:25,scrollX:true}});
}});</script></body></html>"""
    (out_dir / "interactive_cycle_table.html").write_text(html, encoding="utf-8")


def save_figure(fig: plt.Figure, out_dir: Path, name: str,
                formats: Iterable[str] = FIGURE["formats"]) -> None:
    for extension in formats:
        fig.savefig(out_dir / f"{name}.{extension}", dpi=FIGURE["dpi"])
    plt.close(fig)


def scientific_legend(ax, *, loc: str = "best", ncol: int = 1,
                      outside: bool = False) -> None:
    kwargs = {"loc": loc, "ncol": ncol, "frameon": True, "framealpha": 0.96,
              "facecolor": "white", "edgecolor": "0.65", "fancybox": False}
    if outside:
        kwargs["bbox_to_anchor"] = (0.5, -0.18)
    ax.legend(**kwargs)


def annotate_optimum(ax, best: pd.Series, x: float, y: float, text: str) -> None:
    ax.plot(x, y, marker="*", ms=9, color="crimson", label="Глобальный оптимум", zorder=6)
    ax.annotate(text, xy=(x, y), xytext=(8, 10), textcoords="offset points",
                fontsize=FONT["annotation"], ha="left", va="bottom",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.35", "lw": 0.7},
                arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "0.25"})


def contour_map(df: pd.DataFrame, best: pd.Series, column: str, scale: float,
                label: str, title: str, out_dir: Path, name: str,
                aspect: tuple[float, float] = FIGURE["double_column"]) -> None:
    """Contour map at the globally optimal T6 slice."""
    data = df.loc[df["T6_C"].eq(best["T6_C"]) & df["valid"].eq(True)]
    pivot = data.pivot_table(index="P1_MPa", columns="P2_MPa", values=column)
    if pivot.empty:
        return
    x, y, z = pivot.columns.to_numpy(float), pivot.index.to_numpy(float), pivot.to_numpy(float) * scale
    fig, ax = plt.subplots(figsize=aspect)
    contour = ax.contourf(x, y, np.ma.masked_invalid(z), levels=20, cmap=FIGURE["cmap"])
    ax.contour(x, y, np.ma.masked_invalid(z), levels=10, colors="k", linewidths=0.25, alpha=0.45)
    cbar = fig.colorbar(contour, ax=ax, pad=0.02)
    cbar.set_label(label)
    ax.set(xlabel=r"Давление после турбины, $P_2$ (МПа)",
           ylabel=r"Давление перед турбиной, $P_1$ (МПа)", title=f"{title}; $T_6$ = {best['T6_C']:.0f} °C")
    value = best[column] * scale
    annotate_optimum(ax, best, best["P2_MPa"], best["P1_MPa"],
                     f"Оптимум\n{label.split('(')[0].strip()} = {value:.3g}")
    scientific_legend(ax, loc="lower right")
    fig.tight_layout()
    save_figure(fig, out_dir, name)


def plot_efficiency_surface(df: pd.DataFrame, best: pd.Series, out_dir: Path,
                            aspect: tuple[float, float] = FIGURE["double_column"]) -> None:
    data = df.loc[df["T6_C"].eq(best["T6_C"]) & df["valid"].eq(True)]
    pivot = data.pivot_table(index="P1_MPa", columns="P2_MPa", values="eta_asmm_net_initial_heat")
    if pivot.empty:
        return
    x, y = np.meshgrid(pivot.columns.to_numpy(float), pivot.index.to_numpy(float))
    z = 100.0 * pivot.to_numpy(float)
    fig = plt.figure(figsize=aspect)
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(x, y, np.ma.masked_invalid(z), cmap=FIGURE["cmap"],
                              linewidth=0, antialiased=True, alpha=0.92)
    bz = 100.0 * best["eta_asmm_net_initial_heat"]
    ax.scatter(best["P2_MPa"], best["P1_MPa"], bz, marker="*", s=80, color="crimson",
               label="Глобальный оптимум", depthshade=False)
    ax.text(best["P2_MPa"], best["P1_MPa"], bz,
            f"  max = {bz:.3f}%\n  ({best['P1_MPa']:.2f}, {best['P2_MPa']:.2f}) MPa",
            fontsize=FONT["annotation"])
    ax.set(xlabel=r"$P_2$ (МПа)", ylabel=r"$P_1$ (МПа)", zlabel=r"КПД нетто АСММ, $\eta_{нетто}$ (%)",
           title=f"Электрический КПД нетто АСММ; $T_6$ = {best['T6_C']:.0f} °C")
    fig.colorbar(surface, ax=ax, shrink=0.70, pad=0.10, label="КПД (%)")
    scientific_legend(ax, loc="upper left")
    fig.tight_layout()
    save_figure(fig, out_dir, "efficiency_3d_surface")


def plot_recuperator(profile: pd.DataFrame, best: pd.Series, out_dir: Path,
                     aspect: tuple[float, float] = FIGURE["single_column"]) -> None:
    fig, ax = plt.subplots(figsize=aspect)
    ax.plot(profile["x_L"], profile["T_hot_C"], color="#B2182B", label="Горячий поток (2→3)")
    ax.plot(profile["x_L"], profile["T_cold_C"], color="#2166AC", label="Холодный поток (6→5, противоток)")
    j = profile["delta_T_K"].idxmin()
    x = profile.loc[j, "x_L"]
    th, tc, dt = profile.loc[j, ["T_hot_C", "T_cold_C", "delta_T_K"]]
    ax.vlines(x, tc, th, colors="crimson", linestyles="--", lw=1.0, label="Минимальный напор")
    ax.annotate(fr"$\Delta T_{{min}}$ = {dt:.2f} K" + "\n" + fr"$x/L$ = {x:.3f}", xy=(x, (th + tc) / 2),
                xytext=(10, 4), textcoords="offset points", fontsize=FONT["annotation"],
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.35", "lw": 0.7})
    ax.set(xlabel="Относительная координата горячего потока, x/L (–)", ylabel="Температура (°C)",
           title="Температурный профиль рекуператора в оптимуме")
    scientific_legend(ax, loc="best")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()
    save_figure(fig, out_dir, "recuperator_temperature_profile")


def plot_energy_balance(best: pd.Series, out_dir: Path,
                        aspect: tuple[float, float] = FIGURE["double_column"]) -> None:
    labels = ("Потери реактора", "Потери трубопровода", "Отвод в охладителе",
              "Электромех. потери", "Собственные нужды", "Электрическая мощность нетто")
    values = np.array([
        best["Q_R_loss_W"], best["Q_pipe_loss_W"], best["Q_cool_W"],
        best["N_electromech_loss_W"], best["N_aux_total_W"],
        best["N_el_net_W"],
    ]) / 1e6
    fig, ax = plt.subplots(figsize=aspect)
    colors = mpl.colormaps["tab20c"](np.linspace(0.05, 0.85, len(values)))
    bars = ax.bar(np.arange(len(values)), values, color=colors, edgecolor="0.2", linewidth=0.45)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=2, fontsize=FONT["annotation"])
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set(ylabel="Мощность, МВт", title="Энергетический баланс в глобальном оптимуме")
    ax.axhline(0, color="0.2", lw=0.7)
    ax.text(0.99, 0.96,
            fr"Максимум $\eta_{{net}}$ = {100*best['eta_asmm_net_initial_heat']:.3f}%" + "\n" +
            f"$P_1$ = {best['P1_MPa']:.2f} МПа; $P_2$ = {best['P2_MPa']:.2f} МПа",
            transform=ax.transAxes, ha="right", va="top", fontsize=FONT["annotation"],
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.35", "lw": 0.7})
    fig.tight_layout()
    save_figure(fig, out_dir, "energy_balance_optimum")


def plot_specific_works(df: pd.DataFrame, best: pd.Series, out_dir: Path,
                        aspect: tuple[float, float] = FIGURE["double_column"]) -> None:
    data = df.loc[df["valid"].eq(True)].copy()
    fig, axes = plt.subplots(1, 2, figsize=aspect)
    for t6, group in data.groupby("T6_C"):
        reduced = group.groupby("pressure_ratio", as_index=False).agg(
            specific_turbine_work_kJ_kg=("specific_turbine_work_kJ_kg", "max"),
            specific_compressor_work_kJ_kg=("specific_compressor_work_kJ_kg", "min"),
            eta_asmm_net_initial_heat=("eta_asmm_net_initial_heat", "max"))
        reduced = reduced.sort_values("pressure_ratio")
        axes[0].plot(reduced["pressure_ratio"], reduced["specific_turbine_work_kJ_kg"], label=f"T6={t6:.0f} °C")
        axes[1].plot(reduced["pressure_ratio"], 100 * reduced["eta_asmm_net_initial_heat"], label=f"T6={t6:.0f} °C")
    axes[0].set(xlabel=r"Степень расширения турбины, $P_1/P_2$ (–)", ylabel="Максимальная удельная работа турбины, кДж/кг",
                title="Удельная работа турбины")
    axes[1].set(xlabel=r"Степень расширения турбины, $P_1/P_2$ (–)", ylabel="Максимальный КПД нетто АСММ, %",
                title="Влияние степени расширения")
    axes[1].plot(best["pressure_ratio"], 100 * best["eta_asmm_net_initial_heat"], "*", color="crimson", ms=9, label="Глобальный оптимум")
    axes[1].annotate(f"макс. = {100*best['eta_asmm_net_initial_heat']:.3f}%", xy=(best["pressure_ratio"], 100*best["eta_asmm_net_initial_heat"]),
                     xytext=(8, 8), textcoords="offset points", fontsize=FONT["annotation"])
    for ax in axes:
        scientific_legend(ax, loc="upper center", ncol=3, outside=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()
    save_figure(fig, out_dir, "pressure_ratio_and_specific_work")


def make_all_figures(df: pd.DataFrame, profile: pd.DataFrame, best: pd.Series,
                     out_dir: Path) -> None:
    plot_efficiency_surface(df, best, out_dir)
    contour_map(df, best, "eta_asmm_net_initial_heat", 100.0, "КПД нетто АСММ, %",
                "Электрический КПД нетто АСММ", out_dir, "efficiency_contour")
    contour_map(df, best, "N_T_el_W", 1e-6, "Электрическая мощность турбины, МВт",
                "Электрическая мощность турбины", out_dir, "turbine_power_contour")
    contour_map(df, best, "N_K_el_consumed_W", 1e-6, "Потребление компрессора, МВт",
                "Электрическая мощность компрессора", out_dir, "compressor_power_contour")
    plot_recuperator(profile, best, out_dir)
    plot_energy_balance(best, out_dir)
    plot_specific_works(df, best, out_dir)


def states_frame(states: dict[int, State]) -> pd.DataFrame:
    return pd.DataFrame([asdict(state) | {"t_c": state.t_c} for state in states.values()]).sort_values("node")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", type=Path, default=Path(__file__).with_name("props.npy"),
                        help="Path to props.npy (default: next to this script).")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("res_hexe"),
                        help="Output directory.")
    parser.add_argument("--workers", type=int, default=0,
                        help="Worker processes: 0=auto, 1=sequential (default: 0).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = CycleConfig()
    cfg.validate()
    apply_plot_style()
    model = CycleModel(cfg, PropertyModel(args.props, cfg.x_xe))
    workers = args.workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    if workers < 1:
        raise ValueError("--workers must be 0 or a positive integer.")
    print(f"Processes: {workers}; points: {len(cfg.t6_grid_c) * len(cfg.p1_grid_mpa) * len(cfg.p2_grid_mpa)}")
    df = run_sweep(model, workers=workers)
    best_df, _ = best_rows(df)
    if best_df.empty:
        export_excel(df, pd.DataFrame(), pd.DataFrame(), cfg, args.out / "cycle_results.xlsx")
        raise RuntimeError("No valid modes. Inspect the 'reason' column in cycle_results.xlsx.")

    best_index = best_df.index[0]
    candidate = best_df.iloc[0]
    refined, states = model.solve(candidate["P1_MPa"], candidate["P2_MPa"],
                                  candidate["T6_C"], return_states=True, refine_pinch=True)
    for key, value in refined.items():
        df.loc[best_index, key] = value
    best = df.loc[best_index]
    profile = model.recuperator_profile(states, n=max(300, cfg.pinch_points))
    state_table = states_frame(states)
    df.to_csv(args.out / "pressure_sweep_full.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(args.out / "recuperator_profile.csv", index=False, encoding="utf-8-sig")
    state_table.to_csv(args.out / "optimum_states.csv", index=False, encoding="utf-8-sig")
    export_excel(df, profile, state_table, cfg, args.out / "cycle_results.xlsx")
    export_interactive(df, args.out)
    make_all_figures(df, profile, best, args.out)

    print("\nГлобальный оптимум")
    print(f"T6 = {best['T6_C']:.1f} °C; P1 = {best['P1_MPa']:.3f} МПа; P2 = {best['P2_MPa']:.3f} МПа")
    print(f"Электрическая мощность нетто = {best['N_el_net_W']/1e6:.5f} МВт")
    print(f"КПД нетто АСММ по начальному теплу = {100*best['eta_asmm_net_initial_heat']:.5f} %")
    print(f"Теплопотери трубопровода = {best['Q_pipe_loss_W']/1e3:.3f} кВт")
    print(f"Расчётный КПД трубопровода = {100*best['eta_pipeline_actual']:.5f} %")
    print(f"Минимальный напор рекуператора = {best['T_pinch_C']:.4f} К")
    print(f"Результаты: {args.out.resolve()}")


if __name__ == "__main__":
    main()
