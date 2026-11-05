import numpy as np, pandas as pd, os
from scipy.optimize import newton, bisect, least_squares
from multiprocessing import Pool
from time import time

# Q-тепловая мощность, x-мольная доля Xe, T_max-темп. реактора, T_min-темп. охладителя
Q, x, T_max, T_min, N_pinch = 2.75e6, 0.283, 1060, 35, 20
dP_heat_source = dict(Heat=0.0364e-2, Cool=0.497e-2)   # потери в нагревателе/охладителе
dP_water, P_water, dT_water = 5e-2, 6e5, 10             # параметры водяного контура
dP_fix, eff_fix = 4.553e-2, dict(Turb=92.67e-2, Pump=87.06e-2)  # потери в рекуператоре, КПД турбины/компрессора

# Сетки перебора
P_in_line = np.arange(1, 5.1, 0.1, dtype=np.float64) * 1e6      # P на входе в турбину
P_out_line = np.arange(1, 3.1, 0.1, dtype=np.float64) * 1e6     # P на выходе из турбины
T_before_line = np.arange(650, 900, 50, dtype=np.float64)       # T перед турбиной
dT_line = np.arange(5, 16, 1, dtype=np.float64)                 # отсечка T_pinch
dP_line = np.arange(0.03, 0.07, 0.01, dtype=np.float64)         # потери давления
eff_line = dict(Turb=np.arange(85, 100, 5, dtype=np.float64)/1e2,
                Pump=np.arange(75, 95, 5, dtype=np.float64)/1e2)
data_stack = [(P_in, P_out, T_before, dT, dP_fix, eff_fix)
              for P_in in P_in_line for P_out in P_out_line
              for T_before in T_before_line for dT in dT_line]

# Загрузка свойств
PARAMS = ['D', 'H', 'S', 'Cp', 'Vis', 'Tcx']
props = dict(zip(PARAMS, np.load('props.npy', allow_pickle=True)))
props_water = dict(zip(PARAMS, np.load('props_water.npy', allow_pickle=True)))
T_min_props, T_max_props = min(props['H'].grid[2]), max(props['H'].grid[2])

def T_err(fluid, param_name, target, x, P, T_guess):
    # Поиск T по заданному свойству (H или S) методом Ньютона с фолбэком на бисекцию
    err = lambda T: target - fluid[param_name]((x, P, T))
    try: return newton(err, T_guess, tol=1e-6)
    except Exception:
        try: return bisect(err, max(T_guess-200, T_min_props), min(T_guess+200, T_max_props), xtol=1e-4)
        except Exception: return bisect(err, T_min_props, T_max_props, xtol=1e-4)

class Node:
    # Узел цикла: T, P, x, G и свойства D, H, S, Cp, Vis, Tcx
    def __init__(self, idx): self.idx, self.T, self.P, self.G = idx, np.nan, np.nan, np.nan
    def set(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)
        for p in PARAMS: setattr(self, p, props[p]((self.x, self.P, self.T)))

class Cycle:
    # Цикл Брайтона: 6 узлов (1..6) и 5 блоков
    def __init__(self):
        self.nodes = {i: Node(i) for i in range(1, 7)}
        self.blocks = {n: dict(N=np.nan, Q=np.nan) for n in ['Heat','Turb','Cool','Pump','Exch']}
        self.T_pinch = np.nan

    def node(self, idx, **kw):
        # Заполнить узел и интерполировать свойства
        self.nodes[idx].x = x; self.nodes[idx].set(**kw)

    def expander(self, name, i1, i2, eff):
        # Турбина (Turb) или компрессор (Pump): изоэнтропа + реальный КПД
        n1, n2 = self.nodes[i1], self.nodes[i2]
        T2 = n1.T * (n2.P / n1.P) ** 0.4
        H2t = props['H']((x, n2.P, T_err(props, 'S', n1.S, x, n2.P, T2)))
        H2 = n1.H*(1-eff[name]) + H2t*eff[name] if name=='Turb' else n1.H*(1-1/eff[name]) + H2t/eff[name]
        self.node(i2, P=n2.P, T=T_err(props, 'H', H2, x, n2.P, T2), G=n1.G)
        self.blocks[name]['N'] = self.nodes[i2].G * (n1.H - self.nodes[i2].H)

    def heat_source(self, name, i1, i2):
        # Нагреватель (Heat) или охладитель (Cool): Q = G·ΔH
        n1 = self.nodes[i1]
        if name=='Heat': self.nodes[i1].P = self.nodes[i2].P * (1 + dP_heat_source[name])
        elif name=='Cool': self.nodes[i2].P = self.nodes[i1].P * (1 - dP_heat_source[name])
        self.node(i2, P=self.nodes[i2].P, T=self.nodes[i2].T, G=n1.G)
        self.blocks[name]['Q'] = self.nodes[i2].G * (self.nodes[i2].H - n1.H)

    def recuperation(self, args, i11, i12, i21, i22, dT, dP, eff):
        # Рекуператор: подбор T на горячем выходе (i12), баланс H, T_pinch
        n = self.nodes
        n[i12].T = args[0]
        n[i12].P, n[i21].P = n[i11].P*(1-dP), n[i22].P*(1+dP)
        self.node(i12, P=n[i12].P, T=n[i12].T, G=n[i11].G)
        self.heat_source('Cool', i12, i12+1)
        self.expander('Pump', i21-1, i21, eff)
        self.node(i22, P=n[i22].P, T=n[i22].T, G=n[i21].G)
        self.heat_source('Heat', 6, 1)
        H_rec, P_rec, T_rec = [np.vstack([np.linspace(n[i11].__dict__[p], n[i12].__dict__[p], N_pinch), np.linspace(n[i22].__dict__[p], n[i21].__dict__[p], N_pinch)])
                               for p in ['H','P','T']]
        T_rec = T_err(props, 'H', H_rec, x, P_rec, T_rec)
        self.T_pinch = np.min(T_rec[0] - T_rec[1])
        Q_rec = n[i11].G*(n[i11].H - n[i12].H) + n[i21].G*(n[i21].H - n[i22].H)
        self.blocks['Exch']['Q'] = Q_rec
        return [Q_rec]

    def brayton(self, P_in, P_out, T_before, dT, dP, eff):
        # Полный цикл: задать P/T, расход, турбина, рекуператор, КПД
        if P_in <= P_out: return np.nan
        try:
            n = self.nodes
            for i in range(1, 7): n[i].x = x
            n[1].P, n[2].P = P_in, P_out
            n[6].P = P_in * (1 + dP_heat_source['Heat'])
            n[1].T, n[4].T, n[6].T = T_max+273.15, T_min+273.15, T_before+273.15
            G = Q / (props['H']((x, n[1].P, n[1].T)) - props['H']((x, n[6].P, n[6].T)))
            n[1].G = G
            self.node(1, T=n[1].T, P=n[1].P, G=G)
            self.expander('Turb', 1, 2, eff)
            least_squares(self.recuperation, [0.5*(n[2].T + n[4].T)],
                          args=(2, 3, 5, 6, dT, dP, eff),
                          bounds=(n[4].T, n[2].T), xtol=1e-4)
            if np.min(self.T_pinch) < dT: return np.nan
        except Exception: return np.nan
        P_cool, T_cool = P_water*(1 - 0.5*dP_water), n[4].T + dT_water*0.5
        N_cool = (self.blocks['Cool']['Q'] * dP_water * P_water /
                  (props_water['Cp']((1, P_cool, T_cool)) *
                   props_water['D']((1, P_cool, T_cool)) *
                   dT_water * 0.8 * 0.95))
        KPD_brutto = (self.blocks['Turb']['N'] + self.blocks['Pump']['N']) * 0.98 / self.blocks['Heat']['Q']
        KPD_netto = KPD_brutto + N_cool / self.blocks['Heat']['Q']
        self.N_cool = N_cool; self.KPD_brutto = KPD_brutto; self.KPD_netto = KPD_netto
        return KPD_brutto

def run_brayton(*args):
    # Обёртка для Pool: создаёт свежий Cycle
    return Cycle().brayton(*args)

def export_cycle(cycle, path='res/cycle_results.xlsx'):
    # Выгрузка узлов, блоков и сводки в Excel
    os.makedirs('res', exist_ok=True)
    rows = []
    for i, nd in cycle.nodes.items():
        rows.append({'Узел': i, 'T, К': nd.T, 'T, °C': nd.T-273.15, 'P, Па': nd.P, 'P, МПа': nd.P/1e6,
                     'P, бар': nd.P/1e5, 'x': x, 'G, кг/с': nd.G,
                     'D, кг/м³': nd.D, 'H, Дж/кг': nd.H, 'S, Дж/(кг·К)': nd.S,
                     'Cp, Дж/(кг·К)': nd.Cp, 'Vis, мкПа·с': nd.Vis, 'Tcx, Вт/(м·К)': nd.Tcx})
    df_nodes = pd.DataFrame(rows)
    rows = []
    for name, b in cycle.blocks.items():
        rows.append({'Элемент': name, 'Q, Вт': b['Q'], 'Q, кВт': b['Q']/1e3 if pd.notna(b['Q']) else np.nan,
                     'N, Вт': b['N'], 'N, кВт': b['N']/1e3 if pd.notna(b['N']) else np.nan})
    df_blocks = pd.DataFrame(rows)
    df_summary = pd.DataFrame([
        {'Параметр': 'KPD_brutto', 'Значение': getattr(cycle, 'KPD_brutto', np.nan), 'Ед.': '—'},
        {'Параметр': 'KPD_netto',  'Значение': getattr(cycle, 'KPD_netto',  np.nan), 'Ед.': '—'},
        {'Параметр': 'T_pinch',    'Значение': cycle.T_pinch, 'Ед.': 'К'},
        {'Параметр': 'N_cool',     'Значение': getattr(cycle, 'N_cool', np.nan), 'Ед.': 'Вт'},
        {'Параметр': 'Q_heat',     'Значение': cycle.blocks['Heat']['Q'], 'Ед.': 'Вт'},
        {'Параметр': 'N_turb',     'Значение': cycle.blocks['Turb']['N'], 'Ед.': 'Вт'},
        {'Параметр': 'N_pump',     'Значение': cycle.blocks['Pump']['N'], 'Ед.': 'Вт'},
        {'Параметр': 'Q_exch',     'Значение': cycle.blocks['Exch']['Q'], 'Ед.': 'Вт'},
        {'Параметр': 'Q_cool',     'Значение': cycle.blocks['Cool']['Q'], 'Ед.': 'Вт'},
    ])
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        df_nodes.to_excel(w, sheet_name='Nodes', index=False)
        df_blocks.to_excel(w, sheet_name='Blocks', index=False)
        df_summary.to_excel(w, sheet_name='Summary', index=False)
    return path

if __name__ == '__main__':
    start = time()
    with Pool(14) as p:
        KPD = p.starmap(run_brayton, data_stack)
    KPD = np.array(KPD).reshape(len(P_in_line), len(P_out_line), len(T_before_line), len(dT_line))
    os.makedirs('res', exist_ok=True)
    # Полная таблица перебора
    pd.DataFrame(np.vstack([np.hstack([Pi, Po, Tb, dTv, KPD[i,j,k,l]])
                            for i, Pi in enumerate(P_in_line)
                            for j, Po in enumerate(P_out_line)
                            for k, Tb in enumerate(T_before_line)
                            for l, dTv in enumerate(dT_line)]),
                 columns=['P_in','P_out','T_before','dT','KPD']).to_excel('res/full_stack.xlsx', index=False)
    # Оптимум
    opt = np.argwhere(KPD == np.nanmax(KPD))[0]
    dir_opt = f'res/optimum/P_in={P_in_line[opt[0]]:.1e}, P_out={P_out_line[opt[1]]:.1e}, T_before={T_before_line[opt[2]]}, dT={dT_line[opt[3]]}'
    os.makedirs(dir_opt, exist_ok=True)
    # Прогон оптимума + экспорт узлов/блоков
    cycle = Cycle()
    cycle.brayton(P_in_line[opt[0]], P_out_line[opt[1]], T_before_line[opt[2]], dT_line[opt[3]], dP_fix, eff_fix)
    export_cycle(cycle, f'{dir_opt}/cycle_results.xlsx')
    pd.DataFrame({'dT': dT_line,'KPD': [np.nanmax(KPD[:,:,:,i]) for i in range(len(dT_line))]}).to_excel(f'{dir_opt}/dT.xlsx', index=False)
    print('Готово:', time()-start, 'с')