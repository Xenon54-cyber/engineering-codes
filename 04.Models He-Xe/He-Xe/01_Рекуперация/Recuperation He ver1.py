import numpy as np
import pandas as pd
from scipy.optimize import newton, bisect
from matplotlib import pyplot as plt
from multiprocessing import Pool
from time import time
import os
from scipy.interpolate import griddata

FLUIDS = np.array(['He-Xe', 'Water'])

Q = 2.75e6
x = 0.283
T_max = 1060 # °C
T_after_pipe = 1030 # °C
T_min = 35  # °C
N_pinch = 20
dP_heat_source = dict(Heat=0.0364e-2, Cool=0.497e-2)

dP_water, P_water, dT_water = 5e-2, 6e5, 10
dP_fix = 4.553e-2
eff_fix = dict(Turb=92.67e-2, Pump=87.06e-2)

P_in_line = np.arange(30, 51, 1, dtype=np.float64) * 1e5
P_out_line = np.arange(10, 31, 1, dtype=np.float64) * 1e5
T_before_line = np.arange(650, 851, 1, dtype=np.float64)
dT_line = np.arange(5, 16, 1, dtype=np.float64)

dP_line  = np.arange(3, 7, 1, dtype=np.float64) * 1e-2
eff_line = dict(Turb=np.arange(85, 93, 1, dtype=np.float64) / 1e2,
    Pump=np.arange(75, 91, 1, dtype=np.float64) / 1e2,)

data_stack = [(P_in, P_out, T_before, dT, dP_fix, eff_fix)
    for P_in in P_in_line
    for P_out in P_out_line
    for T_before in T_before_line
    for dT in dT_line
]

PARAMS = ['D', 'H', 'S', 'Cp', 'Vis', 'Tcx']
PROPS = {FLUIDS[0]: dict(zip(PARAMS, np.load('props.npy',       allow_pickle=True))),FLUIDS[1]: dict(zip(PARAMS, np.load('props_water.npy', allow_pickle=True))),}

class Node:
    """Узел цикла: T, P, x, G и свойства D, H, S, Cp, Vis, Tcx."""
    def __init__(self, idx, fluid=FLUIDS[0], x=x):
        self.idx   = idx
        self.fluid = fluid
        self.x = x
        self.T = np.nan
        self.P = np.nan
        self.G = np.nan
        self.D = np.nan
        self.H = np.nan
        self.S = np.nan
        self.Cp = np.nan
        self.Vis = np.nan
        self.Tcx = np.nan

    def set(self, **kwargs):
        """Задать T, P, G и интерполировать свойства."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._interpolate()

    def _interpolate(self):
        """Интерполяция свойств в текущем состоянии."""
        for p in PARAMS:
            setattr(self, p, PROPS[self.fluid][p]((self.x, self.P, self.T)))

    def __repr__(self):
        return (f'Node({self.idx}): T={self.T:.2f} K, P={self.P/1e5:.2f} bar, '
                f'G={self.G:.4f} kg/s')

# Класс Block — элемент цикла
class Block:
    """Элемент цикла: Q (теплота), N (мощность), Aux (вспомогательная)."""

    def __init__(self, name):
        self.name = name
        self.Q   = np.nan
        self.N   = np.nan
        self.Aux = np.nan

    def __repr__(self):
        return f'Block({self.name}): Q={self.Q}, N={self.N}, Aux={self.Aux}'

# Вспомогательные функции

def T_err(fluid, param_name, target, x, P, T_guess):
    """Найти T по заданному свойству (H или S) методом Ньютона."""
    err = lambda T: target - PROPS[fluid][param_name]((x, P, T))
    try:
        return newton(err, T_guess, tol=1e-6)
    except Exception:
        T_min_p = min(PROPS[fluid][param_name].grid[2])
        T_max_p = max(PROPS[fluid][param_name].grid[2])
        T_lo = max(T_guess - 200, T_min_p)
        T_hi = min(T_guess + 200, T_max_p)
        try:
            return bisect(err, T_lo, T_hi, xtol=1e-6)
        except Exception:
            return bisect(err, T_min_p, T_max_p, xtol=1e-6)

# Класс Cycle — цикл Брайтона

class Cycle:

    def __init__(self):
        self.nodes  = {i: Node(i) for i in range(7)}
        self.blocks = {name: Block(name)
                       for name in ['Heat', 'Turb', 'Cool', 'Pump', 'Exch', 'Pipe']}
        self.T_pinch = np.nan

    # Тепловой источник/сток

    def heat_source(self, name, node1, node2):
        n1 = self.nodes[node1]
        n2 = self.nodes[node2]
        n2.set(P=n2.P, T=n2.T, G=n1.G)
        self.blocks[name].Q = n2.G * (n2.H - n1.H)


    # Турбина / компрессор

    def expander(self, name, node1, node2, eff):
        n1 = self.nodes[node1]
        n2 = self.nodes[node2]

        P2 = n2.P
        S1 = n1.S
        T2 = n1.T * (P2 / n1.P) ** 0.4
        H2t = PROPS[FLUIDS[0]]['H']((n2.x, P2,T_err(FLUIDS[0], 'S', S1, n2.x, P2, T2)))

        if name == 'Turb':
            H2 = n1.H * (1 - eff[name]) + H2t * eff[name]
        else:
            H2 = n1.H * (1 - 1 / eff[name]) + H2t * 1 / eff[name]
        n2.set(P=P2, T=T_err(FLUIDS[0], 'H', H2, n2.x, P2, T2), G=n1.G)
        self.blocks[name].N = n2.G * (n1.H - n2.H)

    # Замыкание цикла (рекуператор + охладитель + компрессор)
    def loop(self, dT, dP, eff):
        n = self.nodes
        # Узел 6: после рекуператора (холодная сторона)
        n[6].set(P=n[6].P, T=n[6].T, G=n[6].G)
        self.heat_source('Heat', 6, 0)
        # Узел 4: после охладителя
        n[4].set(P=n[4].P, T=n[4].T, G=n[4].G)
        self.expander('Pump', 4, 5, eff)
        # Узел 3: после рекуператора (горячая сторона)
        n[3].H = n[2].H - (n[6].H - n[5].H)
        n[3].T = T_err(FLUIDS[0], 'H', n[3].H, n[3].x, n[3].P, n[5].T)
        n[3].set(P=n[3].P, T=n[3].T, G=n[3].G)
        self.heat_source('Cool', 3, 4)

        # T_pinch в рекуператоре
        H_rec, P_rec, T_rec = [np.vstack([np.linspace(n[2].__dict__[p], n[3].__dict__[p], N_pinch),np.linspace(n[6].__dict__[p], n[5].__dict__[p], N_pinch),])
            for p in ['H', 'P', 'T']
        ]
        T_hot  = T_err(FLUIDS[0], 'H', H_rec[0], n[1].x, P_rec[0], T_rec[0])
        T_cold = T_err(FLUIDS[0], 'H', H_rec[1], n[4].x, P_rec[1], T_rec[1])
        self.T_pinch = np.min(T_hot - T_cold)

        # Теплота рекуператора
        self.blocks['Exch'].Q = (n[2].G * (n[2].H - n[3].H)+ n[5].G * (n[5].H - n[6].H))

    # Полный цикл Брайтона
    def brayton(self, P_in, P_out, T_before, dT, dP, eff):
        if P_in <= P_out:
            return np.nan, np.nan
        try:
            n = self.nodes
            # 1. Задать состав, давления, температуры
            for node in n.values():
                node.fluid = FLUIDS[0]
                node.x = x
            n[0].P = n[1].P = P_in
            n[2].P = P_out
            n[3].P = P_out * (1 - dP)
            n[4].P = P_out * (1 - dP - dP_heat_source['Cool'])
            n[5].P = P_in  * (1 + dP + dP_heat_source['Heat'])
            n[6].P = P_in  * (1 + dP_heat_source['Heat'])
            n[0].T = T_max + 273.15
            n[1].T = T_after_pipe + 273.15
            n[4].T = T_min + 273.15
            n[6].T = T_before + 273.15
            # 2. Расход
            H_0 = PROPS[FLUIDS[0]]['H']((x, n[0].P, n[0].T))
            H_6 = PROPS[FLUIDS[0]]['H']((x, n[6].P, n[6].T))
            G = Q / (H_0 - H_6)
            for node in n.values():
                node.G = G
            # 3. Узел 0
            n[0].set(T=n[0].T, P=n[0].P, G=n[0].G)
            # 4. Паропровод
            self.heat_source('Pipe', 0, 1)
            # 5. Турбина
            self.expander('Turb', 1, 2, eff)
            # 6. Замыкание
            self.loop(dT, dP, eff)
            # 7. Отбраковка
            if self.T_pinch < dT or self.blocks['Exch'].Q < -1e-2:
                return np.nan, np.nan
        except Exception:
            return np.nan, np.nan

        # 8. Вспомогательная мощность водяного насоса
        P_cool = P_water * (1 - 0.5 * dP_water)
        T_cool = n[4].T + dT_water * 0.5
        self.blocks['Cool'].Aux = (self.blocks['Cool'].Q* dP_water * P_water/ (PROPS[FLUIDS[1]]['Cp']((1, P_cool, T_cool))* PROPS[FLUIDS[1]]['D']((1, P_cool, T_cool))
               * dT_water * 0.8 * 0.95))

        # 9. КПД
        KPD_brutto = ((self.blocks['Turb'].N + self.blocks['Pump'].N) * 0.98/ self.blocks['Heat'].Q)
        KPD_netto = KPD_brutto + self.blocks['Cool'].Aux / self.blocks['Heat'].Q

        balance = (np.nansum([b.Q for b in self.blocks.values()])- np.nansum([b.N for b in self.blocks.values()]))
        print(f'KPD: {KPD_netto:.3}; Balance: {balance:.2e}')
        return KPD_netto, self.T_pinch

# Обёртка для multiprocessing
def run_brayton(P_in, P_out, T_before, dT, dP, eff):
    cycle = Cycle()
    return cycle.brayton(P_in, P_out, T_before, dT, dP, eff)

# Одиночный тестовый расчёт
if __name__ == '__main__':
    cycle = Cycle()
    KPD_netto, T_pinch = cycle.brayton(P_in_line[18], P_out_line[16], T_before_line[110], dT_line[0], dP_fix, eff_fix)

    # =============================================================
    # Выгрузка узлов в Excel
    # =============================================================
    os.makedirs('res', exist_ok=True)

    rows = []
    for idx, node in cycle.nodes.items():
        rows.append({
            'Узел':          idx,
            'Fluid':         node.fluid,
            'T, К':          node.T,
            'T, °C':         node.T - 273.15,
            'P, Па':         node.P,
            'P, бар':        node.P / 1e5,
            'x':             node.x,
            'G, кг/с':       node.G,
            'D, кг/м³':      node.D,
            'H, Дж/кг':      node.H,
            'S, Дж/(кг·К)':  node.S,
            'Cp, Дж/(кг·К)': node.Cp,
            'Vis, мкПа·с':   node.Vis,
            'Tcx, Вт/(м·К)': node.Tcx,
        })
    df_nodes = pd.DataFrame(rows)
    df_nodes.to_excel('res/nodes.xlsx', index=False)

    # Блоки
    rows = []
    for name, block in cycle.blocks.items():
        rows.append({
            'Элемент': name,
            'Q, Вт':   block.Q,
            'Q, кВт':  block.Q / 1e3 if pd.notna(block.Q) else np.nan,
            'N, Вт':   block.N,
            'N, кВт':  block.N / 1e3 if pd.notna(block.N) else np.nan,
            'Aux, Вт': block.Aux,
            'Aux, кВт': block.Aux / 1e3 if pd.notna(block.Aux) else np.nan,
        })
    df_blocks = pd.DataFrame(rows)
    df_blocks.to_excel('res/blocks.xlsx', index=False)

    # Сводка
    df_summary = pd.DataFrame([
        {'Параметр': 'KPD_netto',  'Значение': KPD_netto,                'Ед.': '—'},
        {'Параметр': 'T_pinch',    'Значение': T_pinch,                  'Ед.': 'К'},
        {'Параметр': 'P_in',       'Значение': P_in_line[18],            'Ед.': 'Па'},
        {'Параметр': 'P_out',      'Значение': P_out_line[16],           'Ед.': 'Па'},
        {'Параметр': 'T_before',   'Значение': T_before_line[110],       'Ед.': '°C'},
        {'Параметр': 'dT',         'Значение': dT_line[0],               'Ед.': 'К'},
        {'Параметр': 'Q_heat',     'Значение': cycle.blocks['Heat'].Q,   'Ед.': 'Вт'},
        {'Параметр': 'N_turb',     'Значение': cycle.blocks['Turb'].N,   'Ед.': 'Вт'},
        {'Параметр': 'N_pump',     'Значение': cycle.blocks['Pump'].N,   'Ед.': 'Вт'},
        {'Параметр': 'N_cool_aux', 'Значение': cycle.blocks['Cool'].Aux, 'Ед.': 'Вт'},
        {'Параметр': 'Q_exch',     'Значение': cycle.blocks['Exch'].Q,   'Ед.': 'Вт'},
        {'Параметр': 'Q_pipe',     'Значение': cycle.blocks['Pipe'].Q,   'Ед.': 'Вт'},
    ])
    df_summary.to_excel('res/summary.xlsx', index=False)

    # =============================================================
    # Выгрузка блоков в формате "P | T" (Реактор, Турбина, ...)
    # =============================================================
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, Border, Side, PatternFill

    # Формат: (название, idx_вход, idx_выход)
    blocks_def = [
        ('Реактор',              6, 0),
        ('Турбина',              1, 2),
        ('Рекуператор',          2, 3),
        ('Охладитель',           3, 4),
        ('Компрессор',           4, 5),
        ('Рекуператор (холод.)', 5, 6),
        ('Паропровод',           0, 1),
    ]

    wb = Workbook()
    ws = wb.active
    ws.title = 'Цикл'

    thin = Side(border_style='thin', color='000000')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    header_fill = PatternFill('solid', fgColor='D9D9D9')
    p_fill      = PatternFill('solid', fgColor='C6E0B4')
    t_fill      = PatternFill('solid', fgColor='BDD7EE')
    center      = Alignment(horizontal='center', vertical='center')

    ws.column_dimensions['A'].width = 3
    for col in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
        ws.column_dimensions[col].width = 12

    row = 2
    col_start = 2

    for name, i_in, i_out in blocks_def:
        # Заголовок
        ws.cell(row=row, column=col_start,     value='P').fill = header_fill
        ws.cell(row=row, column=col_start + 1, value='T').fill = header_fill
        ws.cell(row=row, column=col_start,     ).alignment = center
        ws.cell(row=row, column=col_start + 1, ).alignment = center
        ws.cell(row=row, column=col_start,     ).border = border
        ws.cell(row=row, column=col_start + 1, ).border = border

        # Вход
        P_in_val  = cycle.nodes[i_in].P  / 1e6
        T_in_val  = cycle.nodes[i_in].T  - 273.15
        ws.cell(row=row + 1, column=col_start,     value=round(P_in_val, 4)).fill = p_fill
        ws.cell(row=row + 1, column=col_start + 1, value=round(T_in_val, 2)).fill = t_fill
        ws.cell(row=row + 1, column=col_start,     ).alignment = center
        ws.cell(row=row + 1, column=col_start + 1, ).alignment = center
        ws.cell(row=row + 1, column=col_start,     ).border = border
        ws.cell(row=row + 1, column=col_start + 1, ).border = border

        # Выход
        P_out_val = cycle.nodes[i_out].P / 1e6
        T_out_val = cycle.nodes[i_out].T - 273.15
        ws.cell(row=row + 2, column=col_start,     value=round(P_out_val, 4)).fill = p_fill
        ws.cell(row=row + 2, column=col_start + 1, value=round(T_out_val, 2)).fill = t_fill
        ws.cell(row=row + 2, column=col_start,     ).alignment = center
        ws.cell(row=row + 2, column=col_start + 1, ).alignment = center
        ws.cell(row=row + 2, column=col_start,     ).border = border
        ws.cell(row=row + 2, column=col_start + 1, ).border = border

        # Подпись
        ws.cell(row=row + 3, column=col_start, value=name).font = Font(bold=True)
        ws.cell(row=row + 3, column=col_start).alignment = center

        row += 6

    wb.save('res/cycle_blocks.xlsx')

    print('\nСохранено:')
    print('  res/nodes.xlsx')
    print('  res/blocks.xlsx')
    print('  res/summary.xlsx')
    print('  res/cycle_blocks.xlsx')
    print('\n=== Узлы ===')
    print(df_nodes.to_string(index=False))
    print('\n=== Блоки ===')
    print(df_blocks.to_string(index=False))