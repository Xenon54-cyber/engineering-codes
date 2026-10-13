import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import cm

# 1. Загрузка интерполяторов
params = ['D', 'H', 'S', 'Cp', 'Vis', 'Tcx']
labels = {
    'D':   ('Плотность',                'кг/м³'),
    'H':   ('Удельная энтальпия',       'Дж/кг'),
    'S':   ('Удельная энтропия',        'Дж/(кг·К)'),
    'Cp':  ('Удельная теплоёмкость',    'Дж/(кг·К)'),
    'Vis': ('Динамическая вязкость',    'мкПа·с'),
    'Tcx': ('Теплопроводность',         'Вт/(м·К)'),}

props = dict(zip(params, np.load('props.npy', allow_pickle=True)))

# 2. Сетки для таблиц и графиков
x_line = np.array([0.28, 0.29])
P_line = np.array([1e5, 3e6, 5e6])
T_line = np.arange(300, 1401, 100, dtype=float)

os.makedirs('res', exist_ok=True)
os.makedirs('res/tables', exist_ok=True)
os.makedirs('res/plots', exist_ok=True)

# 3. Таблицы: свойства от T при фиксированных (x, P)
for x in x_line:
    for P in P_line:
        rows = []
        for T in T_line:
            row = {'x': x, 'P_Pa': P, 'T_K': T}
            for p in params:
                row[p] = float(props[p]((x, P, T)))
            rows.append(row)
        df = pd.DataFrame(rows)
        fname = f'res/tables/props_x{x:.2f}_P{P:.0e}.xlsx'
        df.to_excel(fname, index=False)

# 4. Сводная таблица: все свойства во всех точках
rows = []
for x in x_line:
    for P in P_line:
        for T in T_line:
            row = {'x': x, 'P_Pa': P, 'T_K': T}
            for p in params:
                row[p] = float(props[p]((x, P, T)))
            rows.append(row)
pd.DataFrame(rows).to_excel('res/tables/all_props.xlsx', index=False)

# 5. Графики: свойство от T при фиксированных P
for p in params:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for x in x_line:
        for P in P_line:
            vals = np.array([float(props[p]((x, P, T))) for T in T_line])
            ax.plot(T_line, vals, marker='o', label=f'x={x:.2f}, P={P/1e6:.1f} МПа')
    ax.set_xlabel('Температура, К')
    ax.set_ylabel(f'{labels[p][0]}, {labels[p][1]}')
    ax.set_title(f'{labels[p][0]} от температуры')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(f'res/plots/{p}_vs_T.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Графики: свойство от P при фиксированных T
P_plot = np.linspace(1e5, 6e6, 60)
for p in params:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for x in x_line:
        for T in [300, 700, 1100, 1400]:
            vals = np.array([float(props[p]((x, P, T))) for P in P_plot])
            ax.plot(P_plot / 1e6, vals, marker='', label=f'x={x:.2f}, T={T} К')
    ax.set_xlabel('Давление, МПа')
    ax.set_ylabel(f'{labels[p][0]}, {labels[p][1]}')
    ax.set_title(f'{labels[p][0]} от давления')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(f'res/plots/{p}_vs_P.png', dpi=300, bbox_inches='tight')
    plt.close()

# 7. Графики: свойство от x при фиксированных (P, T)
x_plot = np.linspace(0.28, 0.29, 30)
for p in params:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for P in P_line:
        for T in [300, 700, 1100, 1400]:
            vals = np.array([float(props[p]((x, P, T))) for x in x_plot])
            ax.plot(x_plot, vals, marker='', label=f'P={P/1e6:.1f} МПа, T={T} К')
    ax.set_xlabel('Мольная доля Xe, x')
    ax.set_ylabel(f'{labels[p][0]}, {labels[p][1]}')
    ax.set_title(f'{labels[p][0]} от состава')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(f'res/plots/{p}_vs_x.png', dpi=300, bbox_inches='tight')
    plt.close()

# --- 8. Карты (contour) по (P, T) для каждого свойства ---
P_grid = np.linspace(1e5, 6e6, 60)
T_grid = np.linspace(250, 1400, 80)
X, Y = np.meshgrid(P_grid / 1e6, T_grid)

for p in params:
    for x in x_line:
        Z = np.zeros_like(X)
        for i, T in enumerate(T_grid):
            for j, P in enumerate(P_grid):
                Z[i, j] = float(props[p]((x, P, T)))
        fig, ax = plt.subplots(figsize=(9, 6))
        cf = ax.contourf(X, Y, Z, levels=30, cmap='turbo')
        cs = ax.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.3g')
        ax.set_xlabel('Давление, МПа')
        ax.set_ylabel('Температура, К')
        ax.set_title(f'{labels[p][0]} (x = {x:.2f})')
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(f'{labels[p][0]}, {labels[p][1]}')
        plt.tight_layout()
        plt.savefig(f'res/plots/map_{p}_x{x:.2f}.png', dpi=300, bbox_inches='tight')
        plt.close()

print('Готово. Таблицы: res/tables/. Графики: res/plots/.')