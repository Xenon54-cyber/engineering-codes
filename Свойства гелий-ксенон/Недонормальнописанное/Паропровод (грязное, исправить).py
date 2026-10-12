import numpy as np
import pandas as pd
from scipy.optimize import newton, bisect
from matplotlib import pyplot as plt
from multiprocessing import Pool
from time import time
import os
from scipy.interpolate import griddata

# Рабочие тела: смесь He-Xe и вода
fluids = np.array(['He-Xe', 'Water'])

# Основные параметры цикла:
# Q - подводимая тепловая мощность, Вт
# x - мольная доля ксенона в смеси He-Xe
# T_max - температура после нагревателя, °C
# T_after_pipe - температура после паропровода, °C
# T_min - температура перед охлаждением (после рекуператора), °C
# N_pinch - число точек по рекуператору для расчёта T_pinch
# dP_heat_source - относительные потери давления в нагревателе и охладителе
Q, x, T_max, T_after_pipe, T_min, N_pinch, dP_heat_source = 2.75e6, 0.283, 1060, 1030, 35, 20, dict(Heat=0.0364e-2, Cool=0.497e-2)

# Параметры водяного контура охлаждения:
# dP_water - относительные потери давления в водяном тракте
# P_water  - давление воды, Па
# dT_water - нагрев воды в охладителе, °C
dP_water, P_water, dT_water = 5e-2, 6e5, 10

# Фиксированные потери давления в рекуператоре и КПД турбины/компрессора
dP_fix, eff_fix = 4.553e-2, dict(Turb=92.67e-2, Pump=87.06e-2)

# Сетки для перебора:
# P_in_line     - давление перед турбиной (после нагревателя), Па
# P_out_line    - давление после турбины, Па
# T_before_line - температура перед турбиной, °C
# dT_line       - температурный напор (отсечка) рекуператора, °C
P_in_line = np.arange(30, 51, 1, dtype=np.float64) * 1e5
P_out_line = np.arange(10, 31, 1, dtype=np.float64) * 1e5
T_before_line = np.arange(650, 851, 1, dtype=np.float64)
dT_line = np.arange(5, 16, 1, dtype=np.float64)

# Сетки для анализа чувствительности:
# dP_line  - потери давления в рекуператоре
# eff_line - КПД турбины и компрессора
dP_line = np.arange(3, 7, 1, dtype=np.float64) * 1e-2
eff_line = dict(Turb=np.arange(85, 93, 1, dtype=np.float64) / 1e2, Pump=np.arange(75, 91, 1, dtype=np.float64) / 1e2)

# Полный декартовый перебор всех комбинаций параметров
data_stack = [(P_in, P_out, T_before, dT, dP_fix, eff_fix) for P_in in P_in_line for P_out in P_out_line for T_before in T_before_line for dT in dT_line]

# Список рассчитываемых свойств рабочего тела
params = ['D', 'H', 'S', 'Cp', 'Vis', 'Tcx']

# Таблица узлов цикла: 7 узлов (0..6), для каждого - рабочее тело, T, P, x, G и свойства
nodes = pd.DataFrame(index=np.arange(7), columns=['Fluid', 'T', 'P', 'x', 'G'] + params, dtype=np.float64)
nodes['Fluid'] = pd.Series(nodes['Fluid'], dtype=np.str_)

# Таблица элементов цикла: нагреватель, турбина, охладитель, компрессор, рекуператор, паропровод
# Для каждого элемента - теплота Q, мощность N, вспомогательная величина Aux
blocks = pd.DataFrame(index=['Heat', 'Turb', 'Cool', 'Pump', 'Exch', 'Pipe'], columns=['Q', 'N', 'Aux'], dtype=np.float64)

# Загрузка интерполяторов свойств: для He-Xe из props.npy, для воды из props_water.npy
props = {fluids[0]: dict(zip(params, np.load('props.npy', allow_pickle=True))), fluids[1]: dict(zip(params, np.load('props_water.npy', allow_pickle=True)))}

'''
props = dict(zip(params, np.load('props.npy', allow_pickle = True)))
props_water = dict(zip(params, np.load('props_water.npy', allow_pickle = True)))
T_min_props, T_max_props = min(props['H'].grid[2]), max(props['H'].grid[2])
'''


# Решение обратной задачи: найти T по заданному свойству (H или S) при известных x, P
def T_err(**kwargs):
    params_in = list(kwargs.keys())
    param = [param for param in params_in if param not in ['Fluid', 'x', 'P', 'T']][0]

    err = lambda T: kwargs[param] - props[kwargs['Fluid']][param]((kwargs['x'], kwargs['P'], T))
    T = newton(err, kwargs['T'], tol=1e-6)
    return T

# Заполнение узла: записываем переданные значения и интерполируем все свойства
def Node(node, **kwargs):
    nodes.loc[node, kwargs.keys()] = list(kwargs.values())
    nodes.loc[node, params] = [props[nodes.loc[node, 'Fluid']][param]((nodes.loc[node, 'x'], kwargs['P'], kwargs['T'])) for param in params]
    pass

# Турбина или компрессор: расчёт адиабатного процесса с учётом КПД
def Expander(name, node1, node2, eff):
    P2 = nodes.loc[node2, 'P']
    S1 = nodes.loc[node1, 'S']
    # Оценка T2 по адиабате идеального газа (показатель 0.4)
    T2 = nodes.loc[node1, 'T'] * (P2 / nodes.loc[node1, 'P']) ** 0.4
    # Энтальпия в идеальном (изоэнтропном) процессе
    H2t = props[fluids[0]]['H']((nodes.loc[node2, 'x'], P2, T_err(S=S1, Fluid=fluids[0], x=nodes.loc[node2, 'x'], P=P2, T=T2)))
    # Реальная энтальпия: для турбины и компрессора формулы разные
    H2 = nodes.loc[node1, 'H'] * (1 - eff[name]) + H2t * eff[name] if name == 'Turb' else nodes.loc[node1, 'H'] * (1 - 1 / eff[name]) + H2t * 1 / eff[name]

    # Заполняем узел 2: давление P2, температура по H2, расход тот же
    Node(node2, P=P2, T=T_err(H=H2, Fluid=fluids[0], x=nodes.loc[node2, 'x'], P=P2, T=T2), G=nodes.loc[node1, 'G'])

    # Внутренняя мощность: N = G * (H1 - H2)
    blocks.loc[name, 'N'] = nodes.loc[node2, 'G'] * (nodes.loc[node1, 'H'] - nodes.loc[node2, 'H'])
    pass

# Источник/сток теплоты: нагреватель или охладитель
def Heat_source(name, node1, node2):
    # Узел 2 имеет те же P, T, G (T и P заданы заранее)
    Node(node2, P=nodes.loc[node2, 'P'], T=nodes.loc[node2, 'T'], G=nodes.loc[node1, 'G'])
    # Теплота Q = G * (H2 - H1)
    blocks.loc[name, 'Q'] = nodes.loc[node2, 'G'] * (nodes.loc[node2, 'H'] - nodes.loc[node1, 'H'])
    pass

# Замыкание цикла: рекуператор, охладитель, компрессор, расчёт T_pinch
def Loop(dT, dP, eff):
    # Узел 6 - после рекуператора (холодная сторона), перед нагревателем
    Node(6, P=nodes.loc[6, 'P'], T=nodes.loc[6, 'T'], G=nodes.loc[6, 'G'])
    # 6 -> 0: нагрев в нагревателе
    Heat_source('Heat', 6, 0)
    # Узел 4 - после охладителя, перед компрессором
    Node(4, P=nodes.loc[4, 'P'], T=nodes.loc[4, 'T'], G=nodes.loc[4, 'G'])
    # 4 -> 5: сжатие в компрессоре
    Expander('Pump', 4, 5, eff)

    # Узел 3 - после рекуператора (горячая сторона)
    # Энтальпия: H3 = H2 - (H6 - H5) - баланс рекуператора
    nodes.loc[3, 'H'] = nodes.loc[2, 'H'] - (nodes.loc[6, 'H'] - nodes.loc[5, 'H'])
    # Температура по H3
    nodes.loc[3, 'T'] = T_err(H=nodes.loc[3, 'H'], Fluid=fluids[0], x=nodes.loc[3, 'x'], P=nodes.loc[3, 'P'], T=nodes.loc[5, 'T'])
    Node(3, P=nodes.loc[3, 'P'], T=nodes.loc[3, 'T'], G=nodes.loc[3, 'G'])
    # 3 -> 4: охлаждение
    Heat_source('Cool', 3, 4)

    # Расчёт T_pinch в рекуператоре: минимум разности температур горячего и холодного потоков
    global T_pinch
    H_rec, P_rec, T_rec = [np.vstack([np.linspace(nodes.loc[2, param], nodes.loc[3, param], N_pinch), np.linspace(nodes.loc[6, param], nodes.loc[5, param], N_pinch)]) for param in ['H', 'P', 'T']]
    T_pinch = np.min(T_err(H=H_rec[0], Fluid=fluids[0], x=nodes.loc[1, 'x'], P=P_rec[0], T=T_rec[0]) - T_err(H=H_rec[1], Fluid=fluids[0], x=nodes.loc[4, 'x'], P=P_rec[1], T=T_rec[1]))

    # Тепловая мощность рекуператора: сумма изменений энтальпий обоих потоков
    blocks.loc['Exch', 'Q'] = nodes.loc[2, 'G'] * (nodes.loc[2, 'H'] - nodes.loc[3, 'H']) + nodes.loc[5, 'G'] * (nodes.loc[5, 'H'] - nodes.loc[6, 'H'])
    pass


# Расчёт одного цикла Брайтона при заданных параметрах
def Brayton(P_in, P_out, T_before, dT, dP, eff):
    if P_in > P_out:
        try:
            # Рабочее тело и состав одинаковы для всех узлов
            nodes.loc[:, 'Fluid'] = fluids[0]
            nodes.loc[:, 'x'] = x
            # Давления в узлах с учётом потерь в рекуператоре и теплообменниках
            nodes.loc[:, 'P'] = [P_in] * 2 + [P_out] + [P_out * (1 - dP)] + [P_out * (1 - dP - dP_heat_source['Cool'])] + [P_in * (1 + dP + dP_heat_source['Heat'])] + [P_in * (1 + dP_heat_source['Heat'])]
            # Температуры в узлах 0, 1, 4, 6 (в К)
            nodes.loc[[0, 1, 4, 6], 'T'] = np.array([T_max, T_after_pipe, T_min, T_before]) + 273.15
            # Расход из условия подводимой теплоты Q
            nodes.loc[:, 'G'] = Q / (props[fluids[0]]['H']((nodes.loc[0, 'x'], nodes.loc[0, 'P'], nodes.loc[0, 'T'])) - props[fluids[0]]['H']((nodes.loc[6, 'x'], nodes.loc[6, 'P'], nodes.loc[6, 'T'])))

            # Узел 0 - после нагревателя
            Node(0, T=nodes.loc[0, 'T'], P=nodes.loc[0, 'P'], G=nodes.loc[0, 'G'])
            # 0 -> 1: потери в паропроводе
            Heat_source('Pipe', 0, 1)
            # 1 -> 2: расширение в турбине
            Expander('Turb', 1, 2, eff)
            # Замыкание цикла
            Loop(dT, dP, eff)
            # Отбраковка режима: слишком маленький T_pinch или отрицательная теплота рекуператора
            if T_pinch < dT or blocks.loc['Exch', 'Q'] < -1e-2: return np.nan, np.nan
        except:
            return np.nan, np.nan

        # Параметры воды в охладителе (средние)
        P_cool, T_cool = P_water * (1 - 0.5 * dP_water), nodes.loc[4, 'T'] + dT_water * 0.5
        # Мощность водяного насоса (вспомогательная)
        blocks.loc['Cool', 'Aux'] = blocks.loc['Cool', 'Q'] * dP_water * P_water / (props[fluids[1]]['Cp']((1, P_cool, T_cool)) * props[fluids[1]]['D']((1, P_cool, T_cool)) * dT_water * 0.8 * 0.95)

        # КПД брутто (без учёта собственных нужд)
        KPD_brutto = (blocks.loc['Turb', 'N'] + blocks.loc['Pump', 'N']) * 0.98 / blocks.loc['Heat', 'Q']
        # КПД нетто (с учётом водяного насоса)
        KPD_netto = KPD_brutto + blocks.loc['Cool', 'Aux'] / blocks.loc['Heat', 'Q']
        # Проверка баланса
        balance = np.sum(blocks.loc[:, 'Q']) - np.sum(blocks.loc[:, 'N'])
        print(f'KPD: {KPD_netto:.3}; Balance: {balance:.2}\n')

        # Здесь и ниже нужно выводить T_pinch, мощности турбины/компрессора, КПД нетто

        return KPD_netto, T_pinch

    else:
        return np.nan, np.nan


# Тестовый одиночный расчёт
Brayton(P_in_line[18], P_out_line[16], T_before_line[110], dT_line[0], dP_fix, eff_fix)

if __name__ == '__main__':
    start = time()
    dir_out = os.path.join(os.getcwd(), 'res')
    dir_out_data, dir_out_img = os.path.join(dir_out, 'data'), os.path.join(dir_out, 'img')

    # И по data македирс

    os.makedirs(dir_out_img, exist_ok=True)

    with Pool(14) as p:
        # Сделать таблички по выходу

        KPD, aux_hexe, aux_water, aux_sum = zip(*p.starmap(Heating, data_stack))
    pd.DataFrame(np.column_stack([data_stack, KPD, aux_hexe, aux_water, aux_sum]), columns=['dP', 'dT', 'KPD', 'aux_hexe', 'aux_water', 'aux_sum']).to_excel(os.path.join(dir_out, 'full_stack.xlsx'), index=False)

    # И графики по всему (болванки ниже, но они очень грязные)

'''
def plot_finite_line(ax, x, y, **kwargs):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    # Сортировка для ровной линии графика при отображении от T_pinch
    sort_idx = np.argsort(x[mask])
    ax.plot(x[mask][sort_idx], y[mask][sort_idx], **kwargs)
    return mask


if __name__ == '__main__':
    start = time()

    # Параллельный расчёт всех точек
    with Pool(14) as p:
        results = p.starmap(Brayton, data_stack)

    # Разворачиваем результаты в 4-мерные массивы
    KPD = np.array([r[0] for r in results]).reshape(len(P_in_line), len(P_out_line), len(T_before_line), len(dT_line))
    T_pinch_arr = np.array([r[1] for r in results]).reshape(len(P_in_line), len(P_out_line), len(T_before_line), len(dT_line))

    os.makedirs('res', exist_ok = True)

    # Формируем и сохраняем полную таблицу
    data_rows = []
    for P_in_num, P_in in enumerate(P_in_line):
        for P_out_num, P_out in enumerate(P_out_line):
            for T_before_num, T_before in enumerate(T_before_line):
                for dT_num, dT in enumerate(dT_line):
                    data_rows.append([
                        P_in, P_out, T_before, dT, 
                        T_pinch_arr[P_in_num, P_out_num, T_before_num, dT_num], 
                        KPD[P_in_num, P_out_num, T_before_num, dT_num]
                    ])

    pd.DataFrame(data_rows, columns = ['P_in', 'P_out', 'T_before', 'dT_cutoff', 'T_pinch_actual', 'KPD']).to_excel('res/full_stack.xlsx', index = False)

    # Находим оптимум по КПД
    opt = np.argwhere(KPD == np.nanmax(KPD))[0]
    dir_opt = f'res/optimum/P_in = {P_in_line[opt[0]]:.1e}, P_out = {P_out_line[opt[1]]:.1e}, T_before = {T_before_line[opt[2]]}, dT = {dT_line[opt[3]]}'
    os.makedirs(dir_opt, exist_ok = True)

    # Все комбинации КПД турбины и компрессора
    eff_line_stack = [dict(Turb = turb, Pump = pump) for turb in eff_line['Turb'] for pump in eff_line['Pump']]

    # Извлечение фактического T_pinch для максимальных КПД на каждом шаге отсечения
    KPD_dT = []
    T_pinch_dT = []
    for dT_num, dT in enumerate(dT_line):
        slice_KPD = KPD[:, :, :, dT_num]
        if not np.isnan(slice_KPD).all():
            idx = np.unravel_index(np.nanargmax(slice_KPD), slice_KPD.shape)
            KPD_dT.append(slice_KPD[idx])
            T_pinch_dT.append(T_pinch_arr[idx[0], idx[1], idx[2], dT_num])
        else:
            KPD_dT.append(np.nan)
            T_pinch_dT.append(np.nan)

    KPD_dT = np.array(KPD_dT)
    T_pinch_dT = np.array(T_pinch_dT)

    # Сохраняем зависимость КПД от dT
    pd.DataFrame({'dT_cutoff': dT_line, 'T_pinch_actual': T_pinch_dT, 'KPD': KPD_dT}).to_excel('/'.join([dir_opt, 'dT.xlsx']), index = False)

    # Графики
    dir_opt_img = '/'.join([dir_opt, 'img'])
    os.makedirs(dir_opt_img, exist_ok=True)

    P_in_vals = P_in_line / 1e6
    P_out_vals = P_out_line / 1e6
    # Срез КПД по оптимальному dT
    KPD_slice = np.nanmax(KPD[:, :, :, opt[3]], axis = 2)

    # Функция построения градиентной карты по давлениям
    def pressure_map(values, filename, title, cbar_label, vmin=None):
        values = np.asarray(values, dtype=float)
        x_data, y_data, z_data = [], [], []
        for i, P1 in enumerate(P_in_vals):
            for j, P2 in enumerate(P_out_vals):
                value = values[i, j]
                if np.isfinite(value) and (vmin is None or value >= vmin):
                    x_data.append(P2); y_data.append(P1); z_data.append(value)
        if not z_data:
            return
        x_data, y_data, z_data = map(np.asarray, (x_data, y_data, z_data))
        xi = np.linspace(x_data.min(), x_data.max(), 300)
        yi = np.linspace(y_data.min(), y_data.max(), 300)
        X, Y = np.meshgrid(xi, yi)
        Z = griddata((x_data, y_data), z_data, (X, Y), method='linear')
        if vmin is not None:
            Z[Z < vmin] = np.nan
        fig, ax = plt.subplots(figsize=(9, 6.5))
        zmin = vmin if vmin is not None else np.nanmin(Z)
        zmax = np.nanmax(Z)
        levels = np.linspace(zmin, zmax, 25) if zmax > zmin else 2
        cf = ax.contourf(X, Y, Z, levels=levels)

        # Рисуем контуры
        cs = ax.contour(X, Y, Z, levels=10, linewidths=0.5, colors = 'black')
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.3f', colors = 'black')

        # Точка максимума (zorder=10 выносит ее на самый передний план)
        imax = np.argmax(z_data)
        ax.scatter(x_data[imax], y_data[imax], s=90, marker='*', edgecolors='black', linewidths=0.8, zorder=10)

        if cbar_label == 'КПД нетто':
            text = f'max: {z_data[imax]*100:.2f}%'
            text += f'\n$P_1$={y_data[imax]:.2f} МПа, $P_2$={x_data[imax]:.2f} МПа'

            # Смещаем текст влево (-10) и вниз (-15), и выравниваем по правому верхнему краю плашки
            ax.annotate(text, (x_data[imax], y_data[imax]), 
                        xytext=(-10, -15), 
                        textcoords='offset points', 
                        fontsize=9,
                        ha='right', # Горизонтальное выравнивание
                        va='top',   # Вертикальное выравнивание
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, boxstyle='square,pad=0.35'),
                        zorder=10)   

        ax.set_xlabel(r'$P_2$, МПа'); ax.set_ylabel(r'$P_1$, МПа'); ax.set_title(title)
        ax.grid(True, alpha=0.2, zorder=1)
        cbar = fig.colorbar(cf, ax=ax); cbar.set_label(cbar_label)
        plt.tight_layout(); plt.savefig('/'.join([dir_opt_img, filename]), dpi=300, bbox_inches='tight'); plt.close(fig)

    # Карта КПД нетто от давлений
    pressure_map(KPD_slice, 'KPD_map.png', 'Градиентная карта КПД нетто от давлений', 'КПД нетто', vmin=0.4)

    # Формируем таблицу мощностей по сетке давлений при оптимальных остальных параметрах
    power_rows = []
    for P1 in P_in_line:
        for P2 in P_out_line:
            result = Brayton(P1, P2, T_before_line[opt[2]], dT_line[opt[3]], dP_fix, eff_fix)[0]
            if np.isfinite(result):
                turb = abs(blocks.loc['Turb', 'N']); comp = abs(blocks.loc['Pump', 'N'])
                P_cool = P_water * (1 - 0.5 * dP_water)
                T_cool = nodes.loc[4, 'T'] + dT_water * 0.5
                N_cool = blocks.loc['Cool', 'Q'] * dP_water * P_water / (props_water['Cp']((P_cool, T_cool)) * props_water['D']((P_cool, T_cool)) * dT_water * 0.8 * 0.95)
                power_rows.append([P1/1e6, P2/1e6, turb, comp, abs(N_cool), result])
            else:
                power_rows.append([P1/1e6, P2/1e6, np.nan, np.nan, np.nan, np.nan])

    power_df = pd.DataFrame(power_rows, columns=['P_in','P_out','N_turb','N_comp','N_cool','KPD_netto'])
    power_df.to_excel('/'.join([dir_opt, 'pressure_outputs.xlsx']), index=False)
    Nt = power_df.pivot(index='P_in', columns='P_out', values='N_turb').reindex(index=P_in_vals, columns=P_out_vals).to_numpy() / 1e6
    Nc = power_df.pivot(index='P_in', columns='P_out', values='N_comp').reindex(index=P_in_vals, columns=P_out_vals).to_numpy() / 1e6
    Ncool = power_df.pivot(index='P_in', columns='P_out', values='N_cool').reindex(index=P_in_vals, columns=P_out_vals).to_numpy() / 1e3

    # Карты мощностей
    pressure_map(Nt, 'N_turb.png', 'Внутренняя мощность турбины', '$N_т$, МВт')
    pressure_map(Nc, 'N_comp.png', 'Внутренняя мощность компрессора', '$N_к$, МВт')
    pressure_map(Ncool, 'N_cool.png', 'Потребляемая мощность привода водяного насоса', '$N_в$, кВт')

    # График зависимости КПД от температуры перед турбиной
    plt.figure(figsize=(8,5.5)); plot_finite_line(plt.gca(), T_before_line, KPD[opt[0],opt[1],:,opt[3]]*100, marker='o')
    plt.xlabel('Температура перед турбиной, °C'); plt.ylabel('КПД нетто, %'); plt.title('Зависимость КПД нетто от температуры перед турбиной')
    plt.grid(True, alpha=0.2); plt.tight_layout(); plt.savefig('/'.join([dir_opt_img,'T_before.png']), dpi=300, bbox_inches='tight'); plt.close()

    # Измененный график №6: теперь строится по T_pinch_actual
    plt.figure(figsize=(8,5.5)); plot_finite_line(plt.gca(), T_pinch_dT, KPD_dT*100, marker='o')
    plt.xlabel('Температурный напор рекуператора, °C'); plt.ylabel('КПД нетто, %'); plt.title('Зависимость КПД нетто от фактического $T_{pinch}$')
    plt.grid(True, alpha=0.2); plt.tight_layout(); plt.savefig('/'.join([dir_opt_img,'T_pinch.png']), dpi=300, bbox_inches='tight'); plt.close()

    # Чувствительность к потерям давления в рекуператоре
    KPD_dP = np.array([Brayton(P_in_line[opt[0]],P_out_line[opt[1]],T_before_line[opt[2]],dT_line[opt[3]],dP,eff_fix)[0] for dP in dP_line])
    pd.DataFrame({'dP':dP_line,'dP_percent':dP_line*100,'KPD_netto':KPD_dP}).to_excel('/'.join([dir_opt,'dP.xlsx']), index=False)
    plt.figure(figsize=(8,5.5)); plot_finite_line(plt.gca(), dP_line*100, KPD_dP*100, marker='o')
    plt.xlabel('Потери давления в рекуператоре, %'); plt.ylabel('КПД нетто, %'); plt.title('Чувствительность КПД нетто к потерям давления рекуператора')
    plt.grid(True, alpha=0.2); plt.tight_layout(); plt.savefig('/'.join([dir_opt_img,'dP.png']), dpi=300, bbox_inches='tight'); plt.close()

    # Чувствительность к КПД турбины и компрессора
    KPD_eff=np.array([Brayton(P_in_line[opt[0]],P_out_line[opt[1]],T_before_line[opt[2]],dT_line[opt[3]],dP_fix,eff)[0] for eff in eff_line_stack]).reshape(len(eff_line['Turb']),len(eff_line['Pump']))
    eff_rows=[[turb,pump,KPD_eff[i,j]] for i,turb in enumerate(eff_line['Turb']) for j,pump in enumerate(eff_line['Pump'])]
    pd.DataFrame(eff_rows,columns=['eff_Turb','eff_Pump','KPD_netto']).to_excel('/'.join([dir_opt,'eff.xlsx']),index=False)
    fig,ax=plt.subplots(figsize=(8,6)); Z_eff=np.ma.masked_invalid(KPD_eff*100)
    cf=ax.contourf(eff_line['Pump']*100,eff_line['Turb']*100,Z_eff,levels=25); cs=ax.contour(eff_line['Pump']*100,eff_line['Turb']*100,Z_eff,levels=10,linewidths=0.5, colors='black'); ax.clabel(cs,inline=True,fontsize=8,fmt='%.1f', colors = 'black')
    ax.set_xlabel('КПД компрессора, %'); ax.set_ylabel('КПД турбины, %'); ax.set_title('Чувствительность КПД нетто к КПД турбины и компрессора'); cbar=fig.colorbar(cf,ax=ax); cbar.set_label('КПД нетто, %'); ax.grid(True,alpha=0.2)
    plt.tight_layout(); plt.savefig('/'.join([dir_opt_img,'eff.png']),dpi=300,bbox_inches='tight'); plt.close(fig)

    print('Время расчёта:', time() - start)
'''