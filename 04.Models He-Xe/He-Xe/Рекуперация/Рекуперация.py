import numpy as np
import pandas as pd
from scipy.optimize import newton, bisect, least_squares
from matplotlib import pyplot as plt
from multiprocessing import Pool
from time import time
import os
from scipy.interpolate import griddata

Q, x, T_max, T_min, N_pinch, dP_heat_source, dP_water, P_water, dT_water = 2.75e6, 0.283, 1060, 35, 20, dict(Heat = 0.0364e-2, Cool = 0.497e-2), 5e-2, 6e5, 10
dP_fix, eff_fix = 4.553e-2, dict(Turb = 92.67e-2, Pump = 87.06e-2)

P_in_line = np.arange(1, 5.1, 0.1, dtype = np.float64) * 1e6
P_out_line = np.arange(1, 3.1, 0.1, dtype = np.float64) * 1e6
T_before_line = np.arange(650, 900, 50, dtype = np.float64)
#dT_line = np.arange(5, 20, 5, dtype = np.float64)
dT_line = np.arange(5, 16, 1, dtype = np.float64)

dP_line = np.arange(0.03, 0.07, 0.01, dtype = np.float64)
eff_line = dict(Turb = np.arange(85, 100, 5, dtype = np.float64) / 1e2, Pump = np.arange(75, 95, 5, dtype = np.float64) / 1e2)
#eff_line = dict(Turb = np.arange(85, 96, 1, dtype = np.float64) / 1e2, Pump = np.arange(75, 91, 1, dtype = np.float64) / 1e2)

data_stack = [(P_in, P_out, T_before, dT, dP_fix, eff_fix) for P_in in P_in_line for P_out in P_out_line for T_before in T_before_line for dT in dT_line]

params = ['D', 'H', 'S', 'Cp', 'Vis', 'Tcx']
nodes = pd.DataFrame(index = np.arange(6) + 1, columns = ['T', 'P', 'x', 'G'] + params, dtype = np.float64)
blocks = pd.DataFrame(index = ['Heat', 'Turb', 'Cool', 'Pump', 'Exch'], columns = ['N', 'Q'], dtype = np.float64)

props = dict(zip(params, np.load('props.npy', allow_pickle = True)))
props_water = dict(zip(params, np.load('props_water.npy', allow_pickle = True)))
T_min_props, T_max_props = min(props['H'].grid[2]), max(props['H'].grid[2])

def T_err(**kwargs):
    params_in = list(kwargs.keys())
    param = [param for param in params_in if param not in ['x', 'P', 'T']][0]
    err = lambda T: kwargs[param] - props[param]((kwargs['x'], kwargs['P'], T))
    try:
        T = newton(err, kwargs['T'], tol = 1e-6)
    except:
        T_min = kwargs['T'] - 200
        T_max = kwargs['T'] + 200
        try:
            T = bisect(err, max(T_min, T_min_props), min(T_max, T_max_props), xtol = 1e-6)
        except:
            T = bisect(err, T_min_props, T_max_props, xtol = 1e-6)
    return T

def Node(node, **kwargs):
    nodes.loc[node, kwargs.keys()] = list(kwargs.values())
    for param in params:
        nodes.loc[node, param] = props[param]((nodes.loc[node, 'x'], kwargs['P'], kwargs['T']))
    pass

def Expander(name, node1, node2, eff):
    P2 = nodes.loc[node2, 'P']
    S1 = nodes.loc[node1, 'S']
    T2 = nodes.loc[node1, 'T'] * (P2 / nodes.loc[node1, 'P']) ** 0.4
    H2t = props['H']((nodes.loc[node2, 'x'], P2, T_err(S = S1, x = nodes.loc[node2, 'x'], P = P2, T = T2)))
    H2 = nodes.loc[node1, 'H'] * (1 - eff[name]) + H2t * eff[name] if name == 'Turb' else nodes.loc[node1, 'H'] * (1 - 1 / eff[name]) + H2t * 1 / eff[name]
    
    Node(node2, P = P2, T = T_err(H = H2, x = nodes.loc[node2, 'x'], P = P2, T = T2), G = nodes.loc[node1, 'G'])
    
    blocks.loc[name, 'N'] = nodes.loc[node2, 'G'] * (nodes.loc[node1, 'H'] - nodes.loc[node2, 'H'])
    pass

def Heat_source(name, node1, node2):
    if name == 'Heat': nodes.loc[node1, 'P'] = nodes.loc[node2, 'P'] * (1 + dP_heat_source[name])    
    elif name == 'Cool': nodes.loc[node2, 'P'] = nodes.loc[node1, 'P'] * (1 - dP_heat_source[name])
    
    Node(node2, P = nodes.loc[node2, 'P'], T = nodes.loc[node2, 'T'], G = nodes.loc[node1, 'G'])
    
    blocks.loc[name,'Q'] = nodes.loc[node2, 'G'] * (nodes.loc[node2, 'H'] - nodes.loc[node1, 'H'])
    pass

def Recuperation(node11, node12, node21, node22, dT, dP, eff):
    nodes.loc[[node12, node21], 'P'] = nodes.loc[node11, 'P'] * (1 - dP), nodes.loc[node22, 'P'] * (1 + dP)
    nodes.loc[node12 + 1, 'P'] = nodes.loc[node12, 'P'] * (1 - dP_heat_source['Cool'])
    
    Node(node22, P = nodes.loc[node22, 'P'], T = nodes.loc[node22, 'T'], G = nodes.loc[node22, 'G'])
    Heat_source('Heat', node22, 1)
    Node(node21 - 1, P = nodes.loc[node21 - 1, 'P'], T = nodes.loc[node21 - 1, 'T'], G = nodes.loc[node21 - 1, 'G'])
    Expander('Pump', node21 - 1, node21, eff)
    
    nodes.loc[node12, 'H'] = nodes.loc[node11, 'H'] - (nodes.loc[node22, 'H'] - nodes.loc[node21, 'H'])
    nodes.loc[node12, 'T'] = T_err(H = nodes.loc[node12, 'H'], x = nodes.loc[node12, 'x'], P = nodes.loc[node12, 'P'], T = nodes.loc[node12, 'T'])
    Node(node12, P = nodes.loc[node12, 'P'], T = nodes.loc[node12, 'T'], G = nodes.loc[node12, 'G'])
    
    Heat_source('Cool', node12, node12 + 1)
    
    global T_pinch
    H_rec, P_rec, T_rec = [np.vstack([np.linspace(nodes.loc[node11, param], nodes.loc[node12, param], N_pinch), np.linspace(nodes.loc[node22, param], nodes.loc[node21, param], N_pinch)]) for param in ['H', 'P', 'T']]
    T_rec = T_err(H = H_rec, x = nodes.loc[node11, 'x'], P = P_rec, T = T_rec)
    T_pinch = np.min(T_rec[0] - T_rec[1])
    
    blocks.loc['Exch', 'Q'] =  nodes.loc[node11, 'G'] * (nodes.loc[node11, 'H'] - nodes.loc[node12, 'H']) + nodes.loc[node21, 'G'] * (nodes.loc[node21, 'H'] - nodes.loc[node22, 'H'])
    pass

def Brayton(P_in, P_out, T_before, dT, dP, eff):
    if P_in > P_out:
        try:
            nodes.loc[:, 'x'] = x
            nodes.loc[[1, 2], 'P'] = P_in, P_out
            nodes.loc[6, 'P'] = P_in * (1 + dP_heat_source['Heat'])
            nodes.loc[[1, 4, 6], 'T'] = np.array([T_max, T_min, T_before]) + 273.15
            nodes.loc[:, 'G'] = Q / (props['H']((nodes.loc[1, 'x'], nodes.loc[1, 'P'], nodes.loc[1, 'T'])) - props['H']((nodes.loc[6, 'x'], nodes.loc[6, 'P'], nodes.loc[6, 'T'])))
            
            Node(1, T = nodes.loc[1, 'T'], P = nodes.loc[1, 'P'], G = nodes.loc[1, 'G'])
            Expander('Turb', 1, 2, eff)
            Recuperation(2, 3, 5, 6, dT, dP, eff)
            if T_pinch < dT or blocks.loc['Exch', 'Q'] < -1e-2: return np.nan
            #if np.min(T_pinch) < dT or nodes.loc[3, 'T'] > nodes.loc[2, 'T'] or nodes.loc[5, 'T'] > nodes.loc[6, 'T']: return np.nan
            
        except: return np.nan
        
        P_cool, T_cool = P_water * (1 - 0.5 * dP_water), nodes.loc[4, 'T'] + dT_water * 0.5
        N_cool = blocks.loc['Cool', 'Q'] * dP_water * P_water / (props_water['Cp']((P_cool, T_cool)) * props_water['D']((P_cool, T_cool)) * dT_water * 0.8 * 0.95)
        
        KPD_brutto = (blocks.loc['Turb', 'N'] + blocks.loc['Pump', 'N']) * 0.98 / blocks.loc['Heat', 'Q']
        KPD_netto = KPD_brutto + N_cool / blocks.loc['Heat', 'Q']
        balance = np.sum(blocks.loc[:, 'Q']) - np.sum(blocks.loc[:, 'N'])
        print(f'KPD: {KPD_brutto:.3}; Balance: {balance:.2}\n')
        
        #if KPD_brutto < 0.45: return np.nan
        
        return KPD_brutto#, KPD_netto
    
    else: return np.nan

#Brayton(3.6e6, 2.3e6, T_before_line[-1], dT_line[0], dP_fix, eff_fix)
#Brayton(1.1e6, 1e6, T_before_line[0], dT_line[0], dP_fix, eff_fix)
#nodes.to_excel('res.xlsx')

if __name__ == '__main__':
    start = time()
    
    with Pool(14) as p:
        KPD = p.starmap(Brayton, data_stack)
    KPD = np.array(KPD).reshape(len(P_in_line), len(P_out_line), len(T_before_line), len(dT_line))
    os.makedirs('res', exist_ok = True)
    pd.DataFrame(np.vstack([np.hstack([P_in, P_out, T_before, dT, KPD[P_in_num, P_out_num, T_before_num, dT_num]]) for P_in_num, P_in in enumerate(P_in_line) for P_out_num, P_out in enumerate(P_out_line) for T_before_num, T_before in enumerate(T_before_line) for dT_num, dT in enumerate(dT_line)]), columns = ['P_in', 'P_out', 'T_before', 'dT', 'KPD']).to_excel('res/full_stack.xlsx', index = False)
    
    opt = np.argwhere(KPD == np.nanmax(KPD))[0]
    dir_opt = f'res/optimum/P_in = {P_in_line[opt[0]]:.1e}, P_out = {P_out_line[opt[1]]:.1e}, T_before = {T_before_line[opt[2]]}, dT = {dT_line[opt[3]]}'
    os.makedirs(dir_opt, exist_ok = True)
    
    eff_line_stack = [dict(Turb = turb, Pump = pump) for turb in eff_line['Turb'] for pump in eff_line['Pump']]
    
    KPD_dT = np.array([np.nanmax(KPD[:, :, :, dT_num]) for dT_num, dT in enumerate(dT_line)])
    #KPD_dP = np.array([Brayton(*data_dP) for data_dP in [(P_in_line[opt[0]], P_out_line[opt[1]], T_before_line[opt[2]], dT_line[opt[3]], dP, eff_fix) for dP in dP_line]])
    #KPD_eff = np.array([Brayton(*data_eff) for data_eff in [(P_in_line[opt[0]], P_out_line[opt[1]], T_before_line[opt[2]], dT_line[opt[3]], dP_fix, eff) for eff in eff_line_stack]])
    
    pd.DataFrame(np.vstack([np.hstack([dT, KPD_dT[dT_num]]) for dT_num, dT in enumerate(dT_line)]), columns = ['dT', 'KPD']).to_excel('/'.join([dir_opt, 'dT.xlsx']), index = False)
    #pd.DataFrame(np.vstack([np.hstack([dP, KPD_dP[dP_num]]) for dP_num, dP in enumerate(dP_line)]), columns = ['dP', 'KPD']).to_excel('/'.join([dir_opt, 'dP.xlsx']), index = False)
    #pd.DataFrame(np.vstack([np.hstack([eff['Turb'], eff['Pump'], KPD_eff[eff_num]]) for eff_num, eff in enumerate(eff_line_stack)]), columns = ['eff_Turb', 'eff_Pump', 'KPD']).to_excel('/'.join([dir_opt, 'eff.xlsx']), index = False)
    
    #Графики
    '''
    dir_opt_img = '/'.join([dir_opt, 'img'])
    os.makedirs(dir_opt_img, exist_ok = True)
    
    c = plt.pcolormesh(P_out_line, P_in_line, KPD[:, :, opt[2], opt[3]], cmap = 'turbo')
    plt.xlabel('Давление на выходе из турбины $P_{out}$, Па')
    plt.ylabel('Давление на входе в турбину $P_{in}$, Па')
    P_in_arg_slice, P_out_arg_slice = np.where(np.any(~np.isnan(KPD[:, :, opt[2], opt[3]]), axis = 1))[0], np.where(np.any(~np.isnan(KPD[:, :, opt[2], opt[3]]), axis = 0))[0]
    plt.xlim(P_out_line[P_out_arg_slice[0]], P_out_line[P_out_arg_slice[-1]])
    plt.ylim(P_in_line[P_in_arg_slice[0]], P_in_line[P_in_arg_slice[-1]])
    plt.title(f'КПД ($T_{{before}}$ = {T_before_line[opt[2]]} °C, $dT_{{min}}$ = {dT_line[opt[3]]} °C)')
    plt.colorbar(c, label = 'КПД')
    plt.tight_layout()
    plt.savefig('/'.join([dir_opt_img, 'KPD, P_in, P_out (чистое).png']), dpi=300, bbox_inches='tight')
    plt.clf()
    
    c = plt.contourf(P_out_line, P_in_line, KPD[:, :, opt[2], opt[3]], levels = 500, cmap = 'turbo')
    plt.clabel(plt.contour(P_out_line, P_in_line, KPD[:, :, opt[2], opt[3]], colors = 'black', levels = 5, linewidths = 0.5))
    plt.xlabel('Давление на выходе из турбины $P_{out}$, Па')
    plt.ylabel('Давление на входе в турбину $P_{in}$, Па')
    P_in_arg_slice, P_out_arg_slice = np.where(np.any(~np.isnan(KPD[:, :, opt[2], opt[3]]), axis = 1))[0], np.where(np.any(~np.isnan(KPD[:, :, opt[2], opt[3]]), axis = 0))[0]
    plt.xlim(P_out_line[P_out_arg_slice[0]], P_out_line[P_out_arg_slice[-1]])
    plt.ylim(P_in_line[P_in_arg_slice[0]], P_in_line[P_in_arg_slice[-1]])
    plt.title(f'КПД ($T_{{before}}$ = {T_before_line[opt[2]]} °C, $dT_{{min}}$ = {dT_line[opt[3]]} °C)')
    plt.colorbar(c, label = 'КПД')
    plt.tight_layout()
    plt.savefig('/'.join([dir_opt_img, 'KPD, P_in, P_out (сглаженное).png']), dpi=300, bbox_inches='tight')
    plt.clf()
    '''
    
    #Графики
    dir_opt_img = '/'.join([dir_opt, 'img'])
    os.makedirs(dir_opt_img, exist_ok = True)
    
    # ---- Новая карта чувствительности (с интерполяцией griddata) ----
    # Данные для среза при оптимальных T_before и dT
    P_in_vals = P_in_line
    P_out_vals = P_out_line
    KPD_slice = KPD[:, :, opt[2], opt[3]]  # размер (len(P_in), len(P_out))
    
    # Собираем точки, где КПД не NaN
    x_data, y_data, z_data = [], [], []
    for i, P_in in enumerate(P_in_vals):
        for j, P_out in enumerate(P_out_vals):
            val = KPD_slice[i, j]
            if not np.isnan(val):
                x_data.append(P_out)   # ось X – P_out
                y_data.append(P_in)    # ось Y – P_in
                z_data.append(val)
                x_data = np.array(x_data)
                y_data = np.array(y_data)
                z_data = np.array(z_data)
                
    if len(x_data) > 0:
        # Мелкая сетка для интерполяции
        xi = np.linspace(x_data.min(), x_data.max(), 300)
        yi = np.linspace(y_data.min(), y_data.max(), 300)
        X, Y = np.meshgrid(xi, yi)
        Z = griddata((x_data, y_data), z_data, (X, Y), method='linear')
        
        # Ограничение минимального отображаемого КПД = 0.4
        Z[Z < 0.4] = np.nan
        
        fig, ax = plt.subplots(figsize=(9, 6.5))
        levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 25)
        cf = ax.contourf(X, Y, Z, levels=levels, cmap='turbo')
        cs = ax.contour(X, Y, Z, levels=10, linewidths=0.8, colors='black')
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.3f')
        
        # Точки исходных расчётных данных
        ax.scatter(x_data, y_data, s=18, edgecolors='black', linewidths=0.3, color='white')
        
        # Отметка максимума
        imax = np.argmax(z_data)
        ax.scatter(x_data[imax], y_data[imax], s=90, marker='*', edgecolors='black', linewidths=0.8, color='red')
        ax.annotate(
            f'max: {z_data[imax]:.3f}\nP_in={y_data[imax]:.1e} Pa, P_out={x_data[imax]:.1e} Pa',
        (x_data[imax], y_data[imax]),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=9
        )
        
        ax.set_xlabel('Давление на выходе из турбины $P_{out}$, Па')
        ax.set_ylabel('Давление на входе в турбину $P_{in}$, Па')
        ax.set_title(f'КПД ($T_{{before}}$ = {T_before_line[opt[2]]} °C, $dT_{{min}}$ = {dT_line[opt[3]]} °C)')
        ax.grid(True, alpha=0.2)
        
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label('КПД')
        
        plt.tight_layout()
        plt.savefig('/'.join([dir_opt_img, 'KPD_map.png']), dpi=300, bbox_inches='tight')
        plt.clf()
        
        
        plt.plot(T_before_line, KPD[opt[0], opt[1], :, opt[3]], 'r')
        plt.locator_params(axis = 'x', integer = True)
        plt.xlabel('Температура на входе в реактор, °C')
        plt.ylabel('КПД')
        plt.tight_layout()
        plt.savefig('/'.join([dir_opt_img, 'T_before.png']), dpi = 300, bbox_inches = 'tight')
        plt.clf()
        
        plt.plot(dT_line, KPD_dT, 'r')
        plt.locator_params(axis = 'x', integer = True)
        plt.xlabel('Температура в точке пинча, °C')
        plt.ylabel('КПД')
        plt.tight_layout()
        plt.savefig('/'.join([dir_opt_img, 'dT.png']), dpi = 300, bbox_inches = 'tight')
        plt.clf()
        
'''
    plt.plot(np.int8(dP_line * 1e2), KPD_dP, 'r')
    plt.locator_params(axis = 'x', integer = True)
    plt.xlabel('Падение давления в рекуператоре $dP$, %')
    plt.ylabel('КПД')
    plt.tight_layout()
    plt.savefig('/'.join([dir_opt_img, 'dP.png']), dpi = 300, bbox_inches = 'tight')
    plt.clf()
    
    KPD_eff = KPD_eff.reshape(len(eff_line['Turb']), len(eff_line['Pump']))
    c = plt.contourf(eff_line['Pump'], eff_line['Turb'], KPD_eff, levels = 500, cmap = 'turbo')    
    plt.clabel(plt.contour(eff_line['Pump'], eff_line['Turb'], KPD_eff, colors = 'black', levels = 12, linewidths = 0.5))
    plt.xlabel('КПД компрессора $eff_{Pump}$')
    plt.ylabel('КПД турбины $eff_{Turb}$')
    eff_turb_arg_slice, eff_pump_arg_slice = np.where(np.any(~np.isnan(KPD_eff), axis = 1))[0], np.where(np.any(~np.isnan(KPD_eff), axis = 0))[0]
    plt.xlim(eff_line['Pump'][eff_pump_arg_slice[0]], eff_line['Pump'][eff_pump_arg_slice[-1]])
    plt.ylim(eff_line['Turb'][eff_turb_arg_slice[0]], eff_line['Turb'][eff_turb_arg_slice[-1]])
    plt.title('КПД')
    plt.colorbar(c, label = 'КПД')
    plt.tight_layout()
    plt.savefig('/'.join([dir_opt_img, 'eff.png']), dpi=300, bbox_inches='tight')
    plt.clf()

    print(time() - start)
'''