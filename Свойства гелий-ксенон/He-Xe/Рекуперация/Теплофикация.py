import numpy as np
import pandas as pd
from scipy.optimize import newton, least_squares
import matplotlib.pyplot as plt
from multiprocessing import Pool
from time import time
import os

fluids = np.array(['He-Xe', 'Water'])
x, P_in, T_max, T_pipe, dP_heat_source, eff_expander, N_pinch = 0.283, 4.8e6, 1060, 1030, dict(Heat = 0.0364e-2, Cool = 0.497e-2), dict(Turb = 92.67e-2, Pump = 87.06e-2), 20
P_water, T_water_max, T_water_min, dP_water = 1.6e6, 70, 40, 5e-2, 
Q, eff_exchanger, KPD_hexe, KPD_water = 2.75e6, 95e-2, 0.95 * 0.9, 0.95 * 0.8
Q_water = Q * eff_exchanger

dP_line = np.arange(3, 7, 1) * 1e-2
dT_line = np.arange(50, 350, 50)
data_stack = [(dP, dT) for dP in dP_line for dT in dT_line]
dP_fix = np.where(dP_line == 5e-2)[0][0]

params = ['D', 'H', 'S', 'Cp', 'Vis', 'Tcx']
nodes = pd.DataFrame(index = np.arange(6), columns = ['Fluid', 'T', 'P', 'x', 'G'] + params, dtype = np.float64)
nodes['Fluid'] = pd.Series(nodes['Fluid'], dtype = np.str_)
blocks = pd.DataFrame(index = ['Heat', 'Pump', 'Exch', 'Pipe', 'Pump_water'], columns = ['Q', 'N', 'Aux'], dtype = np.float64)

props = {fluids[0]: dict(zip(params, np.load('props.npy', allow_pickle = True))), fluids[1]: dict(zip(params, np.load('props_water.npy', allow_pickle = True)))}
#T_min_props, T_max_props = min(props[]['H'].grid[2]), max(props['H'].grid[2])

def T_err(**kwargs):
    params_in = list(kwargs.keys())
    param = [param for param in params_in if param not in ['Fluid', 'x', 'P', 'T']][0]
    
    err = lambda T: kwargs[param] - props[kwargs['Fluid']][param]((kwargs['x'], kwargs['P'], T))
    T = newton(err, kwargs['T'], tol = 1e-6)
    return T

def Node(node, **kwargs):
    nodes.loc[node, kwargs.keys()] = list(kwargs.values())
    nodes.loc[node, params] = [props[nodes.loc[node, 'Fluid']][param]((nodes.loc[node, 'x'], kwargs['P'], kwargs['T'])) for param in params]
    pass

def Expander(name, node1, node2, eff):
    P2 = nodes.loc[node2, 'P']
    S1 = nodes.loc[node1, 'S']
    T2 = nodes.loc[node1, 'T'] * (P2 / nodes.loc[node1, 'P']) ** 0.4
    H2t = props[nodes.loc[node2, 'Fluid']]['H']((nodes.loc[node2, 'x'], P2, T_err(S = S1, Fluid = fluids[0], x = nodes.loc[node2, 'x'], P = P2, T = T2)))
    H2 = nodes.loc[node1, 'H'] * (1 - eff[name]) + H2t * eff[name] if name == 'Turb' else nodes.loc[node1, 'H'] * (1 - 1 / eff[name]) + H2t * 1 / eff[name]
    
    Node(node2, P = P2, T = T_err(H = H2, Fluid = fluids[0], x = nodes.loc[node2, 'x'], P = P2, T = T2), G = nodes.loc[node1, 'G'])
    blocks.loc[name, 'N'] = nodes.loc[node2, 'G'] * (nodes.loc[node1, 'H'] - nodes.loc[node2, 'H'])
    pass

def Heat_source(name, node1, node2):
    Node(node2, P = nodes.loc[node2, 'P'], T = nodes.loc[node2, 'T'], G = nodes.loc[node1, 'G'])
    blocks.loc[name, 'Q'] = nodes.loc[node2, 'G'] * (nodes.loc[node2, 'H'] - nodes.loc[node1, 'H'])
    pass

def Loop(args, dT):
    nodes.loc[2, 'T'] = args[0]
    nodes.loc[[0, 1, 2, 3], 'G'] = args[1]
    
    Node(2, P = nodes.loc[2, 'P'], T = nodes.loc[2, 'T'])
    Expander('Pump', 2, 3, eff_expander)
    
    H_rec, P_rec, T_rec = [np.vstack([np.linspace(nodes.loc[1, param], nodes.loc[2, param], N_pinch), np.linspace(nodes.loc[5, param], nodes.loc[4, param], N_pinch)]) for param in ['H', 'P', 'T']]
    T_pinch = np.min(T_err(H = H_rec[0], Fluid = fluids[0], x = nodes.loc[1, 'x'], P = P_rec[0], T = T_rec[0]) - T_err(H = H_rec[1], Fluid = fluids[1], x = nodes.loc[4, 'x'], P = P_rec[1], T = T_rec[1]))
    
    blocks.loc['Exch', 'Q'] = nodes.loc[1, 'G'] * (nodes.loc[2, 'H'] - nodes.loc[1, 'H'])
    
    eq = []
    eq.append(blocks.loc['Exch', 'Q'] + Q_water)
    eq.append(T_pinch - dT)
    return eq

def Heating(dP, dT):
    nodes.loc[:, 'Fluid'] = [fluids[0]] * 4 + [fluids[1]] * 2
    nodes.loc[:, 'x'] = [x] * 4 + [1] * 2
    nodes.loc[:, 'P'] = [P_in] * 2 + [P_in * (1 - dP)] + [P_in * (1 + dP_heat_source['Heat'])] + [P_water] + [P_water * (1 - dP_water)]
    nodes.loc[[0, 1, 4, 5], 'T'] = np.array([T_max, T_pipe, T_water_min, T_water_max]) + 273.15
    
    for node in [0, 1, 4, 5]: Node(node, P = nodes.loc[node, 'P'], T = nodes.loc[node, 'T'])
    nodes.loc[[4, 5], 'G'] = Q_water / (nodes.loc[5, 'H'] - nodes.loc[4, 'H'])
    least_squares(Loop, [nodes.loc[4, 'T'] + dT, Q_water / (nodes.loc[1, 'H'] - nodes.loc[4, 'H'])], args = (dT,), bounds = ([nodes.loc[4, 'T'], 0], [nodes.loc[1, 'T'], np.inf]), xtol = 1e-6)
    Heat_source('Heat', 3, 0)
    Heat_source('Pipe', 0, 1)
    
    blocks.loc['Pump', 'Aux'] = np.abs(blocks.loc['Pump', 'N']) * KPD_hexe
    blocks.loc['Pump_water', 'Aux'] = nodes.loc[4, 'G'] * (nodes.loc[4, 'P'] - nodes.loc[5, 'P']) / (props[fluids[1]]['D']((nodes.loc[4, 'x'], nodes.loc[4, 'P'], nodes.loc[4, 'T'])) * KPD_water)
    
    KPD = (Q_water - np.sum(blocks.loc[:, 'Aux'])) / Q
    balance = np.sum(blocks.loc[:, 'Q']) - np.sum(blocks.loc[:, 'N'])
    print(f'KPD: {KPD:.3}; Balance: {balance:.2}\n')
    return KPD, blocks.loc['Pump', 'Aux'], blocks.loc['Pump_water', 'Aux'], blocks.loc['Pump', 'Aux'] + blocks.loc['Pump_water', 'Aux']

if __name__ == '__main__':
    start = time()
    dir_out = os.path.join(os.getcwd(), 'res')
    dir_out_img = os.path.join(dir_out, 'img')
    os.makedirs(dir_out_img, exist_ok = True)
    
    with Pool(14) as p:
        KPD, aux_hexe, aux_water, aux_sum = zip(*p.starmap(Heating, data_stack))
    pd.DataFrame(np.column_stack([data_stack, KPD, aux_hexe, aux_water, aux_sum]), columns = ['dP', 'dT', 'KPD', 'aux_hexe', 'aux_water', 'aux_sum']).to_excel(os.path.join(dir_out, 'full_stack.xlsx'), index = False)
    
    plt.figure(figsize = (9, 6), dpi = 300)
    plt.plot(dT_line, np.array(aux_hexe).reshape(len(dP_line), len(dT_line))[dP_fix], marker = 'o', markersize = 5, label = 'Насос He-Xe')
    plt.plot(dT_line, np.array(aux_water).reshape(len(dP_line), len(dT_line))[dP_fix], marker = 'o', markersize = 5, label = 'Водяной насос')
    plt.plot(dT_line, np.array(aux_sum).reshape(len(dP_line), len(dT_line))[dP_fix], marker = 'o', markersize = 5, label = 'Суммарная потребляемая мощность')
    plt.xlabel('Минимальный температурный напор в теплообменнике, °C')
    plt.ylabel('Мощность, кВт')
    plt.legend(loc = 'best')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_out_img, 'Мощность.png'))
    plt.show()
    plt.clf()
    
    plt.figure(figsize = (9, 6), dpi = 300)
    KPD_arr = np.array(KPD).reshape(len(dP_line), len(dT_line))
    for i, dP in enumerate(dP_line): plt.plot(dT_line, KPD_arr[i] * 100, marker = 'o', markersize = 5, label = f'ΔP = {int(round(dP * 100))}%')
    plt.xlabel('Минимальный температурный напор в теплообменнике, °C')
    plt.ylabel('КПД нетто, %')
    plt.legend(loc = 'best')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_out_img, 'КПД'))
    plt.show()
    plt.clf()