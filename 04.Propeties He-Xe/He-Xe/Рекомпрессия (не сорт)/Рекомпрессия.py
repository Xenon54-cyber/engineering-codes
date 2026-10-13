import numpy as np
import pandas as pd
from scipy.optimize import newton, bisect, least_squares
from matplotlib import pyplot as plt
from multiprocessing import Pool
from time import time
import os

eff = dict(Turb = np.float64(0.923), Pump = np.float64(0.901))
T_max, T_min, dT, G = np.float64([900, 28, 10, 1])
'''
P_in_line = np.arange(10, 13, 1, dtype = np.float64) * 1e6
P_out_line = np.float64([k * P for k in np.arange(1, 6, 1) / 10 for P in P_in_line])
split_line = np.arange(8, 9, 1, dtype = np.float64) / 10
x_line = np.arange(0, 5, 1, dtype = np.float64) / 10
'''

P_in_line = np.arange(4, 5, dtype = np.float64) * 1e6
#P_out_line = np.arange(1, 41, dtype = np.float64) * 1e5
P_out_line = np.arange(1, 41, dtype = np.float64) * 1e5
split_line = np.arange(10, 11, dtype = np.float64) / 10
#x_line = np.arange(0, 31, dtype = np.float64) / 100
x_line = np.arange(0, 6, dtype = np.float64) / 10

data_stack = [(T_max, T_min, P_in, P_out, split, x) for P_in in P_in_line for P_out in P_out_line for split in split_line for x in x_line]

params = ['D', 'H', 'S', 'Cp', 'Vis', 'Tcx']
nodes = pd.DataFrame(index = np.arange(12) + 1, columns = ['T', 'P', 'x', 'G'] + params, dtype = np.float64)
blocks = pd.DataFrame(index = ['Heat', 'Turb', 'Cond', 'Pump_Main', 'Pump_Recomp', 'Exch_Main', 'Exch_Recomp'], columns = ['N', 'Q'], dtype = np.float64)

props = dict(zip(params, np.load('props.npy', allow_pickle = True)))
#props = dict(zip(params, np.load('props_v2.npy', allow_pickle = True)))
#props = dict(zip(params, np.load('props_RP.npy', allow_pickle = True)))
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

def Expander(name, node1, node2):
    P2 = nodes.loc[node2, 'P']
    S1 = nodes.loc[node1, 'S']
    x2 = nodes.loc[node2, 'x']
    T2 = nodes.loc[node1, 'T'] * (P2 / nodes.loc[node1, 'P']) ** 0.4
    H2t = props['H']((x2, P2, T_err(S = S1, x = x2, P = P2, T = T2)))
    
    if 'Turb' in name:
        H2 = nodes.loc[node1, 'H'] * (1 - eff['Turb']) + H2t * eff['Turb']
    else:
        H2 = nodes.loc[node1, 'H'] * (1 - 1/eff['Pump']) + H2t * 1/eff['Pump']
    Node(node2, P = P2, T = T_err(H = H2, x = x2, P = P2, T = T2), G = nodes.loc[node1, 'G'])
    blocks.loc[name, 'N'] = nodes.loc[node2, 'G'] * (nodes.loc[node1, 'H'] - nodes.loc[node2, 'H'])
    pass

def Heat_source(name, node1, node2):
    Node(node2, P = nodes.loc[node1, 'P'], T = nodes.loc[node2, 'T'], G = nodes.loc[node1, 'G'])
    blocks.loc[name,'Q'] = nodes.loc[node2, 'G'] * (nodes.loc[node2, 'H'] - nodes.loc[node1, 'H'])
    pass

def Recompression(args, split):
    nodes.loc[[4, 8, 3, 12], 'T'] = args#np.float64(args)
    
    Node(3, P = nodes.loc[3, 'P'], T = nodes.loc[3, 'T'], G = G)
    Node(4, P = nodes.loc[4, 'P'], T = nodes.loc[4, 'T'], G = G)
    Node(5, P = nodes.loc[4, 'P'], T = nodes.loc[4, 'T'], G = G * split)
    Heat_source('Cond', 5, 6)
    Expander('Pump_Main', 6, 7)
    Node(8, P = nodes.loc[8, 'P'], T = nodes.loc[8, 'T'], G = nodes.loc[7, 'G'])
    Node(9, P = nodes.loc[4, 'P'], T = nodes.loc[4, 'T'], G = G * (1 - split))
    Expander('Pump_Recomp', 9, 10)
    nodes.loc[11, 'H'] = (nodes.loc[8, 'G'] * nodes.loc[8, 'H'] + nodes.loc[10, 'G'] * nodes.loc[10, 'H']) / (nodes.loc[8, 'G'] + nodes.loc[10, 'G'])
    Node(11, P = nodes.loc[11, 'P'], T = T_err(H = nodes.loc[11, 'H'], x = nodes.loc[11, 'x'], P = nodes.loc[11, 'P'], T = nodes.loc[8, 'T']), G = G)
    Node(12, P = nodes.loc[12, 'P'], T = nodes.loc[12, 'T'], G = nodes.loc[11, 'G'])
    
    dT1_1 = nodes.loc[2, 'T'] - nodes.loc[12, 'T']
    dT1_2 = nodes.loc[3, 'T'] - nodes.loc[11, 'T']
    dT2_1 = nodes.loc[3, 'T'] - nodes.loc[8, 'T']
    dT2_2 = nodes.loc[4, 'T'] - nodes.loc[7, 'T']
    
    q1_1 = nodes.loc[2, 'H'] - nodes.loc[3, 'H']
    q1_2 = nodes.loc[12, 'H'] - nodes.loc[11, 'H']
    q2_1 = nodes.loc[3, 'H'] - nodes.loc[4, 'H']
    q2_2 = nodes.loc[8, 'H'] - nodes.loc[7, 'H']
    
    equation = []
    equation.append(min(dT1_1, dT1_2) - dT)
    equation.append(min(dT2_1, dT2_2) - dT)
    equation.append(nodes.loc[2, 'G'] * q1_1 - nodes.loc[11, 'G'] * q1_2)
    equation.append(nodes.loc[3, 'G'] * q2_1 - nodes.loc[7, 'G'] * q2_2)
    
    blocks.loc['Exch_Main','Q'] = nodes.loc[7, 'G'] * (nodes.loc[8, 'H'] - nodes.loc[7, 'H']) - nodes.loc[3, 'G'] * (nodes.loc[3, 'H'] - nodes.loc[4, 'H'])
    blocks.loc['Exch_Recomp','Q'] = nodes.loc[11, 'G'] * (nodes.loc[12, 'H'] - nodes.loc[11, 'H']) - nodes.loc[2, 'G'] * (nodes.loc[2, 'H'] - nodes.loc[3, 'H'])
    return equation

def Brayton(T_max, T_min, P_in, P_out, split, x):
    if P_in > P_out:
        try:
            nodes.loc[:, 'x'] = x
            nodes.loc[[1, 2], 'P'] = P_in, P_out
            nodes.loc[[7, 8, 10, 11, 12], 'P'] = nodes.loc[1, 'P']
            nodes.loc[[3, 4, 5, 6, 9], 'P'] = nodes.loc[2, 'P']
            nodes.loc[[1, 6], 'T'] = T_max + 273.15, T_min + 273.15
            
            Node(1, T = nodes.loc[1, 'T'], P = nodes.loc[1, 'P'], G = G)
            Expander('Turb', 1, 2)
            least_squares(Recompression, [0.5 * (nodes.loc[2, 'T'] + nodes.loc[6, 'T']), 0.4 * (nodes.loc[2, 'T'] + nodes.loc[6, 'T']), 0.6 * (nodes.loc[2, 'T'] + nodes.loc[6, 'T']), nodes.loc[2, 'T'] - dT], args = (split, ), bounds = (nodes.loc[6, 'T'], nodes.loc[2, 'T']), xtol = 1e-6)
            Heat_source('Heat', 12, 1)
        except:
            return np.nan
        
        KPD = (blocks.loc['Turb', 'N'] + blocks.loc['Pump_Main', 'N'] + blocks.loc['Pump_Recomp', 'N']) / blocks.loc['Heat', 'Q']
        balance = np.sum(blocks.loc[:, 'Q']) - np.sum(blocks.loc[:, 'N'])
        #print(f'KPD: {KPD:.3}; Balance: {balance:.2}; P_in: {P_in:.2}; split: {split}\n')
        return KPD
    
    else:
        return np.nan

'''
#Brayton(T_max, T_min, P_in_line[0], P_out_line[0], split_line[0], x_line[0])
Brayton(T_max, T_min, 11e6, 8e6, 0.9, 0.9)
nodes.to_excel('res.xlsx')
'''

if __name__ == '__main__':
    start = time()
    
    with Pool(12) as p:
        KPD = p.starmap(Brayton, data_stack)
    KPD = np.array(KPD).reshape(len(P_in_line), len(P_out_line), len(split_line), len(x_line))

    os.makedirs('res', exist_ok = True)
    pd.DataFrame(np.vstack([np.hstack([P_in, P_out, split, x, KPD[P_in_num, P_out_num, split_num, x_num]]) for P_in_num, P_in in enumerate(P_in_line) for P_out_num, P_out in enumerate(P_out_line) for split_num, split in enumerate(split_line) for x_num, x in enumerate(x_line)]), columns = ['P_in', 'P_out', 'split', 'x', 'KPD']).to_excel(f'res/T_max = {T_max}, T_min = {T_min}.xlsx')
    #pd.DataFrame(np.vstack([np.hstack([P_in, P_out, split, x, KPD[P_in_num, P_out_num, split_num, x_num]]) for P_in_num, P_in in enumerate(P_in_line) for P_out_num, P_out in enumerate(P_out_line) for split_num, split in enumerate(split_line) for x_num, x in enumerate(x_line)]), columns = ['P_in', 'P_out', 'split', 'x', 'KPD']).to_excel(f'res/T_max = {T_max}, T_min = {T_min} - v2.xlsx')
    #pd.DataFrame(np.vstack([np.hstack([P_in, P_out, split, x, KPD[P_in_num, P_out_num, split_num, x_num]]) for P_in_num, P_in in enumerate(P_in_line) for P_out_num, P_out in enumerate(P_out_line) for split_num, split in enumerate(split_line) for x_num, x in enumerate(x_line)]), columns = ['P_in', 'P_out', 'split', 'x', 'KPD']).to_excel(f'res/T_max = {T_max}, T_min = {T_min} - RP.xlsx')
    
    print(time() - start)
    
    #График
    
    data_2d = KPD[0, :, 0, :].T
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(P_out_line, x_line, data_2d, shading='auto', cmap='viridis')
    ax.set_xlabel('Давление на выходе $P_{out}$, Па')
    ax.set_ylabel('Доля ксенона $x$')
    ax.set_title('КПД в зависимости от $P_{out}$ и $x$')
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('КПД')
    
    plt.tight_layout()
    plt.savefig('KPD.png', dpi = 300, bbox_inches = 'tight')
    #plt.savefig('KPD_v2.png', dpi = 300, bbox_inches = 'tight')
    #plt.savefig('KPD_RP.png', dpi = 300, bbox_inches = 'tight')