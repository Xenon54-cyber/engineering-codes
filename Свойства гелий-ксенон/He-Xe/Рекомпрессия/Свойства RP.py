import numpy as np
import pandas as pd
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
from scipy.optimize import newton, fsolve
from scipy.interpolate import RegularGridInterpolator
from matplotlib.pyplot import plot
import os

if 'RP' not in locals():
    RP = REFPROPFunctionLibrary('C:\Program Files (x86)\REFPROP')
RP.FLAGSdll('PR', 1)

#T_min, T_max = 500, 500
#T_min, T_max = 500, 600
T_min, T_max = 300, 1200
gases = ['Helium', 'Xenon']
params = ['D', 'H', 'S', 'Cp', 'Vis', 'Tcx']

#x_line = np.arange(1, 2, dtype = np.float64) / 10
x_line = np.arange(0, 11, dtype = np.float64) / 10
#x_line = np.arange(101, dtype = np.float64) / 100
#P_line = np.arange(1, 5, dtype = np.float64) * 1e6
P_line = np.arange(1, 51, dtype = np.float64) * 1e5
#P_line = np.append(np.arange(1, 10, dtype = np.float64) * 1e5, np.arange(1, 21, dtype = np.float64) * 1e6)
T_line = np.arange(T_min, T_max + 1, dtype = np.float64)

props = np.zeros([len(params), len(x_line), len(P_line), len(T_line)], dtype = np.float64)

for x_num, x in enumerate(x_line):
    for P_num, P in enumerate(P_line):
        for T_num, T in enumerate(T_line):
            props[:, x_num, P_num, T_num] = np.float64(RP.REFPROPdll(';'.join(gases), 'PT', ';'.join(params), RP.MOLAR_BASE_SI, 0, 0, P, T, [1 - x, x]).Output[:6])         
 
np.save('props_RP.npy', [RegularGridInterpolator([x_line, P_line, T_line], props[params.index(param)]) for param in params[:6]])

'''
data_props = np.zeros((6, 3))
for param_num, param in enumerate(params[:6]):
    prop_true = np.float64(RP.REFPROPdll(';'.join(gases), 'PT', param, RP.MOLAR_BASE_SI, 0, 0, P_line[0], T_line[0], [1 - x_line[0], x_line[0]]).Output[:1])[0]
    prop_not_true = props[params.index(param), 0, 0, 0]
    data_props[param_num, :] = [prop_true, prop_not_true, abs(prop_true - prop_not_true) / prop_true * 100]
os.makedirs('res', exist_ok = True)
pd.DataFrame(data_props, columns = ['True', 'не_True', 'd%'], index = params[:6]).to_excel(f'res/x = {x[0, 0, 0]}; P = {P_line[0]:.2e}; T = {T_line[0]}.xlsx')
'''