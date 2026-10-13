import numpy as np
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
from scipy.interpolate import RegularGridInterpolator

if 'RP' not in locals():
    RP = REFPROPFunctionLibrary('C:\Program Files (x86)\REFPROP')
RP.FLAGSdll('PR', 1)

T_min, T_max = 250, 450
params = ['D', 'H', 'S', 'Cp', 'Vis', 'Tcx']

x_line = np.arange(0, 11, 1, dtype = np.float64) * 1e-1
P_line = np.append(np.arange(1, 10, 1, dtype = np.float64) * 1e5, np.arange(10, 62, 2, dtype = np.float64) * 1e5)
T_line = np.arange(T_min, T_max + 1, dtype = np.float64)

props = np.zeros([len(params), len(x_line), len(P_line), len(T_line)], dtype = np.float64)

for P_num, P in enumerate(P_line):
    for T_num, T in enumerate(T_line):
        props[:, :, P_num, T_num] = np.float64(RP.REFPROPdll('Water', 'PT', ';'.join(params), RP.MASS_BASE_SI, 0, 0, P, T, [1]).Output[:6])[:, None]

np.save('props_water.npy', [RegularGridInterpolator([x_line, P_line, T_line], props[params.index(param)]) for param in params[:6]])