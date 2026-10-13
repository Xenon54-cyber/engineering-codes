import numpy as np
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
from scipy.interpolate import RegularGridInterpolator

if 'RP' not in locals():
    RP = REFPROPFunctionLibrary('C:\Program Files (x86)\REFPROP')
RP.FLAGSdll('PR', 1)

T_min, T_max = 270, 1400
params = ['D', 'H', 'S', 'Cp', 'Vis', 'Tcx']

x_line = np.arange(1, 2, dtype = np.float64)
P_line = np.arange(1, 100, dtype = np.float64) * 1e6
T_line = np.arange(T_min, T_max + 1, dtype = np.float64)

x = x_line[:, np.newaxis, np.newaxis]

props = np.zeros([len(params), len(x_line), len(P_line), len(T_line)], dtype = np.float64)

for x_num, x in enumerate(x_line):
    for P_num, P in enumerate(P_line):
        for T_num, T in enumerate(T_line):
            props[:, x_num, P_num, T_num] = np.float64(RP.REFPROPdll('CO2', 'PT', ';'.join(params), RP.MASS_BASE_SI, 0, 0, P, T, [x]).Output[:6])         

np.save('props_RP_CO2.npy', [RegularGridInterpolator([x_line, P_line, T_line], props[params.index(param)]) for param in params[:6]])