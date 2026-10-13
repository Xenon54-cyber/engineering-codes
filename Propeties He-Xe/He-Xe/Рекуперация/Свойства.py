import numpy as np
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
from scipy.optimize import newton, fsolve
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import os

if 'RP' not in locals():
    RP = REFPROPFunctionLibrary('C:\Program Files (x86)\REFPROP')
RP.FLAGSdll('PR', 1)

T_min, T_max = 250, 1400
gases = ['Helium', 'Xenon']
params = ['D', 'H', 'S', 'Cp', 'Vis', 'Tcx', 'B', 'C', 'dB', 'dC']
params_0 = ['H0', 'S0', 'Vis', 'Tcx']

x_line = np.arange(28, 30, 1, dtype = np.float64) * 1e-2
#x_line = np.arange(0, 1.1, 0.1, dtype = np.float64)
#P_line = np.arange(1, 56, 1, dtype = np.float64) * 1e5
P_line = np.append(np.arange(1, 10, 1, dtype = np.float64) * 1e5, np.arange(10, 62, 2, dtype = np.float64) * 1e5)
T_line = np.arange(T_min - 2, T_max + 3, dtype = np.float64)

props = np.zeros([len(params), len(x_line), len(P_line), len(T_line)], dtype = np.float64)
props_0 = np.zeros([len(gases), len(params_0), len(P_line), len(T_line[1:-1])], dtype = np.float64)

M_He, M_Xe = 4e-3, 131.29e-3
R = 8.31446262

zero_C = lambda C_mix, x, T: -np.float64(RP.REFPROPdll(';'.join(gases), 'PT', 'Cvir', RP.MOLAR_BASE_SI, 0, 0, 0, T, [1 - x, x]).Output[:1])[0] + (1 - x)**3 * C_0 + 3 * (1 - x)**2 * x * C_mix[0] + 3 * (1 - x) * x**2 * C_mix[1] + x**3 * C_1

for T_num, T in enumerate(T_line):
    B_0, C_0 = np.float64(RP.REFPROPdll(gases[0], 'PT', 'Bvir;Cvir', RP.MOLAR_BASE_SI, 0, 0, 0, T, [1]).Output[:2])
    B_1, C_1 = np.float64(RP.REFPROPdll(gases[1], 'PT', 'Bvir;Cvir', RP.MOLAR_BASE_SI, 0, 0, 0, T, [1]).Output[:2])
    B_01 = 2 * np.float64(RP.REFPROPdll(';'.join(gases), 'PT', 'Bvir', RP.MOLAR_BASE_SI, 0, 0, 0, T, [0.5, 0.5]).Output[:1][0]) - 0.5 * (B_0 + B_1)
    C_001, C_011 = fsolve(lambda C_mix, T: [zero_C(C_mix, x, T) for x in [0.5, 0.7]], np.float64([(C_0**2 * C_1)**(1/3), (C_0 * C_1**2)**(1/3)]), args = T)
    for x_num, x in enumerate(x_line):
        props[params.index('B'), x_num, :, T_num] = (1 - x)**2 * B_0 + 2 * (1 - x) * x * B_01 + x**2 * B_1
        props[params.index('C'), x_num, :, T_num] = (1 - x)**3 * C_0 + 3 * (1 - x)**2 * x * C_001 + 3 * x**2 * (1 - x) * C_011 + x**3 * C_1

props[params.index('dB'):params.index('dC') + 1, :, :, 1:-1] = (props[params.index('B'):params.index('C') + 1, :, :, 2:] - props[params.index('B'):params.index('C') + 1, :, :, :-2]) / 2

for P_num, P in enumerate(P_line):
    for T_num, T in enumerate(T_line[1:-1]):
        for gas_num, gas in enumerate(gases):
            props_0[gas_num, :params_0.index('S0') + 1, P_num, T_num] = np.float64(RP.REFPROPdll(gas, 'PT', 'H0;S0', RP.MOLAR_BASE_SI, 0, 0, P, T, [1]).Output[:2])
            props_0[gas_num, params_0.index('Vis'):, P_num, T_num] = np.float64(RP.REFPROPdll(gas, 'PT', 'Vis;Tcx', RP.MASS_BASE_SI, 0, 0, 0, T, [1]).Output[:2])
            
        for x_num, x in enumerate(x_line):
            B, C = props[params.index('B'), x_num, P_num, T_num + 1], props[params.index('C'), x_num, P_num, T_num + 1]
            dB, dC = props[params.index('dB'), x_num, P_num, T_num + 1], props[params.index('dC'), x_num, P_num, T_num + 1]
            Rho = newton(lambda rho: -P / (R * T) + rho + rho**2 * B + rho**3 * C, P / (R * T))
            
            H0 = props_0[0, params_0.index('H0'), P_num, T_num] * (1 - x) + props_0[1, params_0.index('H0'), P_num, T_num] * x
            
            props[params.index('D'), x_num, P_num, T_num + 1] = Rho
            props[params.index('H'), x_num, P_num, T_num + 1] = H0 - R * T**2 * (dB * Rho + 0.5 * dC * Rho**2) + R * T * (B * Rho + C * Rho**2)

props[params.index('Cp'), :, :, 2:-2] = (props[params.index('H'), :, :, 3:-1] - props[params.index('H'), :, :, 1:-3]) / 2
props = props[:, :, :, 2:-2]
props_0 = props_0[:, :, :, 1:-1]
T_line = T_line[2:-2]

x = x_line[:, np.newaxis, np.newaxis]
P = P_line[np.newaxis, :, np.newaxis]
T = T_line[np.newaxis, np.newaxis, :]

B, C = props[params.index('B')], props[params.index('C')]
dB, dC = props[params.index('dB')], props[params.index('dC')]
Rho = props[params.index('D')]

S0 = props_0[0, params_0.index('S0'), :, :] * (1 - x) + props_0[1, params_0.index('S0'), :, :] * x - R * ((1 - x) * np.log(np.where(x != 1, 1 - x, 1)) + x * np.log(np.where(x != 0, x, 1)))
mu_He, mu_Xe = props_0[0, params_0.index('Vis'), :, :], props_0[1, params_0.index('Vis'), :, :]
lambda_He, lambda_Xe = props_0[0, params_0.index('Tcx'), :, :], props_0[1, params_0.index('Tcx'), :, :]

M_mix = M_He * (1 - x) + M_Xe * x
V_cr_Xe, V_cr_He, T_cr_Xe, T_cr_He = 412.45e-6, 190.04e-6, 289.7, 5.2
V_cr = V_cr_Xe * x + V_cr_He * (1 - x)
T_cr = (V_cr_Xe * T_cr_Xe * x + V_cr_He * T_cr_He * (1 - x)) / V_cr
Rho_r = 0.291 * Rho * V_cr
f, n = 1.060, 0.658754

Psi_mu = 0.221 * Rho_r + 1.062 * Rho_r**2 - 0.509 * Rho_r**3 + 0.225 * Rho_r**4
mu_mix = 3.40998e-7 * (T - 45.89)**n
phi12 = mu_Xe / mu_mix * (2 * M_He * M_Xe / (M_He + M_Xe)**2) * (5 / (3 * 1.1) + M_He / M_Xe)
phi21 = mu_He / mu_mix * (2 * M_He * M_Xe / (M_He + M_Xe)**2) * (5 / (3 * 1.1) + M_Xe / M_He)
mu_0 = mu_Xe * x / (x + phi12 * (1 - x)) + mu_He * (1 - x) / ((1 - x) + phi21 * x)

Psi_lambda = 0.645 * Rho_r + 0.33 * Rho_r**2 + 0.0368 * Rho_r**3 - 0.0128 * Rho_r**4
lambda_mix = 15 / 4 * R * mu_mix * f * (M_Xe + M_He) / (2 * M_Xe * M_He)
L11 = x**2 / lambda_Xe + 0.5 * x * (1 - x) / lambda_mix * (15 / 2 * M_Xe**2 + (25 / 4 - 3 * 1.1) * M_He**2 + 4 * 1.1 * M_Xe * M_He) / (1.1 * (M_Xe + M_He)**2)
L22 = (1 - x)**2 / lambda_He + 0.5 * x * (1 - x) / lambda_mix * (15 / 2 * M_He**2 + (25 / 4 - 3 * 1.1) * M_Xe**2 + 4 * 1.1 * M_Xe * M_He) / (1.1 * (M_Xe + M_He)**2)
L12 = -0.5 * x * (1 - x) / lambda_mix * M_Xe * M_He * (55 / 4 - 7 * 1.1) / (1.1 * (M_Xe + M_He)**2)
lambda_0 = (x**2 * L22 - 2 * x * (1 - x) * L12 + (1 - x)**2 * L11) / np.where(x * (1 - x) != 0, L11 * L22 - L12**2, 1) + np.where(x * (1 - x) == 0, lambda_Xe * x + lambda_He * (1 - x), 0)
lambda_cr = 0.304e-4 * T_cr**0.277 / (M_mix**0.465 * (0.291 * V_cr)**0.415)

props[params.index('S')] = S0 - R * T * (dB * Rho + 0.5 * dC * Rho**2) - R * (B * Rho + 0.5 * C * Rho**2) + R * np.log(np.where(1 + B * Rho + C * Rho**2 > 0, 1 + B * Rho + C * Rho**2, 1))
props[params.index('Vis')] = mu_0 + 0.565 * x * 52.25e-6 * Psi_mu
props[params.index('Tcx')] = lambda_0 + (1 - 1 / 2.94) * lambda_cr * Psi_lambda
props[0], props[1:4], props[4] = props[0] * M_mix, props[1:4] / M_mix, props[4] * 1e6

np.save('props.npy', [RegularGridInterpolator([x_line, P_line, T_line], props[params.index(param)]) for param in params[:6]])

'''
data_props = np.zeros((6, 3))
for param_num, param in enumerate(params[:6]):
    prop_true = np.float64(RP.REFPROPdll(';'.join(gases), 'PT', param, RP.MOLAR_BASE_SI, 0, 0, P_line[0], T_line[0], [1 - x_line[0], x_line[0]]).Output[:1])[0]
    prop_not_true = props[params.index(param), 0, 0, 0]
    data_props[param_num, :] = [prop_true, prop_not_true, abs(prop_true - prop_not_true) / prop_true * 100]
os.makedirs('res', exist_ok = True)
pd.DataFrame(data_props, columns = ['True', 'не_True', 'd%'], index = params[:6]).to_excel(f'res/x = {x[0, 0, 0]}; P = {P_line[0]:.2e}; T = {T_line[0]}.xlsx')
'''