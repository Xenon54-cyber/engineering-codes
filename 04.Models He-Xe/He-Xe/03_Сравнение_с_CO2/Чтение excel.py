import numpy as np
import pandas as pd
import os

x = 0.283
params = ['D', 'H', 'S', 'Cp', 'Vis', 'Tcx']
props = dict(zip(params, np.load('props.npy', allow_pickle = True)))

data_rho = pd.read_excel('data.xls', sheet_name = 0)
data_Cp = pd.read_excel('data.xls', sheet_name = 4)
data_mu = pd.read_excel('data.xls', sheet_name = 3)
data_lambda = pd.read_excel('data.xls', sheet_name = 7)

index_P = np.array([num for num in range(2, 48, 3)])
index_T = np.arange(5, 47)

T_line = data_rho.iloc[index_T, 1].to_numpy(np.float64)
P_line = data_rho.iloc[3, index_P].to_numpy(np.float64) * 1e6

data_rho_Xe = data_rho.iloc[index_T, index_P].to_numpy(np.float64)
data_rho_He = data_rho.iloc[index_T, index_P + 1].to_numpy(np.float64)
data_rho_HeXe = data_rho.iloc[index_T, index_P + 2].to_numpy(np.float64)

data_Cp_Xe = data_Cp.iloc[index_T, index_P].to_numpy(np.float64)
data_Cp_HeXe = data_Cp.iloc[index_T, index_P + 2].to_numpy(np.float64)

data_mu_Xe = data_mu.iloc[index_T, index_P + np.arange(len(index_P)) * 2].to_numpy(np.float64)
data_mu_He = data_mu.iloc[index_T, index_P + (np.arange(len(index_P)) + 0.5) * 2].to_numpy(np.float64)
data_mu_HeXe = data_mu.iloc[index_T, index_P + (np.arange(len(index_P)) + 2) * 2].to_numpy(np.float64)

data_lambda_Xe = data_lambda.iloc[index_T, index_P + np.arange(len(index_P)) * 2].to_numpy(np.float64)
data_lambda_He = data_lambda.iloc[index_T, index_P + (np.arange(len(index_P)) + 0.5) * 2].to_numpy(np.float64)
data_lambda_HeXe = data_lambda.iloc[index_T, index_P + (np.arange(len(index_P)) + 2) * 2].to_numpy(np.float64)

model_rho_Xe = np.float64([[props['D']((1, P, T)) for P in P_line] for T in T_line])
model_rho_He = np.float64([[props['D']((0, P, T)) for P in P_line] for T in T_line])
model_rho_HeXe = np.float64([[props['D']((x, P, T)) for P in P_line] for T in T_line])

model_Cp_Xe = np.float64([[props['Cp']((1, P, T)) for P in P_line] for T in T_line])
model_Cp_HeXe = np.float64([[props['Cp']((x, P, T)) for P in P_line] for T in T_line])

model_mu_Xe = np.float64([[props['Vis']((1, P, T)) for P in P_line] for T in T_line])
model_mu_He = np.float64([[props['Vis']((0, P, T)) for P in P_line] for T in T_line])
model_mu_HeXe = np.float64([[props['Vis']((x, P, T)) for P in P_line] for T in T_line])

model_lambda_Xe = np.float64([[props['Tcx']((1, P, T)) for P in P_line] for T in T_line]) * 1e3
model_lambda_He = np.float64([[props['Tcx']((0, P, T)) for P in P_line] for T in T_line]) * 1e3
model_lambda_HeXe = np.float64([[props['Tcx']((x, P, T)) for P in P_line] for T in T_line]) * 1e3

out_rho_Xe = np.vstack([[data_rho_Xe[:, P_num], model_rho_Xe[:, P_num], np.abs((data_rho_Xe[:, P_num] - model_rho_Xe[:, P_num]) * 100 / data_rho_Xe[:, P_num])] for P_num, P in enumerate(P_line)]).T
out_rho_He = np.vstack([[data_rho_He[:, P_num], model_rho_He[:, P_num], np.abs((data_rho_He[:, P_num] - model_rho_He[:, P_num]) * 100 / data_rho_He[:, P_num])] for P_num, P in enumerate(P_line)]).T
out_rho_HeXe = np.vstack([[data_rho_HeXe[:, P_num], model_rho_HeXe[:, P_num], np.abs((data_rho_HeXe[:, P_num] - model_rho_HeXe[:, P_num]) * 100 / data_rho_HeXe[:, P_num])] for P_num, P in enumerate(P_line)]).T

out_Cp_Xe = np.vstack([[data_Cp_Xe[:, P_num], model_Cp_Xe[:, P_num], np.abs((data_Cp_Xe[:, P_num] - model_Cp_Xe[:, P_num]) * 100 / data_Cp_Xe[:, P_num])] for P_num, P in enumerate(P_line)]).T
out_Cp_HeXe = np.vstack([[data_Cp_HeXe[:, P_num], model_Cp_HeXe[:, P_num], np.abs((data_Cp_HeXe[:, P_num] - model_Cp_HeXe[:, P_num]) * 100 / data_Cp_HeXe[:, P_num])] for P_num, P in enumerate(P_line)]).T

out_mu_Xe = np.vstack([[data_mu_Xe[:, P_num], model_mu_Xe[:, P_num], np.abs((data_mu_Xe[:, P_num] - model_mu_Xe[:, P_num]) * 100 / data_mu_Xe[:, P_num])] for P_num, P in enumerate(P_line)]).T
out_mu_He = np.vstack([[data_mu_He[:, P_num], model_mu_He[:, P_num], np.abs((data_mu_He[:, P_num] - model_mu_He[:, P_num]) * 100 / data_mu_He[:, P_num])] for P_num, P in enumerate(P_line)]).T
out_mu_HeXe = np.vstack([[data_mu_HeXe[:, P_num], model_mu_HeXe[:, P_num], np.abs((data_mu_HeXe[:, P_num] - model_mu_HeXe[:, P_num]) * 100 / data_mu_HeXe[:, P_num])] for P_num, P in enumerate(P_line)]).T

out_lambda_Xe = np.vstack([[data_lambda_Xe[:, P_num], model_lambda_Xe[:, P_num], np.abs((data_lambda_Xe[:, P_num] - model_lambda_Xe[:, P_num]) * 100 / data_lambda_Xe[:, P_num])] for P_num, P in enumerate(P_line)]).T
out_lambda_He = np.vstack([[data_lambda_He[:, P_num], model_lambda_He[:, P_num], np.abs((data_lambda_He[:, P_num] - model_lambda_He[:, P_num]) * 100 / data_lambda_He[:, P_num])] for P_num, P in enumerate(P_line)]).T
out_lambda_HeXe = np.vstack([[data_lambda_HeXe[:, P_num], model_lambda_HeXe[:, P_num], np.abs((data_lambda_HeXe[:, P_num] - model_lambda_HeXe[:, P_num]) * 100 / data_lambda_HeXe[:, P_num])] for P_num, P in enumerate(P_line)]).T

out_dir = ['res_RP/Xe', 'res_RP/He', 'res_RP/He-Xe']
for one_out_dir in out_dir: os.makedirs(one_out_dir, exist_ok = True)
cols = pd.MultiIndex.from_tuples([('T, K', '')] + [(f'P = {P:.1e} Па', col) for P in P_line for col in ['Данные', 'Модель', 'd, %']])

pd.DataFrame(np.hstack([T_line[:, None], out_rho_Xe]), columns = cols).to_excel('/'.join([out_dir[0], 'Плотность.xlsx']))
pd.DataFrame(np.hstack([T_line[:, None], out_rho_He]), columns = cols).to_excel('/'.join([out_dir[1], 'Плотность.xlsx']))
pd.DataFrame(np.hstack([T_line[:, None], out_rho_HeXe]), columns = cols).to_excel('/'.join([out_dir[2], 'Плотность.xlsx']))

pd.DataFrame(np.hstack([T_line[:, None], out_Cp_Xe]), columns = cols).to_excel('/'.join([out_dir[0], 'Теплоёмкость.xlsx']))
pd.DataFrame(np.hstack([T_line[:, None], out_Cp_HeXe]), columns = cols).to_excel('/'.join([out_dir[2], 'Теплоёмкость.xlsx']))

pd.DataFrame(np.hstack([T_line[:, None], out_mu_Xe]), columns = cols).to_excel('/'.join([out_dir[0], 'Вязкость.xlsx']))
pd.DataFrame(np.hstack([T_line[:, None], out_mu_He]), columns = cols).to_excel('/'.join([out_dir[1], 'Вязкость.xlsx']))
pd.DataFrame(np.hstack([T_line[:, None], out_mu_HeXe]), columns = cols).to_excel('/'.join([out_dir[2], 'Вязкость.xlsx']))

pd.DataFrame(np.hstack([T_line[:, None], out_lambda_He]), columns = cols).to_excel('/'.join([out_dir[0], 'Теплопроводность.xlsx']))
pd.DataFrame(np.hstack([T_line[:, None], out_lambda_Xe]), columns = cols).to_excel('/'.join([out_dir[1], 'Теплопроводность.xlsx']))
pd.DataFrame(np.hstack([T_line[:, None], out_lambda_HeXe]), columns = cols).to_excel('/'.join([out_dir[2], 'Теплопроводность.xlsx']))