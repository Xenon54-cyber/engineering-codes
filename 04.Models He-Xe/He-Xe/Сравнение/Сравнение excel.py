import numpy as np
import pandas as pd
import os

params = ['D', 'H', 'S', 'Cp', 'Vis', 'Tcx']
props = dict(zip(params, np.load('props.npy', allow_pickle = True)))

x = 0.283
P_list = np.float64([2, 4]) * 1e6

T_line_1 = np.array([370, 380, 500, 550, 1000, 1050, 1300, 1350])
T_line_2 = np.array([350, 400, 500, 600, 1000, 1100, 1300, 1400])
T_line_3 = np.array([370, 380, 500, 550, 1000, 1050, 1300, 1350, 1400])

props_base_1 = np.array([[27.4382065, 26.6050021, 19.5952986, 17.7115001, 9.5989824, 9.1389297, 7.3772294, 7.1037702], [58.2438776, 56.1029847, 39.8844835, 35.8186661, 19.1389707, 18.2143030, 14.6966779, 14.1532296]])
props_base_2 = np.array([[542.32, 534.15, 527.10, 524.22, 520.78, 520.50, 520.13, 520.04], [577.50, 553.18, 535.82, 529.42, 522.27, 521.62, 520.88, 520.60]])
props_base_3 = np.array([[31.8339, 32.5211, 40.5103, 43.6828, 68.5243, 70.9777, 82.5614, 84.7677, 86.9385], [32.9327, 33.5541, 41.1615, 44.2596, 68.8317, 71.2769, 82.8050, 85.0080, 87.1822]])

props_virs_1 = np.array([np.hstack([props['D']((x, P, T)) for T in T_line_1]) for P in P_list])
props_virs_2 = np.array([np.hstack([props['Cp']((x, P, T)) for T in T_line_2]) for P in P_list])
props_virs_3 = np.array([np.hstack([props['Vis']((x, P, T)) for T in T_line_3]) for P in P_list])

d_1 = np.abs((props_base_1 - props_virs_1)) / props_base_1 * 100
d_2 = np.abs((props_base_2 - props_virs_2)) / props_base_2 * 100
d_3 = np.abs((props_base_3 - props_virs_3)) / props_base_3 * 100

data_1 = np.hstack([T_line_1[:, None], np.hstack([np.vstack([props_base_1[i], props_virs_1[i], d_1[i]]).T for i in range(2)])])
data_2 = np.hstack([T_line_2[:, None], np.hstack([np.vstack([props_base_2[i], props_virs_2[i], d_2[i]]).T for i in range(2)])])
data_3 = np.hstack([T_line_3[:, None], np.hstack([np.vstack([props_base_3[i], props_virs_3[i], d_3[i]]).T for i in range(2)])])

cols = pd.MultiIndex.from_tuples([('T, K'),
                                  ('P = 2 МПа', 'Данные'),
                                  ('P = 2 МПа', 'Модель'),
                                  ('P = 2 МПа', 'd, %'),
                                  ('P = 4 МПа', 'Данные'),
                                  ('P = 4 МПа', 'Модель'),
                                  ('P = 4 МПа', 'd, %')])

os.makedirs('res', exist_ok = True)
pd.DataFrame(data_1, columns = cols).to_excel('res/Плотность.xlsx', index = False)
pd.DataFrame(data_2, columns = cols).to_excel('res/Теплоёмкость.xlsx')
pd.DataFrame(data_3, columns = cols).to_excel('res/Вязкость.xlsx')