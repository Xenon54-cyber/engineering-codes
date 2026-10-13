"""
Верификация props.npy по данным заказчика (НИКИЭТ, Смесь.docx).

Состав: x_Xe = 0.283, M_mix ≈ 40.02 г/моль.
Проверяем: Vis, Cp, D при P = 2.0 и 4.0 МПа, T = 350…1400 К.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('res/verify', exist_ok=True)

# =====================================================================
# 1. Загрузка props.npy
# =====================================================================
params = ['D', 'H', 'S', 'Cp', 'Vis', 'Tcx']
props = dict(zip(params, np.load('props.npy', allow_pickle=True)))

# Состав (совпадает с данными заказчика)
x_test = 0.283

# =====================================================================
# 2. Данные заказчика (НИКИЭТ, Смесь.docx)
# =====================================================================

# --- Вязкость, мкПа·с ---
vis_customer = {
    'T':   np.array([370, 380, 500, 550, 1000, 1050, 1300, 1350, 1400]),
    '2.0': np.array([31.8339, 32.5211, 40.5103, 43.6828, 68.5243, 70.9777, 82.5614, 84.7677, 86.9385]),
    '4.0': np.array([32.9327, 33.5541, 41.1615, 44.2596, 68.8317, 71.2769, 82.8050, 85.0080, 87.1822]),
}

# --- Теплоёмкость, Дж/(кг·К) ---
cp_customer = {
    'T':   np.array([350, 400, 500, 600, 1000, 1100, 1300, 1400]),
    '2.0': np.array([542.32, 534.15, 527.10, 524.22, 520.78, 520.50, 520.13, 520.04]),
    '4.0': np.array([577.50, 553.18, 535.82, 529.42, 522.27, 521.62, 520.88, 520.60]),
}

# --- Плотность, кг/м³ ---
rho_customer = {
    'T':   np.array([370, 380, 500, 550, 1000, 1050, 1300, 1350]),
    '2.0': np.array([27.4382, 26.6050, 19.5953, 17.7115, 9.5990, 9.1389, 7.3772, 7.1038]),
    '4.0': np.array([58.2439, 56.1030, 39.8845, 35.8187, 19.1390, 18.2143, 14.6967, 14.1532]),
}

# =====================================================================
# 3. Верификация Vis
# =====================================================================
rows = []
for T, v2, v4 in zip(vis_customer['T'], vis_customer['2.0'], vis_customer['4.0']):
    my2 = float(props['Vis']((x_test, 2.0e6, T)))
    my4 = float(props['Vis']((x_test, 4.0e6, T)))
    rows.append({
        'T_K': T,
        'vis_2.0_customer': v2, 'vis_2.0_props': my2, 'err_2.0_%': (my2 - v2)/v2*100,
        'vis_4.0_customer': v4, 'vis_4.0_props': my4, 'err_4.0_%': (my4 - v4)/v4*100,
    })
df_vis = pd.DataFrame(rows)
print('=== Vis: props.npy vs заказчик ===')
print(df_vis.to_string(index=False))
print(f'Среднее |err 2.0 МПа|: {df_vis["err_2.0_%"].abs().mean():.2f} %')
print(f'Среднее |err 4.0 МПа|: {df_vis["err_4.0_%"].abs().mean():.2f} %')
df_vis.to_excel('res/verify/Vis_customer.xlsx', index=False)

# =====================================================================
# 4. Верификация Cp
# =====================================================================
rows = []
for T, c2, c4 in zip(cp_customer['T'], cp_customer['2.0'], cp_customer['4.0']):
    my2 = float(props['Cp']((x_test, 2.0e6, T)))
    my4 = float(props['Cp']((x_test, 4.0e6, T)))
    rows.append({
        'T_K': T,
        'cp_2.0_customer': c2, 'cp_2.0_props': my2, 'err_2.0_%': (my2 - c2)/c2*100,
        'cp_4.0_customer': c4, 'cp_4.0_props': my4, 'err_4.0_%': (my4 - c4)/c4*100,
    })
df_cp = pd.DataFrame(rows)
print('\n=== Cp: props.npy vs заказчик ===')
print(df_cp.to_string(index=False))
print(f'Среднее |err 2.0 МПа|: {df_cp["err_2.0_%"].abs().mean():.2f} %')
print(f'Среднее |err 4.0 МПа|: {df_cp["err_4.0_%"].abs().mean():.2f} %')
df_cp.to_excel('res/verify/Cp_customer.xlsx', index=False)

# =====================================================================
# 5. Верификация D
# =====================================================================
rows = []
for T, r2, r4 in zip(rho_customer['T'], rho_customer['2.0'], rho_customer['4.0']):
    my2 = float(props['D']((x_test, 2.0e6, T)))
    my4 = float(props['D']((x_test, 4.0e6, T)))
    rows.append({
        'T_K': T,
        'rho_2.0_customer': r2, 'rho_2.0_props': my2, 'err_2.0_%': (my2 - r2)/r2*100,
        'rho_4.0_customer': r4, 'rho_4.0_props': my4, 'err_4.0_%': (my4 - r4)/r4*100,
    })
df_rho = pd.DataFrame(rows)
print('\n=== D: props.npy vs заказчик ===')
print(df_rho.to_string(index=False))
print(f'Среднее |err 2.0 МПа|: {df_rho["err_2.0_%"].abs().mean():.2f} %')
print(f'Среднее |err 4.0 МПа|: {df_rho["err_4.0_%"].abs().mean():.2f} %')
df_rho.to_excel('res/verify/D_customer.xlsx', index=False)

# =====================================================================
# 6. Графики
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# --- Vis ---
ax = axes[0]
T_wide = np.linspace(350, 1400, 100)
ax.scatter(vis_customer['T'], vis_customer['2.0'], s=60, marker='o',
           facecolors='none', edgecolors='blue', linewidths=1.8,
           label='Заказчик, 2.0 МПа', zorder=5)
ax.scatter(vis_customer['T'], vis_customer['4.0'], s=60, marker='s',
           facecolors='none', edgecolors='green', linewidths=1.8,
           label='Заказчик, 4.0 МПа', zorder=5)
ax.plot(T_wide, [float(props['Vis']((x_test, 2.0e6, T))) for T in T_wide],
        'b--', linewidth=1.5, label='props.npy, 2.0 МПа')
ax.plot(T_wide, [float(props['Vis']((x_test, 4.0e6, T))) for T in T_wide],
        'g--', linewidth=1.5, label='props.npy, 4.0 МПа')
ax.set_xlabel('T, К'); ax.set_ylabel(r'$\mu$, мкПа$\cdot$с')
ax.set_title('Вязкость vs заказчик')
ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

# --- Cp ---
ax = axes[1]
ax.scatter(cp_customer['T'], cp_customer['2.0'], s=60, marker='o',
           facecolors='none', edgecolors='blue', linewidths=1.8,
           label='Заказчик, 2.0 МПа', zorder=5)
ax.scatter(cp_customer['T'], cp_customer['4.0'], s=60, marker='s',
           facecolors='none', edgecolors='green', linewidths=1.8,
           label='Заказчик, 4.0 МПа', zorder=5)
ax.plot(T_wide, [float(props['Cp']((x_test, 2.0e6, T))) for T in T_wide],
        'b--', linewidth=1.5, label='props.npy, 2.0 МПа')
ax.plot(T_wide, [float(props['Cp']((x_test, 4.0e6, T))) for T in T_wide],
        'g--', linewidth=1.5, label='props.npy, 4.0 МПа')
ax.set_xlabel('T, К'); ax.set_ylabel('Cp, Дж/(кг·К)')
ax.set_title('Теплоёмкость vs заказчик')
ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

# --- D ---
ax = axes[2]
ax.scatter(rho_customer['T'], rho_customer['2.0'], s=60, marker='o',
           facecolors='none', edgecolors='blue', linewidths=1.8,
           label='Заказчик, 2.0 МПа', zorder=5)
ax.scatter(rho_customer['T'], rho_customer['4.0'], s=60, marker='s',
           facecolors='none', edgecolors='green', linewidths=1.8,
           label='Заказчик, 4.0 МПа', zorder=5)
ax.plot(T_wide, [float(props['D']((x_test, 2.0e6, T))) for T in T_wide],
        'b--', linewidth=1.5, label='props.npy, 2.0 МПа')
ax.plot(T_wide, [float(props['D']((x_test, 4.0e6, T))) for T in T_wide],
        'g--', linewidth=1.5, label='props.npy, 4.0 МПа')
ax.set_xlabel('T, К'); ax.set_ylabel(r'$\rho$, кг/м³')
ax.set_title('Плотность vs заказчик')
ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('res/verify/verification_customer.png', dpi=300, bbox_inches='tight')
plt.close()

# =====================================================================
# 7. Сводка
# =====================================================================
print('\n\n========== СВОДКА ==========')
print(f'Vis, 2.0 МПа:  {df_vis["err_2.0_%"].abs().mean():6.2f} %')
print(f'Vis, 4.0 МПа:  {df_vis["err_4.0_%"].abs().mean():6.2f} %')
print(f'Cp,  2.0 МПа:  {df_cp["err_2.0_%"].abs().mean():6.2f} %')
print(f'Cp,  4.0 МПа:  {df_cp["err_4.0_%"].abs().mean():6.2f} %')
print(f'D,   2.0 МПа:  {df_rho["err_2.0_%"].abs().mean():6.2f} %')
print(f'D,   4.0 МПа:  {df_rho["err_4.0_%"].abs().mean():6.2f} %')

print('\nГотово:')
print('  res/verify/Vis_customer.xlsx')
print('  res/verify/Cp_customer.xlsx')
print('  res/verify/D_customer.xlsx')
print('  res/verify/verification_customer.png')