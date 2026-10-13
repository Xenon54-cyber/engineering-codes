import matplotlib.pyplot as plt
import numpy as np

# ============ ИСХОДНЫЕ ДАННЫЕ ============
Q_reactor = 2.75000
eta_net   = 0.949583
N_SN      = 0.0038704
eta_TO    = 0.99

loss_reactor   = Q_reactor * 0.005
loss_pipe_hot  = Q_reactor * 0.003
loss_TO        = Q_reactor * 0.010
loss_pipe_cold = Q_reactor * 0.003

N_net  = Q_reactor * eta_net
Q_other = Q_reactor - N_net - N_SN - loss_reactor - loss_pipe_hot - loss_TO - loss_pipe_cold

flows = [
    ('Полезная мощность (нетто)', N_net,          '#2ecc71'),
    ('Потери в реакторе',         loss_reactor,   '#e74c3c'),
    ('Потери в труб. (гор.)',     loss_pipe_hot,  '#e67e22'),
    ('Потери в РТО',              loss_TO,        '#c0392b'),
    ('Потери в труб. (хол.)',     loss_pipe_cold, '#d35400'),
    ('Собственные нужды',         N_SN,           '#f39c12'),
    ('Прочие потери цикла',       Q_other,        '#95a5a6'),
]

labels = [f[0] for f in flows]
sizes  = [f[1] for f in flows]
colors = [f[2] for f in flows]

# ============ РИСУНОК: колесо + таблица ============
fig = plt.figure(figsize=(14, 7))
gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.1])

# --- Левая часть: колесо ---
ax1 = fig.add_subplot(gs[0, 0])
wedges, _ = ax1.pie(
    sizes, colors=colors, startangle=90, counterclock=False,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
# Внутри — только проценты, без названий
for i, w in enumerate(wedges):
    ang = (w.theta2 + w.theta1) / 2
    x = np.cos(np.deg2rad(ang))
    y = np.sin(np.deg2rad(ang))
    pct = sizes[i] / Q_reactor * 100
    if pct > 3:   # подписываем только крупные сектора
        ax1.text(x*0.65, y*0.65, f'{pct:.2f}%',
                 ha='center', va='center', fontsize=11, fontweight='bold',
                 color='white')
    elif pct > 0.5:
        ax1.text(x*1.15, y*1.15, f'{pct:.2f}%',
                 ha='center', va='center', fontsize=8, color='black')

# Центральный круг
centre_circle = plt.Circle((0, 0), 0.35, fc='white', edgecolor='black', linewidth=1)
ax1.add_artist(centre_circle)
ax1.text(0, 0.05, f'{Q_reactor:.3f}', ha='center', va='center',
         fontsize=13, fontweight='bold')
ax1.text(0, -0.10, 'МВт', ha='center', va='center', fontsize=10)

ax1.set_title('Колесо баланса мощности\n(режим макс. КПД)',
              fontsize=13, fontweight='bold', pad=15)
ax1.axis('equal')

# --- Правая часть: таблица-легенда ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')

# Заголовок таблицы
header = ['Статья баланса', 'МВт', '%']
rows = []
for name, val, _ in flows:
    rows.append([name, f'{val:.5f}', f'{val/Q_reactor*100:.4f}'])
rows.append(['ИТОГО подведено', f'{Q_reactor:.5f}', '100.0000'])

table = ax2.table(
    cellText=rows, colLabels=header,
    cellLoc='center', loc='center',
    colWidths=[0.55, 0.25, 0.20]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.7)

# Стилизация
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#cccccc')
    if row == 0:
        cell.set_facecolor('#34495e')
        cell.set_text_props(color='white', fontweight='bold')
    elif row == len(rows):
        cell.set_facecolor('#ecf0f1')
        cell.set_text_props(fontweight='bold')
    else:
        # Цветной маркер в первой колонке
        if col == 0:
            cell.set_facecolor(colors[row-1])
            cell.set_alpha(0.25)

ax2.set_title('Расшифровка потоков', fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('balance_wheel_clean.png', dpi=200, bbox_inches='tight')
plt.show()