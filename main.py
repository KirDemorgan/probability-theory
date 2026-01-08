import sys
import math
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler('statistical_report.log', encoding='utf-8', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

excel_path = r'data/data.xls'
sheet_index = 5
use_col = 0

try:
    col = pd.read_excel(excel_path, sheet_name=sheet_index, usecols=[use_col])
    series = pd.to_numeric(col.iloc[:, 0], errors='coerce').dropna().astype(float)
    data = series.tolist()
except Exception as e:
    logger.error(f"Ошибка чтения файла: {e}")
    sys.exit()

n = len(data)
logger.info("=" * 60)
logger.info(f"СТАТИСТИЧЕСКИЙ ОТЧЕТ ПО ВЫБОРКЕ (n = {n})")
logger.info("=" * 60)

# --- 7. Первичный анализ ---
sum_x = sum(data)
sum_x2 = sum(x ** 2 for x in data)
mean_x = sum_x / n
mean_x2 = sum_x2 / n
var_x = mean_x2 - (mean_x ** 2)
std_x = math.sqrt(var_x)

sorted_data = sorted(data)
if n % 2 == 1:
    median = sorted_data[n // 2]
else:
    median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2

m3 = sum((x - mean_x) ** 3 for x in data) / n
m4 = sum((x - mean_x) ** 4 for x in data) / n
As = m3 / (std_x ** 3) if std_x != 0 else 0
Ex = (m4 / (std_x ** 4) - 3) if std_x != 0 else 0

logger.info("\n[ЗАДАНИЕ 7] ВЫБОРОЧНЫЕ ХАРАКТЕРИСТИКИ")
logger.info(f"1. Сумма X = {sum_x:.4f}")
logger.info(f"2. Среднее (X_cp) = ΣX / n = {sum_x:.4f} / {n} = {mean_x:.4f}")
logger.info(f"3. Сумма X^2 = {sum_x2:.4f}")
logger.info(f"4. Среднее квадратов (X^2_cp) = ΣX^2 / n = {mean_x2:.4f}")
logger.info(f"5. Дисперсия (S^2) = X^2_cp - (X_cp)^2 = {mean_x2:.4f} - ({mean_x:.4f})^2 = {var_x:.4f}")
logger.info(f"6. Ср.кв.отклонение (S) = √S^2 = {std_x:.4f}")
logger.info(f"7. Медиана (x_med): {median:.4f}")
logger.info(f"8. Асимметрия (As = m3 / S^3): {As:.4f}")
logger.info(f"9. Эксцесс (Ex = m4 / S^4 - 3): {Ex:.4f}")
logger.info(f"10. ОМП для F4 (theta = max(X)): {max(data):.4f}")

theta = max(data)
k_sturges = int(math.log2(n)) + 1
logger.info(f"\n[ГРАФИК] Параметры группировки: k = [log2({n})] + 1 = {k_sturges} интервалов")

try:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=k_sturges, density=True, edgecolor='black', color='skyblue', alpha=0.7,
            label='Гистограмма (плотность)')
    x_vals = np.linspace(0, theta, 500)
    f_x = (2 * x_vals) / (theta ** 2)
    ax.plot(x_vals, f_x, 'r-', lw=3, label=fr'Плотность F4 ($\hat{{\theta}}={theta:.4f}$)')
    ax.set_title('Гистограмма и теоретическая плотность $F_4$')
    ax.legend()
    fig.savefig('histogram.png', dpi=150)
    plt.close(fig)
    logger.info("-> График сохранен в `histogram.png`")
except Exception as e:
    logger.error(f"Ошибка построения графика: {e}")
1
# --- 8. Критерий серий ---
signs = []
for x in data:
    if x > median:
        signs.append('+')
    elif x < median:
        signs.append('-')

n1 = signs.count('+')
n2 = signs.count('-')
ks = 1
for i in range(1, len(signs)):
    if signs[i] != signs[i - 1]: ks += 1

mean_ks = (2 * n1 * n2) / (n1 + n2) + 1
var_ks = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
std_ks = math.sqrt(var_ks)
z_calc_series = (ks - mean_ks - 0.5) / std_ks if ks < mean_ks else (ks - mean_ks + 0.5) / std_ks

logger.info("\n[ЗАДАНИЕ 8] ПРОВЕРКА СЛУЧАЙНОСТИ (Критерий серий)")
logger.info(f"1. Число элементов > медианы (n1) = {n1}")
logger.info(f"2. Число элементов < медианы (n2) = {n2}")
logger.info(f"3. Количество серий (KS) = {ks}")
logger.info(f"4. M[KS] = (2*n1*n2)/(n1+n2) + 1 = {mean_ks:.4f}")
logger.info(f"5. σ[KS] = √[ (2n1n2(2n1n2-n1-n2)) / ((n1+n2)^2 * (n1+n2-1)) ] = {std_ks:.4f}")
logger.info(f"6. Z_выч = (KS - M[KS] ± 0.5) / σ[KS] = {z_calc_series:.4f}")
logger.info(f"7. Критическое Z_крит (α=0.05) = 1.96")
logger.info(
    f"ВЫВОД: {'Случайна' if abs(z_calc_series) < 1.96 else 'Не случайна'} (т.к. |Z| {'<' if abs(z_calc_series) < 1.96 else '>'} 1.96)")

# --- 10. Хи-квадрат ---
step = theta / k_sturges
chi_sq = 0
logger.info(f"\n[ЗАДАНИЕ 10] КРИТЕРИЙ ХИ-КВАДРАТ (α=0.1, Распределение F4)")
logger.info(f"Формула плотности: f(x) = 2x / {theta:.4f}^2")
logger.info(f"{'Интервал':<20} | {'n_выб':<6} | {'p_i':<8} | {'n_теор':<8} | {'(n-n_т)^2/n_т':<12}")
logger.info("-" * 75)

for i in range(k_sturges):
    left = i * step
    right = (i + 1) * step
    if i < k_sturges - 1:
        ni_obs = len([x for x in data if left <= x < right])
    else:
        ni_obs = len([x for x in data if left <= x <= right])

    p_i = (right ** 2 - left ** 2) / (theta ** 2)
    ni_theor = n * p_i
    term = ((ni_obs - ni_theor) ** 2) / ni_theor if ni_theor > 0 else 0
    chi_sq += term

    logger.info(f"[{left:5.2f}; {right:5.2f}) | {ni_obs:6d} | {p_i:8.4f} | {ni_theor:8.2f} | {term:12.4f}")

nu = k_sturges - 1 - 1
logger.info("-" * 75)
logger.info(f"1. Хи-квадрат выч = Σ[...] = {chi_sq:.4f}")
logger.info(f"2. Степени свободы ν = k - r - 1 = {k_sturges} - 1 - 1 = {nu}")
logger.info(f"3. Для α=0.1 и ν={nu} найди χ^2_крит в таблице 5")

# --- 12. Манн-Уитни ---
mid = n // 2
group1 = data[:mid]
group2 = data[mid:]
n1_mw, n2_mw = len(group1), len(group2)

u_stat = 0
for x in group1:
    for y in group2:
        if x < y:
            u_stat += 1
        elif x == y:
            u_stat += 0.5

e_u = (n1_mw * n2_mw) / 2
s_u = math.sqrt((n1_mw * n2_mw * (n1_mw + n2_mw + 1)) / 12)
z_mw = (u_stat - e_u) / s_u

logger.info("\n[ЗАДАНИЕ 12] ОДНОРОДНОСТЬ (Критерий Манна-Уитни)")
logger.info(f"1. Выборка разделена на две части: n1 = {n1_mw}, n2 = {n2_mw}")
logger.info(f"2. Статистика U (число инверсий) = {u_stat}")
logger.info(f"3. M[U] = (n1*n2)/2 = {e_u:.2f}")
logger.info(f"4. σ[U] = √[ (n1*n2*(n1+n2+1))/12 ] = {s_u:.4f}")
logger.info(f"5. Z_выч = (U - M[U]) / σ[U] = {z_mw:.4f}")
logger.info(f"6. Критическое Z_крит (α=0.01, двустороннее) = 2.57")
logger.info(
    f"ВЫВОД: {'Однородны' if abs(z_mw) < 2.57 else 'Неоднородны'} (т.к. |Z| {'<' if abs(z_mw) < 2.57 else '>'} 2.57)")