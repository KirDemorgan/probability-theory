import sys
import math
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

excel_path = r'data/data.xls'
sheet_index = 5
use_col = 0

try:
    logger.info(f"Attempting to read Excel file `data/data.xls`, sheet index {sheet_index}, column {use_col}")
    col = pd.read_excel(excel_path, sheet_name=sheet_index, usecols=[use_col])
    series = pd.to_numeric(col.iloc[:, 0], errors='coerce').dropna().astype(float)
    data = series.tolist()
except Exception:
    logger.exception(f"Failed to read Excel file `data/data.xls` (sheet {sheet_index + 1}, column {use_col + 1})")
    raise

if not data:
    logger.error(f"No valid numeric data found in `data/data.xls` (sheet {sheet_index + 1}, column {use_col + 1})")
    raise ValueError("No numeric data loaded")

n = len(data)
logger.info(f"Loaded {n} records from `data/data.xls` (sheet {sheet_index + 1}, column {use_col + 1}). "
            f"min={min(data):.4f}, max={max(data):.4f}, sample={data[:5]}")

# --- 7. Первичный анализ ---
mean_x = sum(data) / n
mean_x2 = sum(x**2 for x in data) / n
var_x = mean_x2 - (mean_x**2)
std_x = math.sqrt(var_x)

sorted_data = sorted(data)

if n % 2 == 1:
    median = sorted_data[n//2]
else:
    median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2

m3 = sum((x - mean_x)**3 for x in data) / n
m4 = sum((x - mean_x)**4 for x in data) / n
As = m3 / (std_x**3) if std_x != 0 else float('nan')
Ex = (m4 / (std_x**4) - 3) if std_x != 0 else float('nan')

logger.info("\n--- 7. Характеристики ---")
logger.info(f"Среднее: {mean_x:.4f}")
logger.info(f"Дисперсия: {var_x:.4f}")
logger.info(f"Медиана: {median:.4f}")
logger.info(f"Асимметрия: {As:.4f}")
logger.info(f"Эксцесс: {Ex:.4f}")
logger.info(f"Максимум (ОМП для F4): {max(data)}")

# --- 8. Критерий серий (случайность) ---
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
    if signs[i] != signs[i-1]:
        ks += 1

try:
    denom = math.sqrt((2*n1*n2*(2*n1*n2 - n1 - n2))/((n1+n2)**2 * (n1+n2-1)))
    z_calc_series = ((ks - (2*n1*n2)/(n1+n2) - 1) - 0.5) / denom
except Exception:
    logger.exception("Error computing series test statistic (possible division by zero).")
    raise

logger.info("\n--- 8. Случайность ---")
logger.info(f"n1 (+): {n1}, n2 (-): {n2}, Серий (KS): {ks}")
logger.info(f"Z_выч: {z_calc_series:.4f} (сравнить с Z_крит=1.96)")

# --- 10. Хи-квадрат (вид распределения F4) ---
k = int(math.log2(n)) + 1
theta = max(data)
step = theta / k
chi_sq = 0
logger.info("\n--- 10. Хи-квадрат (Интервалы) ---")

for i in range(k):
    left = i * step
    right = (i + 1) * step
    if i < k - 1:
        ni_obs = len([x for x in data if left <= x < right])
    else:
        ni_obs = len([x for x in data if left <= x <= right])
    pi = (right**2 - left**2) / (theta**2) if theta != 0 else 0
    ni_theor = n * pi
    if ni_theor > 0:
        chi_sq += ((ni_obs - ni_theor)**2) / ni_theor
    logger.info(f"Интервал {i+1} [{left:.2f}-{right:.2f}]: n_выб={ni_obs}, n_теор={ni_theor:.2f}")

logger.info(f"Хи-квадрат выч: {chi_sq:.4f}")

# --- 12. Манн-Уитни (однородность половин) ---
group1 = data[:n//2]
group2 = data[n//2:]
n_mw = len(group1)

u_stat = 0
for x in group1:
    for y in group2:
        if x < y:
            u_stat += 1
        elif x == y:
            u_stat += 0.5

try:
    z_mw = (u_stat - (n_mw**2)/2) / math.sqrt((n_mw**2 * (2*n_mw + 1))/12)
except Exception:
    logger.exception("Error computing Mann-Whitney Z (possible division by zero).")
    raise

logger.info("\n--- 12. Манн-Уитни ---")
logger.info(f"U-статистика: {u_stat}")
logger.info(f"Z_выч: {z_mw:.4f} (сравнить с Z_крит=2.57)")
