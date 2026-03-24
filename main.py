import matplotlib.pyplot as plt
import numpy as np

# 1. ПОЛУЧАЕМ ДАННЫЕ
b = float(input("Введите параметр b (например, 0.2): "))
alfa = float(input("Введите параметр alfa (например, 1.4): "))

# 2. ГЕНЕРАЦИЯ ПОСЛЕДОВАТЕЛЬНОСТИ
n_points = 100000
x = np.zeros(n_points)
y = np.zeros(n_points)

for i in range(1, n_points):

    x[i] = 1 - alfa * y[i-1]**2 + b * x[i-1]
    y[i] = x[i-1]
    
    # 3. СРАЗУ ЛОВИМ СЛИШКОМ БОЛЬШИЕ ЧИСЛА
    if not np.isfinite(x[i]) or not np.isfinite(y[i]):
        print(f"⚠️  Переполнение на шаге {i}! x={x[i]}, y={y[i]}")
        x[i] = 0.0
        y[i] = 0.0

# 4. ОКОНЧАТЕЛЬНАЯ ОЧИСТКА ДАННЫХ
x_clean = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
y_clean = np.nan_to_num(y, nan=0.0, posinf=10.0, neginf=-10.0)

# 5. ДИАГНОСТИКА
# print(f"\n📊 Диагностика данных:")
# print(f"   В x есть NaN: {np.isnan(x).any()}, Inf: {np.isinf(x).any()}")
# print(f"   В y есть NaN: {np.isnan(y).any()}, Inf: {np.isinf(y).any()}")
# print(f"   X диапазон: [{x_clean.min():.2f}, {x_clean.max():.2f}]")
# print(f"   Y диапазон: [{y_clean.min():.2f}, {y_clean.max():.2f}]")

# 6. СОЗДАНИЕ ТОЧЕЧНОГО ГРАФИКА (ВОТ ГЛАВНОЕ ИЗМЕНЕНИЕ!)
fig, ax = plt.subplots(figsize=(12, 8))

# scatter вместо LineCollection - рисует отдельные точки
scatter = ax.scatter(
    x_clean,           # X координаты
    y_clean,           # Y координаты
    c=np.arange(len(x_clean)),  # Цвет по номеру точки
    cmap='viridis',    # Цветовая карта
    s=5,               # Размер точек (можно менять: 1-50)
    alpha=0.7,         # Прозрачность
    edgecolors='none'  # Без границы у точек
)

# 7. БЕЗОПАСНОЕ УСТАНОВЛЕНИЕ ГРАНИЦ
x_min, x_max = x_clean.min(), x_clean.max()
y_min, y_max = y_clean.min(), y_clean.max()

x_range = x_max - x_min
y_range = y_max - y_min
if x_range == 0: x_range = 2.0
if y_range == 0: y_range = 2.0

ax.set_xlim(x_min - 0.1*x_range, x_max + 0.1*x_range)
ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)

# 8. НАСТРОЙКА ВНЕШНЕГО ВИДА
ax.set_xlabel('Ось X', fontsize=12)
ax.set_ylabel('Ось Y', fontsize=12)
ax.set_title(f'Динамическая система: b={b}, α={alfa} (точечный график)', fontsize=14, pad=20)
ax.grid(True, alpha=0.3, linestyle='--')

# Цветовая шкала для точек
# cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
# cbar.set_label('Номер итерации', rotation=270, labelpad=15)

plt.tight_layout()

# 9. СОХРАНЕНИЕ РЕЗУЛЬТАТА
filename = f"scatter_b{b}_a{alfa}.png"
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"✅ Точечный график сохранён как '{filename}'")

plt.show()