import matplotlib.pyplot as plt
import csv
import numpy as np
import os
from collections import defaultdict

# ==================== КОНФИГУРАЦИЯ ====================
DIR_PATH = r'C:/Users/1/Documents/prog/2-8_wt_data/5_wt_data'
FILE_PREFIX = '5_wt_35_'
FILE_SUFFIX = '_stat.csv'
FILE_COUNT = 10
EXPERIMENT_DURATION = 0.446  # seconds
PIXEL_TO_MM = 0.02
RADIUS_THRESHOLD = 30  # порог для больших пузырей
# ======================================================

# Загрузка данных
frame2d = {}
rad2d = {}
freqrad = {}
all_times = []

for file_num in range(1, FILE_COUNT + 1):
    file_path = os.path.join(DIR_PATH, f'{FILE_PREFIX}{file_num}{FILE_SUFFIX}')
    
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        zero_row = next(reader)
        first_row = next(reader)
        first_time = float(first_row[1].replace(',', ''))
        
        for row in reader:
            bubble_id = row[0].strip()
            time_val = float(row[1].replace(',', ''))
            radius = float(row[-1].replace(',', ''))
            
            unique_id = f"{bubble_id}_{file_num}"
            norm_time = time_val - first_time
            
            if unique_id not in rad2d:
                rad2d[unique_id] = []
                frame2d[unique_id] = []
            
            rad2d[unique_id].append(radius)
            frame2d[unique_id].append(norm_time)
            all_times.append(norm_time)

# Анализ максимальных радиусов
for uid, radii in rad2d.items():
    max_radius = max(radii)
    max_index = radii.index(max_radius)
    max_time = frame2d[uid][max_index]
    
    parts = uid.split('_')
    bubble_id = '_'.join(parts[:-1])
    file_num = int(parts[-1])
    
    freqrad[uid] = {
        'max_radius': max_radius,
        'time': max_time,
        'file_num': file_num,
        'bubble_id': bubble_id
    }

# Группировка больших пузырей по файлам
file_bubbles = defaultdict(list)

for uid, data in freqrad.items():
    if data['max_radius'] > RADIUS_THRESHOLD:
        file_num = data['file_num']
        bubble_times = frame2d.get(uid, [])
        first_appearance = min(bubble_times)
        
        file_bubbles[file_num].append({
            'first_appearance': first_appearance,
            'max_radius': data['max_radius'],
            'bubble_id': data['bubble_id']
        })

# Сортировка и группировка по порядковым номерам
for file_num in file_bubbles:
    file_bubbles[file_num].sort(key=lambda x: x['first_appearance'])

order_stats = defaultdict(list)
max_orders = max(len(bubbles) for bubbles in file_bubbles.values())

for order in range(max_orders):
    for bubbles in file_bubbles.values():
        if order < len(bubbles):
            order_stats[order].append(bubbles[order]['first_appearance'])

# Выбор файла для отображения
max_bubbles_file = max(file_bubbles, key=lambda x: len(file_bubbles[x]))
print(f"Файл с максимальным количеством пузырей: {max_bubbles_file}")

selected_file = input(f"Введите номер файла (1-{FILE_COUNT}, Enter для {max_bubbles_file}): ")
selected_file = int(selected_file) if selected_file.strip() else max_bubbles_file

# Данные для выбранного файла
selected_times = []
selected_radii = []
for data in freqrad.values():
    if data['file_num'] == selected_file:
        selected_times.append(data['time'])
        selected_radii.append(data['max_radius'])

# Визуализация
if selected_times and order_stats:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Конвертация единиц
    max_time = max(all_times)
    selected_times_seconds = [t / max_time * EXPERIMENT_DURATION for t in selected_times]
    selected_radii_mm = [r * PIXEL_TO_MM for r in selected_radii]
    
    # Настройка размеров точек
    large_radii = [r for r in selected_radii if r > RADIUS_THRESHOLD]
    min_large_radius = min(large_radii) if large_radii else 0
    max_large_radius = max(large_radii) if large_radii else 0
    
    # Подготовка параметров точек
    alpha_values = []
    colors = []
    sizes = []
    
    for radius in selected_radii:
        if radius <= RADIUS_THRESHOLD:
            alpha_values.append(0.3)
            colors.append('lightblue')
            sizes.append(40)  # маленькие пузыри
        else:
            alpha_values.append(0.8)
            colors.append('blue')
            # Размер пропорционален радиусу для больших пузырей
            if max_large_radius > min_large_radius:
                normalized = (radius - min_large_radius) / (max_large_radius - min_large_radius)
                sizes.append(60 + normalized * 140)  # от 60 до 200
            else:
                sizes.append(130)  # средний размер если все одинаковые
    
    # Верхний график
    ax1.scatter(selected_times_seconds, selected_radii_mm, 
                alpha=alpha_values, c=colors, s=sizes, edgecolors='darkblue', linewidths=0.5)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Maximum radius (mm)')
    ax1.set_title('Statistics from one experiment')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Нумерация больших пузырей
    large_bubbles = []
    for time, radius_mm, original_radius in zip(selected_times_seconds, selected_radii_mm, selected_radii):
        if original_radius > RADIUS_THRESHOLD:
            large_bubbles.append({'time': time, 'radius_mm': radius_mm})
    
    large_bubbles.sort(key=lambda x: x['time'])
    
    for order, bubble in enumerate(large_bubbles):
        ax1.annotate(f"{order+1}", (bubble['time'], bubble['radius_mm']), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=9, fontweight='bold', alpha=0.8)
    
    # Нижний график - боксплоты
    order_stats_seconds = {}
    for order, times in order_stats.items():
        order_stats_seconds[order] = [t / max_time * EXPERIMENT_DURATION for t in times]
    
    orders = sorted(order_stats_seconds.keys())
    boxplot_data = [order_stats_seconds[order] for order in orders]
    
    box_plot = ax2.boxplot(boxplot_data, positions=orders, widths=0.6, patch_artist=True)
    
    # Стилизация боксплотов
    colors = plt.cm.Set3(np.linspace(0, 1, len(orders)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(box_plot[element], color='black', linewidth=1.5)
    
    ax2.set_xlabel('Bubble sequence number')
    ax2.set_ylabel('Nucleation time (s)')
    ax2.set_title('Temporal distribution of bubble nucleation')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Отображение количества наблюдений
    for i, order in enumerate(orders):
        n = len(order_stats_seconds[order])
        ax2.text(i+1, ax2.get_ylim()[1] * 0.95, f'n={n}', 
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(r'C:\Users\1\Documents\prog\working_options\bubble_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Статистика
    print(f"\nСтатистика для файла {selected_file}:")
    print(f"Всего пузырей: {len(selected_times)}")
    print(f"Больших пузырей (> {RADIUS_THRESHOLD} px): {len(large_radii)}")
    print(f"Диапазон радиусов: {min(selected_radii_mm):.2f} - {max(selected_radii_mm):.2f} мм")
    
    for order in sorted(order_stats_seconds.keys()):
        data = order_stats_seconds[order]
        if data:
            median_val = np.median(data)
            q1, q3 = np.percentile(data, [25, 75])
            print(f"Пузырь #{order+1}: t̄={median_val:.3f} с, IQR={q1:.3f}-{q3:.3f} с")