import matplotlib.pyplot as plt
import csv
import numpy as np
import os
from collections import defaultdict

# Конфигурация
DIR_PATH = r'C:/Users/1/Documents/prog/2_wt_data/'  # Директория с файлами
FILE_PREFIX = '2_wt_'                              # Префикс имени файла
FILE_SUFFIX = '_stat.csv'                          # Суффикс имени файла
FILE_COUNT = 10                                     # Количество файлов

frame2d = {}  # Словарь времен по ID (нормированные)
rad2d = {}    # Словарь радиусов по ID
freqrad = {}  # Словарь максимальных радиусов и их времен
all_times = []  # Все временные метки

# Загрузка данных для всех файлов
for file_num in range(1, FILE_COUNT + 1):
    file_path = os.path.join(DIR_PATH, f'{FILE_PREFIX}{file_num}{FILE_SUFFIX}')
    
    try:
        with open(file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            try:
                # Читаем и обрабатываем первую строку
                zero_row = next(reader)
                first_row = next(reader)
                first_time = float(first_row[1].replace(',', ''))
            except StopIteration:
                print(f"Файл {file_path} пуст!")
                continue
                
            for row in reader:
                if not row or len(row) < 3:
                    continue
                    
                try:
                    # Обработка данных
                    bubble_id = row[0].strip()
                    time_val = float(row[1].replace(',', ''))
                    radius = float(row[-1].replace(',', ''))
                    
                    # Создаем уникальный ID
                    unique_id = f"{bubble_id}_{file_num}"
                    
                    # Нормировка времени
                    norm_time = time_val - first_time
                    
                    # Сохраняем данные
                    if unique_id not in rad2d:
                        rad2d[unique_id] = []
                        frame2d[unique_id] = []
                    
                    rad2d[unique_id].append(radius)
                    frame2d[unique_id].append(norm_time)
                    all_times.append(norm_time)
                    
                except ValueError as e:
                    print(f"Ошибка преобразования в файле {file_path}: {e}")
    
    except FileNotFoundError:
        print(f"Файл {file_path} не найден")
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {str(e)}")

# Собираем данные о максимальных радиусах
for uid, radii in rad2d.items():
    if not radii:
        continue
        
    max_radius = max(radii)
    max_index = radii.index(max_radius)
    
    if uid in frame2d and len(frame2d[uid]) > max_index:
        max_time = frame2d[uid][max_index]
        
        if max_radius > 30:
            freqrad[uid] = {
                'max_radius': max_radius,
                'time': max_time,
                'file_num': int(uid.split('_')[-1])  # Сохраняем номер файла
            }

# Собираем времена первого появления пузырей (для каждого файла)
file_first_appearance = defaultdict(list)

for uid, times in frame2d.items():
    if not times:
        continue
        
    # Извлекаем номер файла из uid (формат: "bubbleId_fileNum")
    parts = uid.split('_')
    if len(parts) < 2:
        continue
    file_num = int(parts[-1])
    
    # Время первого появления - минимальное время в списке
    first_appearance = min(times)
    
    # Убеждаемся, что пузырь соответствует критерию (max_radius > 30)
    if uid in freqrad:
        file_first_appearance[file_num].append(first_appearance)

# Сортируем времена появления в каждом файле
for file_num in file_first_appearance:
    file_first_appearance[file_num].sort()

# Группируем времена по порядковому номеру пузыря
order_stats = defaultdict(list)
for times in file_first_appearance.values():
    for order, time_val in enumerate(times):
        order_stats[order].append(time_val)

# Рассчитываем среднее и стандартное отклонение
order_means = {}
order_stds = {}
for order, times in order_stats.items():
    if times:
        order_means[order] = np.mean(times)
        order_stds[order] = np.std(times, ddof=1)

# Найдем файл с максимальным количеством пузырей
max_bubbles = 0
max_bubbles_file = 1
for file_num in range(1, FILE_COUNT + 1):
    bubble_count = sum(1 for data in freqrad.values() if data['file_num'] == file_num)
    if bubble_count > max_bubbles:
        max_bubbles = bubble_count
        max_bubbles_file = file_num

print(f"Файл с максимальным количеством пузырей: {max_bubbles_file} (пузырей: {max_bubbles})")

# Запрос номера файла у пользователя
while True:
    try:
        user_input = input(f"Введите номер файла для отображения (1-{FILE_COUNT}, Enter для файла {max_bubbles_file}): ")
        if user_input.strip() == "":
            selected_file = max_bubbles_file
            break
        selected_file = int(user_input)
        if 1 <= selected_file <= FILE_COUNT:
            break
        else:
            print(f"Ошибка: номер должен быть от 1 до {FILE_COUNT}")
    except ValueError:
        print("Ошибка: введите целое число")

# Фильтрация данных для выбранного файла
selected_times = []
selected_radii = []
for uid, data in freqrad.items():
    if data['file_num'] == selected_file:
        selected_times.append(data['time'])
        selected_radii.append(data['max_radius'])

# Визуализация
if selected_times:
    plt.figure(figsize=(18, 12))
    
    # Отображаем пузыри только для выбранного файла
    plt.scatter(selected_times, selected_radii, alpha=0.6, color='blue', 
                label=f'Пузыри из файла {selected_file}')
    
    # Определяем пределы осей для позиционирования подписей
    min_radius = min(selected_radii) if selected_radii else 0
    max_radius_val = max(selected_radii) if selected_radii else 100
    max_time_val = max(selected_times) if selected_times else 100
    
    # Рисуем вертикальные линии с усами и номерами
    y_pos = max_radius_val * 1.05  # Начальная позиция по Y для номеров
    y_step = max_radius_val * 0.05  # Шаг для номеров
    
    # Позиции для подписей снизу (время среднего)
    bottom_y = min_radius - (max_radius_val - min_radius) * 0.15
    
    # Отображаем все средние значения
    for order, mean_val in sorted(order_means.items()):
        std_val = order_stds[order]
        
        # Рисуем вертикальную линию среднего
        plt.axvline(
            x=mean_val, 
            color='red', 
            linestyle='--', 
            linewidth=1.0,
            alpha=0.4
        )
        
        # Рисуем усы (горизонтальные линии для std)
        whisker_y = y_pos + y_step * (order % 7)  # Циклическое позиционирование
        plt.hlines(
            y=whisker_y,
            xmin=mean_val - std_val,
            xmax=mean_val + std_val,
            color='black',
            linewidth=1.5,
            alpha=0.3
        )
        
        # Рисуем вертикальные черточки на концах усов
        plt.vlines(
            x=mean_val - std_val,
            ymin=whisker_y - y_step*0.1,
            ymax=whisker_y + y_step*0.1,
            color='black',
            linewidth=1.5
        )
        plt.vlines(
            x=mean_val + std_val,
            ymin=whisker_y - y_step*0.1,
            ymax=whisker_y + y_step*0.1,
            color='black',
            linewidth=1.5
        )
        
        # Рисуем точку в среднем значении
        plt.scatter(
            mean_val, 
            whisker_y, 
            color='red', 
            s=40, 
            zorder=5
        )
        
        # Добавляем номер пузыря
        plt.text(
            mean_val, 
            whisker_y + y_step*0.3, 
            f"{order+1}", 
            fontsize=9, 
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
        
        # Подпись времени среднего внизу графика
        # Чередуем позиции для предотвращения наложения
        text_offset = (order % 3) * 0.03 * (max_radius_val - min_radius)
        plt.text(
            mean_val, 
            bottom_y - text_offset, 
            f"{mean_val:.2f}", 
            fontsize=8,
            ha='center',
            va='top',
            rotation=45,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.7)
        )
        
        # Подпись отклонения над усом
        plt.text(
            mean_val, 
            whisker_y + y_step*0.2, 
            f"σ={std_val:.2f}", 
            fontsize=8,
            ha='center',
            va='bottom',
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="black", alpha=0.7)
        )
    
    # Увеличиваем верхний предел оси Y для размещения номеров
    # и нижний для размещения подписей времени
    plt.ylim(
        bottom=bottom_y - (max_radius_val - min_radius) * 0.15,
        top=y_pos + y_step * 10
    )
    
    # Увеличиваем правый предел оси X для размещения всех элементов
    max_order = max(order_means.keys()) if order_means else 0
    if max_order > 0:
        last_mean = order_means[max_order]
        plt.xlim(right=last_mean * 1.15)
    
    plt.xlabel('Нормированное время', fontsize=12)
    plt.ylabel('Максимальный радиус', fontsize=12)
    plt.title(f'Зависимость максимального радиуса от времени (Файл {selected_file})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Добавляем легенду для элементов
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, 
               label=f'Пузыри (Файл {selected_file})'),
        Line2D([0], [0], color='red', marker='o', linestyle='', markersize=8, label='Среднее время появления'),
        Line2D([0], [0], color='black', lw=1.5, label='Стандартное отклонение')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'freq_file_{selected_file}.png', dpi=300)
    plt.show()
    
    print(f"Отображено пузырей: {len(selected_times)}")
    print(f"Отображено средних значений: {len(order_means)}")
else:
    print(f"Для файла {selected_file} нет данных о пузырях с радиусом >30")