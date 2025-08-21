import matplotlib.pyplot as plt
import csv
import os
from matplotlib.patches import ConnectionPatch

frame2d = {} # Тут времена по индексам
rad2d = {} # словарь индексов и всех радиусов
freqrad = {} # нужные времена при ид
ind = [] # все индексы

# Чтение данных о пузырях
with open(r'C:\Users\1\Documents\prog\5_wt_data/5_wt_35_10_stat.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if not row:
            continue
        penultimate = row[-1].strip()
        first = row[0].strip()
        second = row[1].strip()
        try:
            num = float(penultimate.replace(',', ''))
            id = int(float(first)) 
            sec = float(second.replace(',',''))
            ind.append(id)
            if id not in rad2d:
                rad2d[id] = []
            if id not in frame2d:
                frame2d[id] = []
            rad2d[id].append(num)
            frame2d[id].append(sec)  # Сохраняем время как число

        except ValueError:
            print(f"Пропущено нечисловое значение: {penultimate}") 

freqrad = {}

for id in rad2d.keys():
    if id in frame2d and id in rad2d:
        maxrad = max(rad2d[id])
        maxind = rad2d[id].index(maxrad)  # смотрим индекс максимального радиуса на данном индексе
        maxframe = frame2d[id][maxind]    # смотрим фрейм, когда пузырь достиг радиуса    
        if maxrad > 35:
            freqrad[id] = {
                'max_radius': maxrad,  # на каждом ид есть максрадиус и фрейм
                'time': maxframe
            }
        else:
            continue

# Чтение данных о струях из папки jets
jet_times = []
jet_ranges = []
jet_radii = []

jets_path = r'C:\Users\1\Documents\prog\jets\jets_5wt_10.csv'  # Предполагаемый путь к файлу со струями

try:
    with open(jets_path, 'r') as jet_file:
        jet_reader = csv.reader(jet_file)
        
        # Пропускаем заголовок
        header = next(jet_reader)
        print(f"Заголовок CSV: {header}")
        
        for row in jet_reader:
            if not row or len(row) < 3:
                print(f"Пропущена пустая или неполная строка: {row}")
                continue
            try:
                # Убираем возможные пробелы и преобразуем в числа
                frame = float(row[0].strip().replace(',', '.'))
                radius = float(row[1].strip().replace(',', '.'))
                jet_range = float(row[2].strip().replace(',', '.'))
                
                jet_times.append(frame)
                jet_ranges.append(jet_range)
                jet_radii.append(radius)
                print(f"Добавлена струя: frame={frame}, range={jet_range}, radius={radius}")
            except ValueError as e:
                print(f"Ошибка преобразования в строке {row}: {e}")
                
except FileNotFoundError:
    print(f"Файл струй не найден: {jets_path}")
except Exception as e:
    print(f"Ошибка при чтении файла струй: {e}")

print(f"Прочитано {len(jet_times)} записей о струях")
print(f"Обработано {len(freqrad)} записей о пузырях")

# Создаем фигуру с двумя подграфиками
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# Верхний график: пузыри
x_bubbles = [v['time'] for v in freqrad.values()]
y_bubbles = [v['max_radius'] for v in freqrad.values()]

bubble_scatter = ax1.scatter(x_bubbles, y_bubbles, alpha=0.7, color='blue', s=80)
#ax1.set_xlabel('Time (Frame)', fontsize=12)
ax1.set_ylabel('Max Radius', fontsize=12)
ax1.set_title('Bubble Max Radius Over Time', fontsize=14)
ax1.grid(True, alpha=0.3)

# Добавляем подписи к точкам на верхнем графике (пузыри) в стиле струй
for i, (x, y) in enumerate(zip(x_bubbles, y_bubbles)):
    ax1.text(
        x, y + 2,  # Смещение по Y
        f"Frame: {x:.0f}\nRadius: {y:.1f}", 
        fontsize=8,
        ha='center',
        va='bottom',
        alpha=0.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="blue", alpha=0.7)
    )

# Нижний график: струи
if jet_times and jet_ranges:
    jet_scatter = ax2.scatter(jet_times, jet_ranges, alpha=0.7, color='red', s=80)
    ax2.set_xlabel('Time (Frame)', fontsize=12)
    ax2.set_ylabel('Jet Range', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Добавляем подписи к точкам на нижнем графике (струи)
    for i, (x, y, r) in enumerate(zip(jet_times, jet_ranges, jet_radii)):
        ax2.text(
            x, y + 5,  # Смещение по Y
            f"Frame: {x:.0f}\nRange: {y:.1f}\nRadius: {r:.1f}", 
            fontsize=8,
            ha='center',
            va='bottom',
            alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.7)
        )
else:
    ax2.text(0.5, 0.5, 'No jet data available', 
             horizontalalignment='center', verticalalignment='center',
             transform=ax2.transAxes, fontsize=14)
    ax2.set_title('Jet Range Over Time (No Data)')

# Соединяем соответствующие пузыри и струи с одинаковыми фреймами
connections = 0
for bubble_time, bubble_radius in zip(x_bubbles, y_bubbles):
    # Ищем струю с таким же фреймом
    for jet_time, jet_range in zip(jet_times, jet_ranges):
        if abs(bubble_time - jet_time) < 0.1:  # Допуск для сравнения float значений
            # Создаем ConnectionPatch для соединения точек
            con = ConnectionPatch(
                xyA=(bubble_time, bubble_radius), 
                xyB=(jet_time, jet_range),
                coordsA='data', 
                coordsB='data',
                axesA=ax1, 
                axesB=ax2,
                color='green', 
                linestyle='--', 
                alpha=0.7,
                linewidth=1.5
            )
            ax2.add_artist(con)
            connections += 1
            break

print(f"Соединено {connections} пар пузырь-струя")

# Выравниваем оси X обоих графиков
if jet_times and x_bubbles:
    x_min = min(min(x_bubbles), min(jet_times))
    x_max = max(max(x_bubbles), max(jet_times))
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    
    # Добавляем вертикальные линии для лучшего сопоставления по времени
    for ax in [ax1, ax2]:
        ax.axvline(x=x_min, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=x_max, color='gray', linestyle=':', alpha=0.5)

# Добавляем легенду
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Bubbles'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Jets'),
    Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Bubble-Jet Connection')
]
ax1.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()