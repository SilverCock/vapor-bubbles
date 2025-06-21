import matplotlib.pyplot as plt
import csv
import numpy as np

area = []

with open(r'C:/Users/1/Documents/prog/2_wT_1_stat.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if not row:
            continue

        penultimate = row[4].strip()
        try:
            num = float(penultimate)
            area.append(num)
        except ValueError:
            print(f"Пропущено нечисловое значение: {penultimate}")
if not area:
    print("Нет данных для построения гистограммы")
    exit()

plt.figure(figsize=(12, 7))
plt.hist(
    area,
    bins=50,                   
    color='#4c72b0',            
    edgecolor='#2a4d69',        
    alpha=0.85,                 
    density=False             
)

plt.title('Area', fontsize=14)
plt.xlabel('Area', fontsize=12)
plt.ylabel('Quantity', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.text(
    0.95, 0.95,                 
    f'Всего значений: {len(area)}\nMin: {min(area):.4f}\nMax: {max(area):.4f}',
    transform=plt.gca().transAxes,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

plt.tight_layout()
plt.savefig('Area_histogram.png', dpi=150)
plt.show()



