import csv
import matplotlib.pyplot as plt

rad2d = {} #все ралиусы с ид
rad = [] #только максимальный радиус каждого ид

with open(r'G:\\experement\\result_stats\\2_wT_stat\\2_wT_3_stat.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if not row:
            continue

        penultimate = row[-1].strip()
        first = row[0].strip()
        try:
            num = float(penultimate.replace(',', ''))
            id = int(float(first)) 
            if id not in rad2d:
                rad2d[id] = []
            rad2d[id].append(num)

        except ValueError:
            continue
rad = [max(values) for values in rad2d.values()] #заполняем максимальными радиусами

if not rad:
    print("Нет данных для построения гистограммы")
    exit()
    
plt.figure(figsize=(12, 7))
counts, bins, patches = plt.hist(
    rad,
    bins=30,                   
    color='#4c72b0',            
    edgecolor='#2a4d69',        
    alpha=0.85,                 
    density=False             
)

plt.title('2_wt_1', fontsize=14)
plt.xlabel('Equivalent radius', fontsize=12)
plt.ylabel('Quantity', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.text(
    0.95, 0.95,                
    f'Total values: {len(rad)}\nMin: {min(rad):.4f}\nMax: {max(rad):.4f}',
    transform=plt.gca().transAxes,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)
for i in range(len(counts)):
    x = (bins[i] + bins[i+1]) / 2 
    plt.text(x, counts[i] + 0.1, f'{int(counts[i])}', 
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('eq_radius_histogram.png', dpi=150)
plt.show()
