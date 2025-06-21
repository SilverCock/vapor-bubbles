import csv
import matplotlib.pyplot as plt

eq_radius = []

with open(r'C:/Users/1/Documents/prog/2_wT_1_stat.csv', 'r') as file:
    reader = csv.reader(file)
    
    for row in reader:
        if not row:
            continue
            
        last = row[-1].strip() 
        
        try:

            num = float(last)
            eq_radius.append(num)
        except ValueError:
            print(f"Пропущено нечисловое значение: {last}")

if not eq_radius:
    print("Нет данных для построения гистограммы")
    exit()

plt.figure(figsize=(12, 7))
plt.hist(
    eq_radius,
    bins=50,                   
    color='#4c72b0',            
    edgecolor='#2a4d69',        
    alpha=0.85,                 
    density=False             
)

plt.title('Equivalent radius', fontsize=14)
plt.xlabel('Equivalent radius', fontsize=12)
plt.ylabel('Quantity', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.text(
    0.95, 0.95,                
    f'Всего значений: {len(eq_radius)}\nMin: {min(eq_radius):.4f}\nMax: {max(eq_radius):.4f}',
    transform=plt.gca().transAxes,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

plt.tight_layout()
plt.savefig('eq_radius_histogram.png', dpi=150)
plt.show()