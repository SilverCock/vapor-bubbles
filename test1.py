import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy import stats

DIR_PATH = r'C:\Users\1\Documents\prog\jets'

all_x = []
all_y = []

folders = ['2_wt', '4_wt', '5_wt', '6_wt', '8_wt']
markers = ['o', 's', '^', 'D', 'v']

plt.figure(figsize=(10, 6))

for i, folder in enumerate(folders):
    for file in glob.glob(os.path.join(DIR_PATH, folder, "*.csv")):
        df = pd.read_csv(file)
        if df.shape[1] >= 3:
            x_data = df.iloc[:, 1]
            y_data = df.iloc[:, 2]

            plt.scatter(df.iloc[:, 1], df.iloc[:, 2], marker=markers[i], alpha=0.7, label=folder)

            all_x.extend(x_data)
            all_y.extend(y_data)    
        else:
            print(f"Пропущен файл {file}: недостаточно столбцов ({df.shape[1]})")



x_array = np.array(all_x)
y_array = np.array(all_y)

slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, y_array)
line = slope * x_array + intercept

plt.plot(x_array, line, color='blue', linewidth=1, alpha = 0.6,
            label=f'trend line (R²={r_value**2:.3f})')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
print(slope)
plt.xlabel('Radius, px')
plt.ylabel('Jet range, px')
plt.title('range(radius)')
plt.grid(True, alpha=0.3)
#plt.savefig(".\graph.pdf")
plt.show()