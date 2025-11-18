import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

px_to_mm = 0.02
framerate = 25000

data = pd.read_csv('./data/6_wt_data/6_wt_35_3_statistics.csv')

max_radius = data.loc[data.groupby("id")["eq_radius"].idxmax(), ['id', 'frame','eq_radius']
                      ].reset_index(drop=True)

max_radius['eq_radius'] = max_radius['eq_radius'] * px_to_mm
max_radius['second'] = max_radius['frame'] / framerate
duration = max_radius['second'].max() - max_radius['second'].min()

print(duration)

threshholds = [0.6, 0.75, 0.9]
norm_counts = pd.DataFrame(columns=['threshhold', 'norm_count'])

for threshhold in threshholds:
    large_bubbles = max_radius[max_radius['eq_radius'] > threshhold]
    count = len(large_bubbles)
    norm_count = count / duration

    norm_counts = pd.concat([pd.DataFrame([[threshhold,norm_count]], columns=norm_counts.columns), norm_counts], ignore_index=True)

print(norm_counts)
