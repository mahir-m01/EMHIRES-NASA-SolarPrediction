import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('EMHIRESPV_TSh_CF_Country_19862015.csv')


# Extracting the first 24 hours for a single country (ES - Spain for example)
one_day = df['ES'].iloc[0:24]

plt.figure(figsize=(10, 6))
plt.plot(one_day, marker='o', color='gold', linewidth=2, label='Solar Efficiency')
plt.fill_between(range(24), one_day, color='yellow', alpha=0.3)

plt.annotate('Night: No Sun', xy=(2, 0), xytext=(2, 0.1), arrowprops=dict(arrowstyle='->'))
plt.annotate('Sunrise: Production Starts', xy=(7, 0.05), xytext=(1, 0.3), arrowprops=dict(arrowstyle='->'))
plt.annotate('Solar Noon: Peak Power', xy=(13, one_day.max()), xytext=(15, 0.5), arrowprops=dict(arrowstyle='->'))

plt.title('A Single Day in the Life of a Solar Panel')
plt.ylabel('Efficiency (Capacity Factor 0.0 to 1.0)')
plt.xlabel('Hour of the Day (0-23)')
plt.xticks(range(0, 24))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

