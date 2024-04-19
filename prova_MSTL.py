import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import MSTL

# Caricare i dati
data = pd.read_csv("D:/VISUAL STUDIO/GITHUB_CODICE_OGGETTI/data/Dataset/electricity_consumption.csv")
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Assicurarsi che i dati siano ordinati cronologicamente
data.sort_index(inplace=True)

data_size = 7000
target = 'electricity_consumption'
electricity = data[target][:data_size]

mstl = MSTL(electricity, periods=[24, 24 * 7, 24 * 7 * 4])
res = mstl.fit()
ax = res.plot()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=2, figsize=[10,10])
res.seasonal["seasonal_24"].iloc[:24*3].plot(ax=ax[0])
ax[0].set_ylabel(target)
ax[0].set_title("Daily seasonality")

res.seasonal["seasonal_168"].iloc[:24*7*3].plot(ax=ax[1])
ax[1].set_ylabel(target)
ax[1].set_title("Weekly seasonality")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=2, figsize=[10,10])
res.seasonal["seasonal_168"].iloc[:24*7*3].plot(ax=ax[0])
ax[0].set_ylabel(target)
ax[0].set_title("Weekly seasonality")

res.seasonal["seasonal_672"].iloc[:24*7*4*3].plot(ax=ax[1])
ax[1].set_ylabel(target)
ax[1].set_title("Monthly seasonality")

plt.tight_layout()
plt.show()

"""
fig, ax = plt.subplots(nrows=2, figsize=[10,10])
mask = res.seasonal.index.month==5
res.seasonal[mask]["seasonal_24"].iloc[:24*3].plot(ax=ax[0])
ax[0].set_ylabel("seasonal_24")
ax[0].set_title("Daily seasonality")

res.seasonal[mask]["seasonal_168"].iloc[:24*7*3].plot(ax=ax[1])
ax[1].set_ylabel("seasonal_168")
ax[1].set_title("Weekly seasonality")

plt.tight_layout()
plt.show()
"""