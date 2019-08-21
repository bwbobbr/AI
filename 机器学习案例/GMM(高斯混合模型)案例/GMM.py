import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("Fremont.csv", index_col='Date', parse_dates=True)
# print(data.head())      # 默认输出前5个数据

data.plot()
plt.show()