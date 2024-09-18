import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 讀取數據
data = pd.read_csv('LiDAR_guiding_analyze.csv')

# 計算DBSCAN和HDBSCAN的平均值（Ground Truth）
dbscan_gt = data[['DBSCAN-Test1', 'DBSCAN-Test2', 'DBSCAN-Test3']].mean(axis=1)
hdbscan_gt = data[['HDBSCAN-Test1', 'HDBSCAN-Test2', 'HDBSCAN-Test3']].mean(axis=1)

# 計算最大和最小誤差
# dbscan_max = data[['DBSCAN-Test1', 'DBSCAN-Test2', 'DBSCAN-Test3']].max(axis=1)
# dbscan_min = data[['DBSCAN-Test1', 'DBSCAN-Test2', 'DBSCAN-Test3']].min(axis=1)
# hdbscan_max = data[['HDBSCAN-Test1', 'HDBSCAN-Test2', 'HDBSCAN-Test3']].max(axis=1)
# hdbscan_min = data[['HDBSCAN-Test1', 'HDBSCAN-Test2', 'HDBSCAN-Test3']].min(axis=1)

# 計算每個取樣點的標準差
dbscan_std = data[['DBSCAN-Test1', 'DBSCAN-Test2', 'DBSCAN-Test3']].std(axis=1)
hdbscan_std = data[['HDBSCAN-Test1', 'HDBSCAN-Test2', 'HDBSCAN-Test3']].std(axis=1)

plt.rcParams.update({'font.size': 13, 'font.family': 'Times New Roman'})
plt.figure(figsize=(15, 6))

plt.plot(data['位置編號'], dbscan_gt, label='DBSCAN', color='tab:orange')
plt.fill_between(data['位置編號'], dbscan_gt - dbscan_std, dbscan_gt + dbscan_std, alpha=0.2, color='tab:orange')
plt.plot(data['位置編號'], hdbscan_gt, label='HDBSCAN', color='tab:green')
plt.fill_between(data['位置編號'], hdbscan_gt - hdbscan_std, hdbscan_gt + hdbscan_std, alpha=0.2, color='tab:green')


fontsize = 20
fontname = 'Times New Roman'

plt.rcParams.update({'font.size': fontsize})
plt.xlim(0, 90)
plt.ylim(-150, 150)
plt.xticks(fontsize=fontsize, fontname=fontname)
plt.yticks(fontsize=fontsize, fontname=fontname)
plt.legend(prop={'family': fontname}, loc='upper center')
plt.grid(True)
plt.savefig('DBSCAN_HDBSCAN_error_distribution.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()



# print(data[['DBSCAN-Test1', 'DBSCAN-Test2', 'DBSCAN-Test3']].std(axis=1))
# print(data[['HDBSCAN-Test1', 'HDBSCAN-Test2', 'HDBSCAN-Test3']].std(axis=1))
