# @Time     : 2022/8/7
# @Author   : Haldate
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='Times New Roman')

plt.title('MUTAG', fontsize=20)

x = [1, 2, 5, 7, 10, 11, 13, 15, 17, 20]
# x = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]


gcn_y = [73.11, 69.50, 73.44, 74.00, 76.94, 80.17, 83.44, 74.89, 70.06, 69.06]
gin_y = [81.00, 75.28, 80.33, 82.89, 85.56, 86.28, 84.67, 83.50, 83.33, 81.78]
asap_y = [77.72, 73.28, 76.61, 77.67, 78.28, 78.17, 81.39, 77.39, 75.56, 74.28]

# plt.xticks(x)

plt.plot(x, gcn_y, color='red', marker='o',  linestyle='-', label='GCN')
plt.plot(x, gin_y, color='blue', marker='D', linestyle='-', label='GIN')
plt.plot(x, asap_y, color='green', marker='P', markersize=10, linestyle='-', label='ASAP')

plt.legend()

plt.xlabel("topk values", fontsize=15)
plt.ylabel("accuracy", fontsize=15)

plt.savefig('mutag.pdf')
plt.show()

