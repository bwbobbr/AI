import matplotlib.pyplot as plt
import numpy as np
n = 1024
X = np.random.normal(0,1,n)
Y = np.random.normal(0,1,n)
T = np.arctan2(X,Y)

plt.scatter(X,Y,s=75,c=T,alpha=0.5)
# 绘制散点图
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
# 限制大小

plt.xticks(())
plt.yticks(())
# 隐藏x,y边间的数值
plt.show()