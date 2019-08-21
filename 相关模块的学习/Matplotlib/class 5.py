import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-3,3,50) # 生成点(-1,50)长度1
y1 = 2*x+1
y2 = x**2

plt.figure()

plt.xlim(-1,2)      # 限制x轴范围
plt.ylim(-2,3)
plt.xlabel("I am X")        # 坐标轴的命名
plt.ylabel("I am Y")

new_ticks = np.linspace(-1,2,5)     # (-1,2)分成5个
plt.xticks(new_ticks)               # 换x轴显示的分度数值(只显示该范围的)
# 数字形式的互换
plt.yticks([-2,-1.8,-1,1.22,3,],
           [r'$really\ bad$',r'$bad$',r'$normal$','good','very good'])
# r正则表达,变字体
# 数字换文字一一对应

l1, = plt.plot(x,y2,label='up')
l2, = plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--',label='down')
# linewidth 宽度, linestyle 线样式, label 图例名字

plt.legend(handles=[l1,l2,],labels=['aaa','bbb'],loc='best')
# 图例更改名字

plt.show()
