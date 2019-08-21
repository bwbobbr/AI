import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-3,3,50) # 生成点(-1,50)长度1
y1 = 2*x+1
y2 = x**2

plt.figure()
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')
# linewidth 宽度, linestyle 线样式

plt.xlim(-1,2)      # 限制x轴范围
plt.ylim(-2,3)
plt.xlabel("I am X")        # 坐标轴的命名
plt.ylabel("I am Y")

new_ticks = np.linspace(-1,2,5)     # (-1,2)分成5个
print(new_ticks)
plt.xticks(new_ticks)               # 换x轴显示的分度数值(只显示该范围的)
# 数字形式的互换
plt.yticks([-2,-1.8,-1,1.22,3,],
           [r'$really\ bad$',r'$bad$',r'$normal$','good','very good'])
# r正则表达,变字体
# 数字换文字一一对应

# gca = get current axis
ax = plt.gca()
ax.spines['right'].set_color('none')        # 让某边的边框消失
ax.spines['top'].set_color("none")

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# x轴由bottom代替, y轴由lelf代替  ('bottom','left'为之前的边框)

ax.spines['bottom'].set_position(('data',0))   # 注意这里要两个括号,'data'依照数值(由:outward和axes)
# 挪动x,y的位置----横坐标的值就是 y轴 的-1--x轴对着y轴上的-1值
ax.spines['left'].set_position(('data',0))


plt.show()
