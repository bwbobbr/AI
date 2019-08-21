import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
# -3到3上生成50个点
y = 2*x + 1

plt.figure(num=1, figsize=(8, 5),)
# 画布1,大小8*5
plt.plot(x, y,)
# 绘制x,y关系图像

ax = plt.gca()
# 对画布进行处理
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# 边框消失
ax.xaxis.set_ticks_position('bottom')
# x轴坐标用bottom边框表示
ax.spines['bottom'].set_position(('data', 0))
# x轴对准y轴上0值
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

x0 = 1
y0 = x0*2+1
plt.scatter([x0,],[y0,],s=50,color='b')
# 绘制x0,y0点
plt.plot([x0,x0],[y0,0],'k--',linewidth=2.5,color='gray')
# 连接两点,第一个[两个点 x坐标],[两个点 y坐标], k--表示用虚线绘制, 线宽,

# method1
##########################################
plt.annotate(r'$2x+1={0}$'.format(y0),xy=(x0,y0),xycoords='data',xytext=(+30,-30),
             textcoords='offset points',fontsize=16,arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
''' 
xy****表示选择的基点(定位点)参考于'data'数据,位置在(30,-30)下面
textcoords是关于位置的描述, arrowprops是箭头的描述
'''
# method2
###############################################
plt.text(-3.7,3,r'$This\ is\ the\ some\ text.\ \mu\ \sigma_i\ \alpha_t$',
         fontdict={'size':12,'color':'r'})


plt.show()