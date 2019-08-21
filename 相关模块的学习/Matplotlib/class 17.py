import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig, ax = plt.subplots()
# fig = plt.figure() 一样可以执行
x = np.arange(0, 2*np.pi, 0.01)
#line, = plt.plot(x, np.sin(x)) 一样可以执行
line, = ax.plot(x, np.sin(x))       # ax就是画布==plt
# 返回的列表故有个','

def animate(i):
    line.set_ydata(np.sin(x + i/50.0))  # update the data
    # 更新y数据
    return line,

def init():
    line.set_ydata(np.sin(x))
    return line,

ani = animation.FuncAnimation(fig=fig, func=animate, frames=100, init_func=init,
                              interval=20, blit=False)
# fig:画布,func:更新数据,frames:总长度帧数,init_func:最开始的画面,interval:刷新频率
# blit:是不是更新整张图()画布--更换为False,不更换True
plt.show()