import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation
import time
start = time.clock()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line, = ax1.plot(np.random.rand(10))

def update(data):
    line.set_ydata(data)
    return line,

def data_gen():
    while True:
        yield np.random.rand(10)

ani = animation.FuncAnimation(fig,update,data_gen,interval=2*1000,blit=False)
end = time.clock()
print("{0}".format(end-start))
plt.show()