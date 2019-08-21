import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
x = np.arange(-10,20,0.25)
y = np.arange(-10,20,0.25)
x,y = np.meshgrid(x,y)
fun = x**2 + y**2 + x*y
ax.plot_surface(x,y,fun)
plt.show()