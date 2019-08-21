import matplotlib.pyplot as plt
import numpy as np

# 面向对象创建画板
fig = plt.figure()
ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)
# ax1.plot([0,1],[2,3],c='black')
# ax1.set_xlim([0,0.1])
# ax2.plot([1,3],[1,4],c='black')
x = np.linspace(0,10,100)
y = [x*2 for x in x]
ax1.set_xlim(2,5)
ax1.set_title("xiao")
# ax1.set_xlable("123")
ax1.plot(x,y)


plt.show()