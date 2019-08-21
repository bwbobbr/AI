import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-3,3,50) # 生成点(-3,50)长度1
y1 = 2*x+1
y2 = x**2
#plt.figure()       # 图像1
#plt.plot(x,y1)

plt.figure(num=3, figsize=(8,5))    # 图像3,大小(8,5)
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')
# linewidth 宽度, linestyle 线样式

plt.show()
