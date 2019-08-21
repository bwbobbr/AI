import matplotlib.pyplot as plt

plt.figure()
plt.subplot(2,1,1)
plt.plot([0,1],[0,1])
# 第一张图片占了上面的三张图片

plt.subplot(2,3,4)
x_values = range(0,10)
y_values = [x**2+1 for x in x_values]
plt.plot(x_values,y_values)

plt.subplot(2,3,5)
plt.plot([0,1],[0,3])

plt.subplot(2,3,6)
plt.plot([0,1],[0,4])

plt.show()