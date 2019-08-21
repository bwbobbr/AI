import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

circle_target = mpatches.Circle((0, 0), radius=5, edgecolor='r', fill=False)
plt.xlim(-80, 80)
plt.ylim(-80, 80)
plt.axes().add_patch(circle_target)

# plt.show()

N = 1000
u, sigma = 0, 20
points = sigma * np.random.rand(N, 2) + u
plt.scatter([x[0] for x in points], [x[1] for x in points], c=np.random.rand(N), alpha=0.5)

counts = 0
for point in points:
    if point[0] ** 2 + point[1] ** 2 < (8-5) ** 2:
        counts += 1

accuracy = counts / N
print(f'套中娃娃的命中率为{accuracy}')
