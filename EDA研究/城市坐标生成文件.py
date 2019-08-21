import numpy as np
order = np.random.uniform(-25,25,size=[10,2])
np.set_printoptions(precision=3)        # 设置精度
print(order)
np.savetxt("order_axis.txt", order,fmt='%0.3f')