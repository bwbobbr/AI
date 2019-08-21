import numpy as np
# Class one
array = np.array([[1,2,3],
                  [4,5,6]], dtype=int)   # 前面的array只是一个名字表示一个数组
print(array)
print("ndim:",array.ndim)       # 数组中的维数
print("size:",array.size)       # 数组中的元素个数
print("shape:",array.shape)     # 数组是几列几行，横着的[1,2,3]是列,[1,4]是行

# Class two
print("--"*20)
a = np.zeros((3,5))
print(a)
a = np.arange(2,10,2).reshape(2,2)  # 考虑数字
print(a)

# Class three
print("--"*20)
a = np.array([[10,20,30],
              [40,50,60],
              [70,80,90]])
b = np.arange(1,10).reshape(3,3)
print(a)
print(b)
print("*"*10)
c = a - b       # 数组直接运算
print(c)
c = a * b       # 矩阵里面各个元素的乘积
print(c)
c = a.dot(b)    # 矩阵的乘法
print(c)
print("*"*10)
print(np.sum(a, axis=1))     # 按行求和
print(np.max(a, axis=0))     # 每列的最大值

# Class four
print("\nClass Four\n"+"--"*20)
array1 = np.arange(1,16).reshape(3,5)
print(array1)
print(np.argmax(array1))    # 求最大值所在的位置
print(np.average(array1,axis=0))    # 求平均值
print(np.median(array1,axis=1))            # 求数组中的中位数
print(np.cumsum(array1))            # 累加,某位置就是某位置的前面全部数值的和

print("*"*10)
print(np.diff(array1, axis=0))  # 差值:表示列与列之间个数之间的差值(上下比值),比完之后列数减1
print(np.diff(array1, axis=1))  # 行比较，左右比值
                                  #      若axis = 1行与行之间的差值,比完之后行数减1
print(np.nonzero(array1))       # 输出的是非0位置坐标每个list中对应相取就是位置坐标
print("*"*10)
array2 = np.arange(15,0,-1).reshape(3,5)
print(array2)
print(np.sort(array2))          # 每行按大小排序
print("*"*10)
print(np.transpose(array2))     # 求矩阵的转置
print("*"*10)
print(np.clip(array2,4,9))      # 在矩阵中，4-9的数值不变，大于9的值变为9，小于4的值变为4