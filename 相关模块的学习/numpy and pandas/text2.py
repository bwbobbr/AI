# Class five
import numpy as np
print("\npart three:分索引\n"+"--"*20)
A1 = np.arange(3,15).reshape(3,4)
print(A1)
print(A1[2])    # 输出第二行
print(A1[1,1:3])   # 输出第一行1-3（1,3）位置上的数
print(A1[1,1])      # 等价于[1][1]某位置上的数
print("--"*20)
print(A1.flatten())
for item in A1.flat:        # 输出每个值
    print(item)

print("\npart three:合并\n"+"--"*20)

A = np.array([1,1,1])
B = np.array([2,2,2])

C = np.vstack((A, B))
D = np.hstack((A, B))
print(np.transpose(A))      # 无法打印这个A的转置
print(A.shape,C.shape)      # 由输出可以看出A没有维度（3，）无列数
print(type(A),type(C))
print(C,"\n", D)
print("**"*20)
A = A[:,np.newaxis]         # 在A的列上加了一个维度，从而使A的转置存在
B = B[:,np.newaxis]
C1 = np.vstack((A, B))      # 上下合并
D1 = np.hstack((A, B))      # 左右合并
print(C1.shape)
print(C1,"\n--\n", D1)
E = np.concatenate((A,A,A,B),axis=1)
print(E)

print("\npart three:分割\n"+"--"*20)
A2 = np.arange(12).reshape(3,4)
print(A2)
print(np.split(A2,2,axis=1))
print(np.split(A2,3,axis=0))
print("-"*40)
print(np.vsplit(A2,3))      # 等价于axis=0
print(np.hsplit(A2,2))
print("非等量分割")
print(np.array_split(A2,2,axis=0))

print("\npart three:赋值\n"+"--"*20)
A3 = np.arange(4)
print(A3)
A4 = A3.copy()      # 传值不传址
A3[0] = 666
print(A3,"\n",A4)