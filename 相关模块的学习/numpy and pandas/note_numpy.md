# numpy 相当于列表功能
## Class1--import numpy as np   (例子:text1)
- np.array  矩阵转化为数组形式
- array.shape  几行几列
    - c.shape[0]:表示矩阵的行数; c.shape[1]:表示矩阵的列数
- array.ndim  数组的维数
- array.size  总元素个数

## Class2--np.array(XXXX)
- np.array([数据], dtype=np.int64(float64))
- np.zeros((row, column))                                       # 生成几行几列的0数组
- np.arange(起始值, 终止值, 步长) <==> number(起始值....)      # 生成的是一维数组
- np.arange().reshape(row,column)                             # 生成几行几列数组
- np.linspace(起始,终止,有几个)                                 # 取点，生成线段,一维数组也可以.reshape重新定义维数
    - np.linspace(-1,2,5)       # 输出的结果[-1.   -0.25  0.5   1.25  2.  ]

## Class3--数组的运算('+','-','*','/',ect)
- np.sin(array)                                             # 数组每个元素求sin
- c = a*b                                                   # 这是矩阵里面的元素逐个相乘
- np.dot(a,b) <==> a.dot(b)                                  # 这是有关矩阵运算里面的乘法
- np.random.random((row,column))                             # 随机生成几行几列的数字(0-1)
- np.sum(a) // np.sum(a, axis=1)                            # 数组里面元素的和,axis=1表示的是行求和,axis=0表示的是列求和
- np.min/max() // np.min(a, axis=0)

## Class4
- np.argmin/argmax(a)           # 最小/大值的索引(位置,用一个数来定位的,相当于一维数组)第几个，输出的是索引从0开始
- np.mean/average(a)             # 求平均值根据是axis=0后者是axis=1来确定是某行还是某列, 不写就是数组中全部求解平均值
- np.median(a)                   # 求中位数(可以行,列,添加axis=0 or 1即可)
- np.cumsum(a)                      # 累加值:第几个就是前面数字之和,个数不变
- np.diff(a)                       # 累差:列数会减一
- np.nonzero(a)                    # 输出非零数的索引(用俩个数来定位,俩个一维数组)
- np.sort(a)                       # 每行由小到大排序
- np.transpose(a)                 # 矩阵的转置,求解转置矩阵
- np.clip(a, a_min, a_max)          # 矩阵里定义的一个范围,里面的数值不变,范围之外的数值变成最小值或者是最大值

## Class5                   (例子text2)
- A[X,Y]                               # 索引某个数
- A[X,Y:Z]                          # 某行（Y：Z）位置上的数
- A.flatten()                       # 将一个数组变成一维数组（行向量）
### 合并
- np.vstack(A, B)                  # "列发生变化",将分开输入的两个一维数组进行上下合并二维数组,最多三个合并(行合并)
- np.hstack(A, B))                    # "行发生变化",将分开输入的两个一维数组进行左右合并一维数组，类似于append
- A[:,np.newaxis]                   # A.shape -->(N,1) or (1,N) 可以将一维数组进行转置
- A[np.newaxis,:]                   # 左行右列，增加一个维度，使得一维数组可以转置（正常情况下，一位数组是不能进行转置）
- np.concatenate((A,A,A,B,B),axis=0)        # 多个array的合并  axis定义在哪个维度进行合并(=0)纵向合并,(=1)是横向合并
### 分割
- np.split(A,2,axis=1)             # 竖着切 "|" 横向分割分成两块,等量分割,
- np.array_split(A,3,axis=1)        # 不等量分割
- np.vsplit(A,3)                    # 等价于(np.split(A,3,axis=0))竖着分割,横着切 "一"
- np,hsplit(A,2)                    # 同理与上面的那个
    - tips: axis=1等价于h(split),行上的数值发生了变化,左右变化
    - tips: axis=0等价于v(split),列上的数值发生了变化,上下变化
###赋值
- 传值不传址
- b = a.copy()  # 类似其他模块是地址上的传入

## 随机列表的产生
- 1.np.random.randint(X1,X2,size=(3,3))         # 随机出来的整数
- 1.np.random.uniform(X1,X2,size=(3,3))         # 随机出来的实数, 带有小数