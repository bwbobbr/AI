import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


'''
iris.target  数据的目标值[0 0 1 0 1...]
iris.feature_names 数据的特征名

pd.DataFrame(数据, index=list(''), columns=list(''))  默认索引为0,1,2...
'''
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(df.head(2))
df['label'] = iris.target   # 数据中添加一列目标值
# print(df)


df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
label_counts = df.label.value_counts()      # 统计目标值出现的次数
# print(label_counts)
# print(df)


# plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
# plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend()
# plt.show()

# print(df)
# print('---')
data = np.array(df.iloc[:100, [0, 1, -1]])
print(data)
# print('---')
# data2 = df.data       iris.data 数据
# print(data2)

# 前100个的数据
X, y = data[:, :-1], data[:, -1]

# X,y等同于X1,y1....
X1,y1 = iris.data[:100,:2], iris.target[:100]

# 占比
ratio = len(np.where(np.array([1 if i==1 else 0 for i in y])!=0)[0]) / len(y)
# print(ratio)

# print(X,y)

class perceptron_model:

    def __init__(self):
        self.w = np.ones(len(data[0])-1, dtype=np.float32)
        self.b = 0.0
        self.alpha = 0.1

    def function(self, x):
        return np.dot(x, self.w) + self.b

    # 随机梯度下降
    def fit(self,X_trian, y_train):
        parameter = False

        # 感知机中二分类, y=-1, y=1
        for i in np.where(y_train==0):
            y_train[i] = -1
        while not parameter:
            count = 0
            for i in range(len(X_trian)):
                if y_train[i] * self.function(X_trian[i]) <= 0:
                    self.w += self.alpha*np.dot(y_train[i], X_trian[i])
                    self.b += self.alpha*y_train[i]
                    count += 1
            if count == 0:
                parameter = True
        return 'Perceptron Model'

per = perceptron_model()
per.fit(X, y)
x_points = np.linspace(4,7,10)
y_ = - (per.w[0] * x_points + per.b) / per.w[1]
plt.plot(x_points, y_)

plt.scatter(data[:50, 0], data[:50, 1], label='0')
plt.scatter(data[50:100, 0], data[50:100, 1], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

















