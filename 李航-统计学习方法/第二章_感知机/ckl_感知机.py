from sklearn.linear_model import Perceptron
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


data = np.array(df.iloc[:100, [0, 1, -1]])
# print(data)

# 前100个的数据
X, y = data[:, :-1], data[:, -1]

# 调用库来求解感知机
# 感知机中直接识别此为二分类, 不用将y=0的值进行转化

clf = Perceptron(fit_intercept=False, max_iter=1000, shuffle=False)
clf.fit(X, y)
print(clf.coef_)
print(clf.intercept_)

x_points = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_points + clf.intercept_) / clf.coef_[0][1]
plt.plot(x_points, y_)

plt.scatter(data[:50, 0], data[:50, 1], label='0')
plt.scatter(data[50:100, 0], data[50:100, 1], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()


















