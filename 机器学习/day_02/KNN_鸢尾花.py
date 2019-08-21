from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
# 读取数据
iris = load_iris()

print(iris.target)
# 训练集和测试集的划分
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25)

# print(x_train)
# 特征工程
# 标准化
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)
# print(x_train)

# 使用KNN算法进行分类

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)

# 得出预测率
predict = knn.predict(x_test)
print(f'预测的结果为:{predict}')
print(f'预测的准确率{knn.score(x_test,y_test)}')


