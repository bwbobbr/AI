import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
# print(iris_X[:2,])
# print(iris_y)

X_train,X_test,y_train,y_test = train_test_split(iris_X,iris_y,test_size=0.3)
# print(y_train)

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)            # 用训练集来拟合出knn预测函数
print(f"拟合概率{np.mean(y_test==knn.predict(X_test))*100}%")
