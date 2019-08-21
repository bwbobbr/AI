from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge
from sklearn.metrics import mean_squared_log_error
from sklearn.externals import joblib



def liner_regression():
    '''
    线性回归说明
    :return:
    '''
    # 读取数据
    data_boston = load_boston()

    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(data_boston.data,data_boston.target,test_size=0.25)

    # 进行标准化
    #　犹豫特征与目标矩阵的维度不一样故，标准化时其应该分开计算
    x_std = StandardScaler()
    x_train = x_std.fit_transform(x_train)
    x_test = x_std.transform(x_test)

    y_std = StandardScaler()
    y_train = y_std.fit_transform(y_train.reshape(-1,1))
    y_test = y_std.transform(y_test.reshape(-1,1))

    # 算法流程
    # 线性回归
    # 使用正规化优化方式
    LR = LinearRegression()
    LR.fit(x_train,y_train)

    print(f"正规化优化输出权重系数:{LR.coef_}")

    # 预测结果
    predict_LR = y_std.inverse_transform(LR.predict(x_test))
    print(predict_LR)

    # 回归评估
    error_LR = mean_squared_log_error(y_std.inverse_transform(y_test),predict_LR)

    print(f"正规化优化回归评估:{error_LR}")

    # 使用梯度下降进行优化
    sgd = SGDRegressor()
    sgd.fit(x_train,y_train)

    print(f'采用梯度下降生成的权重系数:{sgd.coef_}')

    # 进行预测

    predict_SGD = y_std.inverse_transform(sgd.predict(x_test))
    print(f'采用梯度下降的预测值:{predict_SGD}')

    # 回归评估
    error_SGD = mean_squared_log_error(y_std.inverse_transform(y_test),predict_SGD)
    print(f'采用梯度下降的均方误差{error_SGD}')


    # 使用Ridge岭回归进行优化(线性回归+正则化(防止过拟合))
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train,y_train)

    # 进行预测
    predict_Ridge = y_std.inverse_transform(ridge.predict(x_test))
    print(f'采用岭回归的预测值:{predict_Ridge}')

    # 回归评估
    error_ridge = mean_squared_log_error(y_std.inverse_transform(y_test),predict_Ridge)
    print(f'采用岭回归是的均方误差{error_ridge}')


def liner_regression_save_model():
    '''
    线性回归说明
    :return:
    '''
    # 读取数据
    data_boston = load_boston()

    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(data_boston.data, data_boston.target, test_size=0.25)

    # 进行标准化
    # 　犹豫特征与目标矩阵的维度不一样故，标准化时其应该分开计算
    x_std = StandardScaler()
    x_train = x_std.fit_transform(x_train)
    x_test = x_std.transform(x_test)

    y_std = StandardScaler()
    y_train = y_std.fit_transform(y_train.reshape(-1, 1))
    y_test = y_std.transform(y_test.reshape(-1, 1))

    # 算法流程
    # 线性回归

    # 使用Ridge岭回归进行优化(线性回归+正则化(防止过拟合))
    '''
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train, y_train)

    joblib.dump(ridge,'./model/Ridge_model.pkl')
    '''
    mode_ridge = joblib.load('./model/Ridge_model.pkl')
    # 进行预测
    predict_Ridge = y_std.inverse_transform(mode_ridge.predict(x_test))
    print(f'采用岭回归的预测值:{predict_Ridge}')
    # 回归评估
    error_ridge = mean_squared_log_error(y_std.inverse_transform(y_test), predict_Ridge)
    print(f'采用岭回归是的均方误差{error_ridge}')


    # # 进行预测
    # predict_Ridge = y_std.inverse_transform(ridge.predict(x_test))
    # print(f'采用岭回归的预测值:{predict_Ridge}')
    #
    # # 回归评估
    # error_ridge = mean_squared_log_error(y_std.inverse_transform(y_test), predict_Ridge)
    # print(f'采用岭回归是的均方误差{error_ridge}')


if __name__ == '__main__':
    # liner_regression()
    liner_regression_save_model()




