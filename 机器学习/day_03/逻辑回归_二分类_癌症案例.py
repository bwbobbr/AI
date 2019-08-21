from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import  numpy as np

'''
数据描述
（1）699条样本，共11列数据，第一列用语检索的id，后9列分别是与肿瘤
相关的医学特征，最后一列表示肿瘤类型的数值。
（2）包含16个缺失值，用”?”标出。

'''

def logistics():
    '''
    逻辑回归分析癌症预测
    :return:None
    '''
    # 构造列标签名字
    column = ['Sample code number','Clump Thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion', 'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

    # 读取数据
    # 由于数据中初始行中,没有标签,故认为第一行为数据的标签,因此需要构造列标签名
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=column)

    # print(data)

    # 缺失值进行处理
    data = data.replace(to_replace='?',value=np.nan)
    data = data.dropna()        # 滤除缺失特征值的样本
    # x['age'].fillna(x['age'].mean(),inplace=True)  将特征缺失值用平均值补上

    # print(data)
    # 进行数据的分割
    x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]], test_size=0.25)


    # 进行标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)


    # 算法优化,逻辑回归
    logist = LogisticRegression(C=1.0)
    logist.fit(x_train,y_train)

    # 预测结果
    y_predict = logist.predict(x_test)
    print(f'采用逻辑回归预测的结果为:{y_predict}')
    print(f'采用逻辑回归的预测准确率为:{logist.score(x_test,y_test)}')
    # 采用召回率分析
    print(f'采用逻辑回归,癌症预测的召回率为:{classification_report(y_test,y_predict)}')


    return

if __name__ == '__main__':
    logistics()