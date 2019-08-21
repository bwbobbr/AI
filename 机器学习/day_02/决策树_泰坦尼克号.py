'''
泰坦尼克号数据
在泰坦尼克号和titanic2数据帧描述泰坦尼克号上的个别乘客的生存状态。在泰坦尼克号的数据帧不包含从剧组信息，但它确实包含了乘客的一半的实际年龄。
关于泰坦尼克号旅客的数据的主要来源是百科全书Titanica。这里使用的数据集是由各种研究人员开始的。其中包括许多研究人员创建的旅客名单，由Michael A. Findlay编辑。
我们提取的数据集中的特征是票的类别，存活，乘坐班，年龄，登陆，home.dest，房间，票，船和性别。乘坐班是指乘客班（1，2，3），是社会经济阶层的代表。
其中age数据存在缺失。

'''
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier

def decision():
    '''
    决策树对泰坦尼克号进行预测生死
    :return:
    '''
    data_titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

    # 处理数据,找出特征值和目标值

    x = data_titan[['pclass','age','sex']]

    y = data_titan['survived']

    # 缺失值处理
    x['age'].fillna(x['age'].mean(),inplace=True)

    # print(x)
    # 分割训练集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

    print(x_train)
    # 特征工程处理 特征-->类别-->one_hot编码
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))     # 将特征中的字符串转化为字典形式
    x_test = dict.transform(x_test.to_dict(orient="records"))

    print(x_train)
    print(dict.get_feature_names())


    '''
    
    # 用决策树进行预测
    decision = DecisionTreeClassifier()
    decision.fit(x_train,y_train)

    # 预测准确率
    print(f"准确率为:{decision.score(x_test,y_test)}")

    # 导出决策树的结构
    export_graphviz(decision,out_file='decision_tree.dot',feature_names=dict.get_feature_names())
    '''

    # 随机森林进行预测(超参数调优)
    random_tree = RandomForestClassifier()

    # 网格搜索调优
    parameter = {'n_estimators':[120,200,300,500,800,1200],'max_depth':[5,8,15,25,30]}
    gc = GridSearchCV(random_tree, param_grid=parameter, cv=2)

    gc.fit(x_train,y_train)
    print(f"随机森林的准确率:{gc.score(x_test,y_test)}")
    print(f'最有的算法参数为:{gc.best_estimator_}')
    print(f'查看选择的参数模型:{gc.best_params_}')




if __name__ == '__main__':
    decision()











