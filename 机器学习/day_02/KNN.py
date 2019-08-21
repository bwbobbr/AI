'''
文件位置：instance_ex
文件说明
train.csv，test.csv
row_id：签入事件的id
xy：坐标
准确度：定位精度
时间：时间戳
place_id：业务的ID，这是您预测的目标
sample_submission.csv - 具有随机预测的正确格式的样本提交文件
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd

def KNN_cls():
    '''
    K-近邻测试用户签到位置
    :return: None
    '''
    # 读取数据
    data = pd.read_csv("B:/PyCharm_Python_Project/PyCharm/AI/"
                       "instance_ex/facebook-v-predicting-check-ins/train.csv")
    # print(data.head(10))

    # 处理数据
    #1、缩小数据
    data = data.query("x>1.0 & x<1.25 & y>2.5 & y<2.75")

    # 处理时间的数据
    time_value = pd.to_datetime(data['time'],unit='s')      # 时间年月日的转化
    # print(time_value)

    # 把日期格式转换为字典格式
    time_value = pd.DatetimeIndex(time_value)       # 里面还有年月日时秒等...字典形式的特征

    # 构造一些特征---日、天、时等
    data['day_04'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday

    # 把time_value中的时间戳删除
    data = data.drop(['time'],axis=1)      # axis=1表示一列的改变

    # 去除无关特征
    # data = data.drop(['row_id'],axis=1)
    print(data)

    # 把签到数量少于n个目标位置删除
    place_count = data.groupby('place_id').count()      # 以'place_id'分组,总计次数
    tf = place_count[place_count.row_id > 3].reset_index()          # reset_index??
    data = data[data['place_id'].isin(tf.place_id)]             # data中保留所有tf.place_id>3要求的项np.where

    # 去除数据当中的特征值和目标值
    y = data['place_id']
    x = data.drop(['place_id'],axis=1)

    # 进行数据的分割训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)


    # 特征工程（标准化）
    std = StandardScaler()

    # 对测试集和训练集的特征值进行标准化
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 进行算法流程
    knn = KNeighborsClassifier(n_neighbors=5)
    '''
    # fit,predict,score
    knn.fit(x_train,y_train)

    # 得出预测结果
    y_predict = knn.predict(x_test)
    print(f"预测的目标签到位置为:{y_predict}")
    print(f"预测的准确率为:{knn.score(x_test,y_test)}")
    '''
    # 构造一些参数进行测试
    parameter = {'n_neighbors':[1,3,4]}

    # 进行网格搜索
    gc = GridSearchCV(knn,param_grid=parameter,cv=2)        # cv是分成的部分,交叉验证

    gc.fit(x_train,y_train)

    # 预测准确率
    print("在测试集上准确率：", gc.score(x_test, y_test))

    print("在交叉验证当中最好的结果：", gc.best_score_)

    print("选择最好的模型是：", gc.best_estimator_)

    print("每个超参数每次交叉验证的结果：", gc.cv_results_)

    return None

if __name__ == '__main__':
    KNN_cls()