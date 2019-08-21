from sklearn.datasets import load_iris,fetch_20newsgroups,load_boston
# load_iris小规模数据集,已经存在的; fetch_20newsgroups 需网上获取或预先下载
from sklearn.model_selection import train_test_split


'''
1.dataturn函数 数据集返回类型的说明
2.spilt_data函数  数据集的分割
'''
def dataturn():
    '''
    数据返回说明(分类--离散型)
    :return:
    '''
    li = load_iris()

    print("获取特征值")      # 150(样本)*4(特征个数)
    print(li.data)
    print("目标值")        # 分的类别
    print(li.target)

    print(li.DESCR)     # 数据描述

    print(li.feature_names)     # 特征名

    print(li.target_names)      # 标签名--分的类别


def dataturn2():
    '''
    数据集的返回说明(回归--连续型)
    :return:
    '''
    boston = load_boston()

    print(boston.data)
    print(boston.target)


def split_data_load():
    '''
    数据集的划分,小数据类型处理datasets.load_*
    :return:
    '''
    li = load_iris()
    # 花类型

    x_train,x_test,y_train,y_test = train_test_split(li.data,li.target,test_size=0.25)
    # 乱序的读取25%作为测试集
    print(f"训练集{x_train,y_train};'\n\n'测试集{x_test,y_test}")


def split_data_fetch():
    '''
    数据集的划分,大数据类型处理datasets.fetch_*
    :return:
    '''
    news = fetch_20newsgroups(subset='all')
    print(news.data)
    print(news.target)

    # x_train,x_test,y_train,y_test = train_test_split(li.data,li.target,test_size=0.25)
    # # 乱序的读取25%
    # print(f"训练集{x_train,y_train};'\n\n'测试集{x_test,y_test}")

if __name__ == '__main__':
    dataturn()
    # split_data_load()
    # split_data_fetch()
    # dataturn2()