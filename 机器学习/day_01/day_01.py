from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import jieba
import numpy as np

def divtvec():
    """
    字典数据抽取
    :return:
    """
    # 实例化

    dict = DictVectorizer(sparse=False)     # 二维数组,对字典数据进行特征值化
    # dict = DictVectorizer()

    # 调用fit_transform
    data = dict.fit_transform([{'city': '北京','temperature':100},
                        {'city': '上海','temperature':60},
                        {'city': '深圳','temperature':30},])

    # 字典数据抽取：把字典中一些类别数据，分别进行转换成特征，数字型不转换
    # 数据先转换

    # 获取特征
    print(dict.get_feature_names())

    print(dict.inverse_transform(data))

    print(data)
    # print(data.toarray())

    return None

def countvec():
    """
    对文本特征值化
    :return:
    """
    cv = CountVectorizer()      # 无sparse开启关闭选项
    # data = cv.fit_transform(["life is short,i like python is",
    #                   "life is too long,i dislike python"])

    data = cv.fit_transform(["人生 苦短，我 用 python","人生 漫长，不用 python"])
    # 空格分开，中文要先分词处理

    # print(data)     # sparse矩阵
    print(data.toarray())
    # 统计每篇文章特征出现的次数，单个字母不统计，没有列入特征中
    '''
    [[0 2 1 1 0 1 1 0]
    [1 1 1 0 1 1 0 1]]
    '''

    print(cv.get_feature_names())

    # print(cv.inverse_transform(data))

    return None


def cutword():
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")
    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    # 转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)

    # 把列表转换成字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)


    return c1,c2,c3

def wordsvec():
    """
    中文特征值化
    :return:
    """
    c1,c2,c3 = cutword()

    print(c1,c2,c3)
    cv = CountVectorizer()

    data = cv.fit_transform([c1,c2,c3])

    print(cv.get_feature_names())

    print(data.toarray())
    return None


def tfidfvec():
    """
    tfidf 词汇 重要性词汇特征值提取
    :return:
    """
    c1,c2,c3 = cutword()

    print(c1,c2,c3)
    tf = TfidfVectorizer()

    data = tf.fit_transform([c1,c2,c3])

    print(tf.get_feature_names(),'\n',len(tf.get_feature_names()))

    print(data.toarray())       # 将sparse矩阵转化
    return None


def normalization():
    '''
    归一化处理
    :return:
    '''
    # mm = MinMaxScaler(feature_range=(2,3))        数据由0-1变为2-3之间的
    mm = MinMaxScaler()

    data = mm.fit_transform([[90,2,10,40],
                      [60,4,15,45],
                      [75,3,13,46]])
    print(data)


def standard():
    '''
    标准化缩放---正态分布 mean == 0, var == 1
    :return:
    '''

    stand = StandardScaler()
    data = stand.fit_transform([[ 1., -1., 3.],
                                [ 2., 4., 2.],
                                [ 4., 6., -1.]])
    print(data)

def im():
    '''
    缺失值处理(填补)
    :return:
    '''
    im = Imputer(missing_values='NaN',strategy='mean',axis=0)

    data = im.fit_transform([[1, 2],
                             [np.nan, 3],
                             [7, 6]])

    print(data)


def var():
    '''
    过滤式
    特征选择---删除低方差的特征
    :return:
    '''
    var_delete = VarianceThreshold(threshold=0.0)       # 将方差为0.0的特征删除

    data = var_delete.fit_transform([[0, 2, 0, 3],
                                     [0, 1, 4, 3],
                                     [0, 1, 1, 3]])
    print(data)


def pca():
    '''
    主成分分析进行降维
    :return:
    '''
    pca = PCA(n_components=0.9)

    data = pca.fit_transform([[2,8,4,5],
                              [6,3,0,8],
                              [5,4,9,1]])
    print(data)


if __name__ =='__main__':
    # divtvec()
    # countvec()
    # wordsvec()
    # tfidfvec()
    # normalization()
    # standard()
    im()
    # var()
    # pca()