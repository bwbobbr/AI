# 2019-7-3 开始机器学习视频学习

## 特征工程与文本提取

## csv文件名必须为英文格式的,说明测试文件中

### 划分训练集和测试集中返回的参数必须为'x_train,x_test,y_train,y_test'不能改变参数
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

## day_01
    day_01.py / 降维案例.py
- 字典数据抽取
- 文本特征值
- 中文特征值化
- 归一化/标准化
- 缺失值的补全/过滤低特征
- 降维PCA
- 降维

## day_02
    day_02.py / KNN.py / KNN_鸢尾花.py / 决策树_泰坦尼克号.py
    
### day_02.py
- 数据返回说明(离散型和连续型)
- 数据集的划分

### KNN.py
- K-近领算法
- KNN_鸢尾花.py 案例

###　决策树_泰坦尼克号.py
- 决策树
- 随机森林

## day_03
    线性回归_boston房价.py / 逻辑回归_二分类_癌症案例.py
- 线性回归
- 岭回归 == 线性回归 + 正则化(防止过拟合)
- 模型的加载和保存
- 逻辑回归


## day_04-----TensorFlow
    day_04.py
    
### day_04.py
    tensorboard    --logdir=/tmp/tensorflow/summary/test/
    http://localhost:6006在chrome中开启
- draw_graph 图的构建,张量(tensor)与运算操作(Operation)的说明
- change_shape 形状的概念--静态形状和动态形状
- tensor_product 张量的生成
- op 变量
-       初始化变量, 且在会话中开启初始化
- tensor_product 线性回归
-       变量定义的时候写上数据的类型, dtype=tf.float32 
-       tensorboard中变量作用域, 增加变量显示


## day_05
    day_05.py / CSV文件读取案例.py / picture读取 / 二进制 / tfrecord

### day_05.py
- 模拟同步操作
- 模拟异步操作

### CSV文件读取案例.py
    其中文件读取中分为'特征值部分'与'目标值'
    
### picture文件案例.py
    仅仅是读取数据，只有 特征值部分 无 目标值

### 二进制文件案例.py
    需要将其中的‘特征值’ 与 ‘目标值’分离出来可以一一对应比较训练

## day_06
    神经网络_手写识别.py / CNN_手写识别.py
 
### 神经网络_手写识别.py
- loss交叉熵的损失值需要是平均值
- 简单地神经网络模型, 只有全连接层

###　CNN_手写识别.py
- 注意函数中返回值是什么
- API的熟悉
- 交叉熵损失
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
- 优化函数梯度下降 GradientDescentOptimizer
- 自适应矩阵--Adam 这个名字来源于adaptive moment estimation
- Dropout函数, Googlenet中有该步骤, 防止模型的过拟合

## day_07
    
















