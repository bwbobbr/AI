import numpy as np
import scipy as sp
from scipy.optimize import leastsq  # 最小二乘法 计算其误差最小
import matplotlib.pyplot as plt

'''
举例：我们用目标函数 𝑦=𝑠𝑖𝑛2𝜋𝑥 , 加上一个正太分布的噪音干扰，用多项式去拟合【例1.1 11页】
'''
# ps: np.poly1d([1,2,3]) 生成  1𝑥2+2𝑥1+3𝑥0
# 多项式的生成

def real_func(x):
    # 真实值
    return np.sin(2 * np.pi * x)

def fit_func(p, x):
    # 多项式(h(x) = W * X.T)
    #
    f = np.poly1d(p)
    return f(x)

def residuals_func(p, x, y):
    # 误差值
    ret = fit_func(p, x) - y
    return ret

def fitting(x,y,M):
    # 多项式的项数
    p_init = np.random.randn(M+1)

    # 最小二乘法
    p_lsq = leastsq(residuals_func, p_init, args=(x,y))
    '''
    leastsq(误差函数, 多项式(w参数的值), 数据点(特征+目标值))
    '''


    print(f'Fitting Parameter:{p_init[0]}')
    return p_lsq

def fitting2(x,y,M):
    # 多项式的项数
    p_init = np.random.randn(M+1)

    # 最小二乘法
    p_lsq = leastsq(residual_func_regularization, p_init, args=(x,y))
    '''
    leastsq(误差函数, 多项式(w参数的值), 数据点(特征+目标值))
    '''
    print(f'Fitting Parameter:{p_init[0]}')
    return p_lsq

def draw(x, x_points, y, p_lsq, p_lsq_regular):
    # 绘制图像
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitting')
    plt.plot(x_points, fit_func(p_lsq_regular[0], x_points), label='regular')

    plt.plot(x, y, 'bo',label='noise')
    plt.legend()
    plt.show()

def residual_func_regularization(p, x, y,regular = 0.0001):
    ret = fit_func(p, x) - y
    # 引入正则化项, 防止过拟合的发生
    ret = np.append(ret, np.sqrt(0.5 * np.square(p) * regular))
    return ret


def main():
    regular = 0.0001
    x = np.linspace(0, 1, 10)
    x_points = np.linspace(0, 1, 1000)  # 绘图
    y_true = real_func(x)
    y = [i+np.random.normal(0, 0.1) for i in y_true]

    p_lsq = fitting(x, y, M=9)
    p_lsq_regular = fitting2(x, y, M=9)


    # 此处M的取值为多项式的项数
    # print(p_lsq)
    draw(x, x_points, y, p_lsq, p_lsq_regular)




if __name__ == '__main__':
    main()