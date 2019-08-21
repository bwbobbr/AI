import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

'''
https://blog.csdn.net/saltriver/article/details/52194918
在x = [0,2], 计算 y = x**2 的定积分
近似方法--蒙特卡罗 求解定积分

'''

def draw_picture():
    '''
    区域图像的绘制
    :return:
    '''
    x = np.linspace(0, 2, 1000)
    y = x**2
    plt.plot(x, y)
    plt.fill_between(x, y, where=(y>0), color='red', alpha=0.5)
    # plt.show()

def imitate():
    '''
    模拟随机点的产生
    :return:
    '''
    draw_picture()
    N = 1000
    points = [[xy[0] * 2, xy[1] * 4] for xy in np.random.rand(N, 2)]
    plt.scatter([x[0] for x in points], [x[1] for x in points], s=5, c=np.random.rand(N), alpha=0.5)
    plt.show()
    return points, N

def compute_area():
    '''
    根据分布点计算面积
    :return:
    '''
    counts = 0
    area = 2 * 4
    points, N = imitate()
    for xy in points:
        if xy[0] ** 2 > xy[1]:
            counts += 1
    MC_area = area * counts/N
    return MC_area

def direct_compute():
    area = integrate.quad(lambda x:x ** 2, 0, 2)[0]
    print(area)



if __name__ == '__main__':
    # estimate_area = compute_area()
    # print(estimate_area)
    direct_compute()
