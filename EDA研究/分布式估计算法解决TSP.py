import numpy as np
import matplotlib.pyplot as plt

'''
参考网站
http://mwangblog.com/?p=1754
该算法主要的就是
训练概率矩阵p的值

'''

def distance(city):
    '''
    计算距离, 矩阵之间的距离公式
    https://blog.csdn.net/frankzd/article/details/80251042
    :param city:
    :return:
    '''
    dists_matrix = np.zeros([len(city),len(city)])
    dists_matrix = np.sqrt(-2 * np.dot(city, city.T) + np.sum(np.square(city), axis=1) + np.transpose(
        [np.sum(np.square(city), axis=1)]))
    # print(f'距离矩阵{dists_matrix}')

    return dists_matrix

def make_pop(pop_size, city_number, p):
    '''
    种群的生成
    采样不合理, 导致最终的结果不收敛, 一直在震荡
    :param pop_size: 种群的大小
    :param city_number: 城市的数量
    :param p: 城市之间的概率矩阵
    :return: pop
    '''
    pop = np.zeros([pop_size, city_number]).astype(int)

    for i in range(pop_size):
        city_p = np.zeros((city_number))
        for j in range(city_number):
            if j == 0:
                pop[i, j] = np.random.randint(0,city_number)
            else:
                city_p[:] = p[pop[i, j], :][:]
                city_p[pop[i, 0:j]] = 0
                # print(city_p)

                # 此处的轮盘赌选择有问题?
                sum_city_p = np.sum(city_p)
                city_p = city_p / sum_city_p
                index = np.where(city_p >= np.random.random())

                if len(city_p[index]) == 0:
                    temp_city = np.array([k for k in range(city_number)])
                    temp_city[pop[i, 0:j]] = 0
                    index = np.where(temp_city > 0)
                pop[i, j] = index[0][0]
        pop[i, :] += 1
    return pop

def callength(pop, dist_matrix):
    '''
       计算个体的适应度函数, 即行驶路径
       :param pop: 种群
       :param dist_matrix: 距离矩阵
       :return: sum_distance
       '''
    pop_number, city_number = pop.shape
    sum_distance = np.zeros((pop_number))

    for i in range(pop_number):
        for j in range(city_number-1):
            sum_distance[i] += dist_matrix[pop[i, j]-1, pop[i, j+1]-1]
        sum_distance[i] += dist_matrix[pop[i,-1]-1, pop[i, 0]-1]

    return sum_distance

def selection(pop, mc_scale, fitness):
    '''
    优胜劣汰种群选择机制
    :param pop: 种群
    :param mc_scale: 择优选择的个数
    :param fitness: 适应度函数
    :return: select_pop
    '''

    select_pop = pop[np.argsort(fitness)[:mc_scale],:]

    return select_pop

def updata_p(select_pop):
    '''
    更新概率矩阵
    :param select_pop: 种群选择出来优秀个体
    :return: p(更新概率矩阵)
    '''
    pop_size, city_size = select_pop.shape
    count = np.zeros([city_size, city_size]).astype(int)
    for i in range(pop_size):
        for j in range(city_size-1):
            a = select_pop[i, j] - 1
            b = select_pop[i, j + 1] - 1
            count[a, b] += 1
            count[b, a] += 1
        a = select_pop[i, -1] - 1
        b = select_pop[i, 0] - 1
        count[a, b] += 1
        count[b, a] += 1
    sum_count = np.sum(count, axis=1)
    p = count/sum_count
    p = np.where(p>0 , p, 0.000001)

    return p

def draw(max_gen, best_fits, best_lists, avg_fit, dist_matrix):
    '''
    绘制迭代曲线图
    :param max_gen: 迭代次数
    :param best_fits: 记录每代最优解
    :param best_lists: 记录每代最优的适应度值
    :param avg_fit: 记录每代平均适应度值
    :return: None
    '''
    plt.figure()
    plt.plot([i for i in range(1,max_gen+1)], best_fits.reshape(100,))
    plt.show()

    plt.plot([i for i in range(1,max_gen+1)], avg_fit.reshape(100,))
    plt.show()

    fitness = callength(best_lists, dist_matrix)
    select_pop = best_lists[np.argsort(fitness)[:1],:]

    city = np.loadtxt('order_axis.txt')
    plt.scatter(city[:, 0], city[:, 1])
    # print(select_pop)

    delivery = list(select_pop[0])
    delivery.append(delivery[0])
    delivery = np.array(delivery) - 1

    plt.plot(city[delivery, 0], city[delivery, 1])
    plt.show()


    return None

def main():
    '''
    主程序
    :return:
    '''
    city = np.loadtxt('order_axis.txt')
    city_number = len(city)
    pop_size = 300
    max_gen = 100

    p = np.ones([city_number, city_number])   # 概率矩阵
    # print(p)
    mc_scale = int(np.ceil(pop_size * 0.3))      # 取样保留的种群规模, 向上取整
    # print(mc_scale)
    dist_matrix = distance(city)            # 计算城市之间的距离矩阵

    best_lists = np.zeros([max_gen, city_number]).astype(int)       # 记录每代最优解(城市配送顺序)
    best_fits = np.zeros([1, max_gen])      # 记录每代最优的适应度值
    avg_fit = np.zeros([1, max_gen])        # 记录每代平均适应度值

    for i in range(max_gen):
        pop = make_pop(pop_size, city_number, p)        # 生成初始种群

        fitness = callength(pop, dist_matrix)
        bfiti = np.where(fitness == np.min(fitness))[0][0]   # 获取最小适应度函数的索引
        bfit = np.min(fitness)
        best_lists[i, :] = pop[bfiti, :]
        best_fits[0, i] = bfit
        avg_fit[0, i] = sum(fitness) / pop_size

        select_pop = selection(pop, mc_scale, fitness)      # 优胜劣汰,保留优秀个体

        # 更新概率矩阵
        p = updata_p(select_pop)

    # fitness = callength(best_lists, dist_matrix)
    # draw(max_gen, best_fits, best_lists, avg_fit, dist_matrix)
    print(p)

    return None

if __name__ == '__main__':
    main()