import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def plot_linear_cube(x, y, z, dx, dy, dz):
    fig = plt.figure()
    ax = Axes3D(fig)
    xx = [x, x, x+dx, x+dx, x]
    yy = [y, y+dy, y+dy, y, y]
    kwargs = {'alpha': 1, 'color': 'black'}
    kwargs_test = {'alpha': 1, 'color': 'grey','linestyle': '--'}
    ax.plot3D(xx, yy, [z]*5, **kwargs)
    ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
    # 绘制的是举行[X1,X2,X3,X4,X1],[Y1,Y2,Y3,Y4,Y1]...一一对应, 绘制的过程中要回到原点
    ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
    ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs_test)

    ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
    plt.title('Cube')
    plt.show()

if __name__ == "__main__":
    plot_linear_cube(0,0,0,100,120,130)