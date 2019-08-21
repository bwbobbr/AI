# Matplotlib
    import matplotlib.pyplot as plt
## Class one 
- `plt.plot(x,y)` # 绘制x-y图(自变量,因变量)
- `plt.show()`  # 绘图

## Class two
    图像, 画板的显示
- plt.tight_layout() # 自动调整图像边缘,使图像最大充满figure
- plt.figure()      # 图像绘制,多图像则多plt.figure()
    - `plt.figure(num=3,figsize=(8,5))`   # 图像名字和图像大小
    - dpi,facecolor,edgecolor,frameon(框架是否绘制) 包含参数
- `plt.plot(x,y,color='red',linewidth=1.0,linestyle='--')`
    - color:颜色
    - linewidth:线宽
    - linestyle:线样式

## Class three
    设置坐标轴1
```python
plt.xlim(-1,2)              # x轴上取值范围的限定同理可以设置y的取值范围
plt.xlabel("I am X")       # 坐标轴的命名
plt.xticks(new_ticks)       # 换x轴显示的分度数值(可换数字或者是文字)
plt.yticks([-2,-1.8,-1,1.22,3,],
            ['really bad','bad','normal','good','very good'])
```


## Class four
    设置坐标轴2
```python
ax = plt.gca()
ax.spines['right'].set_color('none')        # 让某边的边框消失
ax.spines['top'].set_color("none")

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# x轴由bottom代替, y轴由lelf代替  ('bottom','left'为之前的边框)

ax.spines['bottom'].set_position(('data',-1))   # 注意这里要两个括号,'data'依照数值(由:outward和axes)
# 挪动x,y的位置----横坐标的值就是 y轴 的-1--x轴对着y轴上的-1值
ax.spines['left'].set_position(('data',0))

```

## Class five
    图例legend
 ```python
 plt.plot(x,y,lable="name")
 plt.legend() # 默认状态
```

## Class six
    图片中添加注解annotation,详细的见 class 6.py
- `plt.scatter(x0,y0)` 描点
- `plt.annotate()`     # 传统方法的添加注释
- `plt.text()`        # 简单的添加注释

## Class seven
    图片的能见度tick
- 参数 alpha=0.7 设置透明度
```python
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    # 提取出所有的x,y坐标上的表示将其尺寸大小变为12
    label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.8, zorder=2))
    # 相当于文本框, 前景色,边框,透明度
plt.show()
```

## Class eight
    Scatter散点图
- plt.scatter()


## Class nine
    柱状图bar
    
## Class ten
    等高线图


## Class twelve
    3D图
- `from mpl_toolkits.mplot3d import Axes3D`导入的模块


## Class thirteen
    一个figure绘画多个图
- plt.plot(2,2,2)
    分成4幅图, 里面的第二幅ax

## Class fourteen
    分格显示  与class thirteen相同的绘图,
- 用到新模块`import matplotlib.gridspec as gridspec`
- 方法一: `ax1 = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=1)`
- 方法二:
```python
gs = gridspec.GridSpec(3,3)
ax11 = plt.subplot(gs[0,:])
```

## Class fifteen
    图中图
 
 
## Class sixteen
    次坐标
 
## Class seventeen
    动画图
- `from matplotlib import animation`
- `ani = animation.FuncAnimation(fig=fig, func=animate, frames=100, init_func=init,
                              interval=20, blit=False)`
    - fig:画布,func:,frames:总长度帧数,init_func:最开始的画面,interval:刷新频率,blit:是不是更新整张图()画布--更换为False,不更换True