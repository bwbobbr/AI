import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# method 1:subplot2grid
###############################################
plt.figure(num='method 1')
ax1 = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=1)
# 基准是(3,3)
# 分成3行3列,从0,0开始,占一行三列(colspan=3,rowspan=1)
ax1.plot([1,2],[1,2])
ax1.set_title("ax1_title")
ax1.set_xlim()
# 与之前的变化就是添加 'set_'XXX

ax2 = plt.subplot2grid((3,3),(1,0),colspan=2,rowspan=1)
ax3 = plt.subplot2grid((3,3),(1,2),colspan=1,rowspan=2)
ax4 = plt.subplot2grid((3,3),(2,0),colspan=1,rowspan=1)
ax5 = plt.subplot2grid((3,3),(2,1),colspan=1,rowspan=1)

# method 2:gridspec
##############################################################
plt.figure(num='method 2')
gs = gridspec.GridSpec(3,3)
ax11 = plt.subplot(gs[0,:])
ax22 = plt.subplot(gs[1,:2])
ax33 = plt.subplot(gs[1:3,2])
ax44 = plt.subplot(gs[2,0])
ax55 = plt.subplot(gs[2,1])

# method 3:easy to define structure
##############################################################
#plt.figure(num='method 3')
f,((ax111,ax222),(ax333,ax444)) = plt.subplots(2,2,sharex=True,sharey=True)
# 只能绘制简单的分割
ax111.scatter([0,1],[2,1])


plt.tight_layout()
plt.show()