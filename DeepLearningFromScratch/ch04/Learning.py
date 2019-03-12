import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
#%%
#定义三维数据
def function_2(x,y):
    return x**2 + y**2
#%%
fig = plt.figure()
ax = Axes3D(fig)
x = np.arange(-3,3,0.1)
y = np.arange(-3,3,0.1)
X,Y = np.meshgrid(x,y)#创建网格，这个是关键
Z = function_2(X,Y)
plt.xlabel('x')
plt.ylabel('y')

ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
plt.show()

#%%
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

#%%
def function_temp(x0):
    return x0**2 + 4.0**2.0

x = numerical_diff(function_temp, 3.0)