import numpy as np
from gradient import numerical_gradient
import matplotlib.pylab as plt

def gradient_descent(f, init_x, lr=0.01, step_num = 100):
    x = init_x
    x_history = []
    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f,x)
        x -= lr * grad
        # print('1:', x_history,' 2:', x)
    return x, np.array(x_history)           # x是当前，x_history是所有
#%%
def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

if __name__ == '__main__':
    init_x = np.array([-3.0, 4.0])
    x, x_history = gradient_descent(function_2, init_x = init_x, lr=0.1, step_num=100)
    # print(x, '    ', x_history)

    plt.plot( [-5, 5], [0,0], '--b')
    plt.plot( [0,0], [-5, 5], '--b')
    plt.plot(x_history[:,0], x_history[:,1], 'o')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()