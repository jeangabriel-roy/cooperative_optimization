import numpy as np
import matplotlib.pyplot as plt
import pickle
from utilities import Cov, Cov2

def solve(x_m, x_n, y):
    n = len(x_n)

    M = Cov2(x_n, x_m)
    A = (0.5**2)*Cov(x_m) + M.T @ M
    b = M.T @ y

    # here the regularization parameter nu is 1.0
    A = A + 1.*np.eye(len(x_m))

    # it is good to compute the max/min eigenvalues of A for later, but only for small-size matrices
    if n<101:
        ei, EI =np.linalg.eig(A)
        vv = [min(ei), max(ei)]
        print('Min and max eigenvalues of A : ', print(vv))

    alpha = np.linalg.solve(A,b)

    return alpha

def plot_me(x,y, alpha, ind, selection=True):

    plt.plot(x,y,'o')

    xo = np.linspace(-1,1,100)
    if selection:
        x2 = [x[i] for i in ind]
    else:
        x2 = np.linspace(-1, 1, 10)


    yo = Cov2(xo, x2) @ alpha
    plt.plot(xo, yo, '-')
    plt.xlabel(r'$x$ feature')
    plt.ylabel(r'$y$ label')
    plt.grid()

    plt.show()


"""
Main follows
"""
if __name__ == "__main__" :
    with open('first_database.pkl', 'rb') as f:
        x,y = pickle.load(f)


    num_points = 100000
    alpha, ind = solve(x[:num_points],y[:num_points], selection=True)

    print('Result summary -----------------')
    print('Optimal centralised alpha = ', alpha)

    plot_me(x[:num_points],y[:num_points], alpha, ind, selection=True)