import numpy as np
import matplotlib.pyplot as plt
from utilities import get_comm_mat, gradf
from centralized_solution import Cov, Cov2

def GT(ind, graph, N_iter, step_size, x, y) :
    """
    Docstring for DGD
    
    :param ind: index of nodes selected for Nystrom approx
    :param graph: adjaceny matrix of communication graph
    :param N_iter: max iter number
    :param step_size: step size of algorithm
    """
    a = len(graph)
    nup = len(x)//a #number of data points dealt with by each agent
    x_m = x[ind]
    m = len(ind)
    W = get_comm_mat(graph)
    alphas = [[np.random.rand(m) for _ in range(a)]]
    grads = np.array([gradf(x_m, x[i*nup : (i+1)*nup], y[i*nup : (i+1)*nup], alphas[0][i], a) for i in range(a)]) #gradient estimate initialization
    
    n_iter = 0


    while n_iter < N_iter :
        curr_alpha = np.array(alphas[-1])
        New_alpha = np.zeros((a, m)) #created to allow synchronicity of updates
        new_grads = np.zeros((a, m))

        for i in range(a) :
            x_n = x[i*nup : (i+1)*nup]
            y_n = y[i*nup : (i+1)*nup]
            New_alpha[i] = np.dot(curr_alpha.T, W[i]) - step_size * grads[i] #wheight update
            new_grads[i] = np.dot(grads.T, W[i]) + (gradf(x_m, x_n, y_n, New_alpha[i], a) - gradf(x_m, x_n, y_n, curr_alpha[i], a)) #gradient update

        alphas.append(New_alpha)
        grads = new_grads
        n_iter+=1
        if n_iter%500 == 0 : print("completed iteration " + str(n_iter))

    return alphas