import numpy as np
import matplotlib.pyplot as plt
from utilities import get_comm_mat
from centralized_solution import Cov, Cov2

def gradf(x_m, x_n, y_n, alpha, a) :
    """
    Docstring for cost_function
    
    :param alpha: wheights
    :param x_m: points selected for nystrom approx
    :param x_n: data points
    :param y_n: data labels
    """
    grad = np.zeros(len(alpha))
    Kmn = Cov2(x_m,x_n)
    for i in range(len(x_n)) :
        grad += (np.dot(Kmn[:,i],alpha) - y_n[i])*Kmn[:,i] 
    grad += 0.25*(1/a) * np.dot(Cov(x_m), alpha) + (1/a)*alpha
    return grad

def DGD(ind, graph, N_iter, step_size, x, y) :
    """
    Docstring for DGD
    
    :param ind: index of nodes selected for Nystrom approx
    :param graph: adjaceny matrix of communication graph
    :param N_iter: max iter number
    :param step_size: step size of algorithm
    """
    a = len(graph)
    numpoints = len(x)//a #number of data points dealt with by each agent
    x_m = x[ind]
    m = len(ind)
    W = get_comm_mat(graph)
    alphas = [[np.random.rand(m) for _ in range(a)]]
    n_iter = 0

    while n_iter < N_iter :
        curr_alpha = np.array(alphas[-1])
        New_alpha = np.zeros((a, m)) #created to allow synchronicity of updates
        for i in range(a) :
            x_n = x[i*numpoints : (i+1)*numpoints]
            y_n = y[i*numpoints : (i+1)*numpoints]
            New_alpha[i] = np.dot(curr_alpha.T, W[i]) - step_size * gradf(x_m, x_n, y_n, curr_alpha[i], a)
        
        alphas.append(New_alpha)
        print("completed iteration " + str(n_iter))
        n_iter+=1

    return alphas 


