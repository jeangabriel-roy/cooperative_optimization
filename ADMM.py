import numpy as np
import matplotlib.pyplot as plt
from utilities import get_comm_mat, neighbors
from centralized_solution import Cov, Cov2


def solve_local(x_m, x_n, y, lambdas, ys, node, neighbours, a, beta):
    '''Docstring for solve_local
    :param x_m: points selected for nystrom approx
    :param x_n: data points for the agent
    :param y: data labels for the agent
    :param ys: edge variables 
    :param lambdas: multipliers for the consensus constraints
    :param node: index of the agent
    :param neighbours: list of indices of the neighbours of the agent
    '''

    Kmn = Cov2(x_n, x_m)
    A = (1/a)*( 0.25*Cov(x_m)  + 1.*np.eye(len(x_m)) ) + Kmn.T @ Kmn 
    b = Kmn.T @ y

    #then add multipliers and edge variables (in the linear term b)
    for j in neighbours :
            A += beta * np.eye(len(x_m))
            b += beta * ys[j][node] - lambdas[node][j]

    alpha = np.linalg.solve(A,b)

    return alpha


def ADMM(ind, graph, N_iter, beta, x, y) :
    """
    Docstring for ADMM
    
    :param ind: index of nodes selected for Nystrom approx
    :param graph: adjaceny matrix of communication graph
    :param N_iter: max iter number
    :param beta: step size of algorithm
    """
    a = len(graph)
    nup = len(x)//a #number of data points dealt with by each agent
    x_m = x[ind]
    m = len(ind)
    alphas = [[np.random.rand(m) for _ in range(a)]]
    lambdas = np.zeros((a,a,m)) #multipliers initialization
    ys = np.array([[(alphas[0][i] + alphas[0][j])/2 for i in range(a)] for j in range(a)]) #edge variables initialization
    
    neighbours = neighbors(graph) #dictionary containing the list of neighbors of each agent
    n_iter = 0

    while n_iter < N_iter :
        New_alpha = np.zeros((a, m)) #created to allow synchronicity of updates

        #first computation step
        for i in range(a) :
            x_n = x[i*nup : (i+1)*nup]
            y_n = y[i*nup : (i+1)*nup]
            New_alpha[i] = solve_local(x_m, x_n, y_n, lambdas, ys, i, neighbours[i], a, beta) #local optimization step
        
         #communication and second computation step
        for i in range(a) :
            for j in range(a) :
                if j in neighbours[i]:
                    #double updates for symmetry
                    ys[i][j] = (New_alpha[i] + New_alpha[j])/2
                    ys[j][i] = (New_alpha[i] + New_alpha[j])/2

                    lambdas[i][j] = lambdas[i][j] + beta*(New_alpha[i] - ys[i][j]) #multiplier update

        alphas.append(New_alpha)
        n_iter+=1
        if n_iter%500 == 0 : print("completed iteration " + str(n_iter))

    return alphas