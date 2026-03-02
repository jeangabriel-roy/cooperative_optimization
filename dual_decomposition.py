import numpy as np
import matplotlib.pyplot as plt
from utilities import get_comm_mat, neighbors
from centralized_solution import Cov, Cov2

def solve_local(x_m, x_n, y, lambda_, node, neighbours, a):
    '''Docstring for solve_local
    :param x_m: points selected for nystrom approx
    :param x_n: data points for the agent
    :param y: data labels for the agent
    :param lambda_: multipliers for the consensus constraints
    :param node: index of the agent
    :param neighbours: list of indices of the neighbours of the agent
    '''

    Kmn = Cov2(x_n, x_m)
    A = (1/a)*( 0.25*Cov(x_m)  + 1.*np.eye(len(x_m)) ) + Kmn.T @ Kmn 
    b = Kmn.T @ y

    #then add multipliers (in the linear term b)
    for j in neighbours :
        if node < j :
            b -= lambda_[node][j-node-1]
        else :
            b+= lambda_[j][node-j-1]

    alpha = np.linalg.solve(A,b)

    return alpha

def dual_decomposition(ind, graph, N_iter, step_size, x, y) :
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
    alphas = [[np.random.rand(m) for _ in range(a)]]
    lambdas = [[np.zeros(m) for _ in range(k+1, a)] for k in range(a)] #multipliers initialization
    neighbours = neighbors(graph) #dictionary containing the list of neighbors of each agent
    n_iter = 0

    while n_iter < N_iter :
        New_alpha = np.zeros((a, m)) #created to allow synchronicity of updates
        new_lambdas = [[np.zeros(m) for _ in range(k+1, a)] for k in range(a)] #created to allow synchronicity of updates
        for i in range(a) :
            x_n = x[i*nup : (i+1)*nup]
            y_n = y[i*nup : (i+1)*nup]
            New_alpha[i] = solve_local(x_m, x_n, y_n, lambdas, i, neighbours[i], a) #local optimization step
        
        for i in range(a) :
            for j in range(i+1, a) :
                if j in neighbours[i] : 
                    new_lambdas[i][j-i-1] = lambdas[i][j-i-1] + step_size*(New_alpha[i] - New_alpha[j]) #multiplier update

        alphas.append(New_alpha)
        lambdas = new_lambdas
        n_iter+=1
        if n_iter%500 == 0 : print("completed iteration " + str(n_iter))

    return alphas