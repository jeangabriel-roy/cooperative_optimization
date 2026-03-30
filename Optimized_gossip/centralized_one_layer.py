import numpy as np
import matplotlib.pyplot as plt
from utilities_gossip import neighbors, generate_graph, random_walk_matrix, W_of_P, spectral_gap
from gossip import  practical_test_P
from subgradient_descent import project_P


def minimal_subgradient_descent_step(P, neighbours, graph, n_iter) :
    n = len(P)
    #compute eigenvector associated with the second largest eigenvalue
    W = W_of_P(P)
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    eigenvalues, eigenvectors = zip(*sorted(zip(np.abs(eigenvalues), eigenvectors.T)))
    v = eigenvectors[-2] #eigenvector associated with the second largest eigenvalue
        
    #compute subgradient
    G = np.zeros((n,n))
    for i in range(n) :
            for j in neighbours[i] :
                G[i,j] = -1/(2*n) * (v[i] - v[j])**2
        
    #update P
    P = P - (1/(n_iter+1))*G

    #projet P on the feasible space
    P = project_P(P, neighbours, graph)

    return P

if __name__ == "__main__" :


    size = 10
    p = 0.04
    graph = generate_graph(size, p, seed = 1)
    P = random_walk_matrix(graph)
    x0 = np.random.rand(size)
    print(spectral_gap(P))
    print(practical_test_P(P,x0, graph))

    n_iter = 0
    for i in range(15000) :
        P2 = P.copy()
        P = minimal_subgradient_descent_step(P, neighbors(graph), graph, n_iter)
        if (i+1)%1000 == 0 :
             print("apprixmate norm of subgradient"+str(i+1)+" : "+str(np.linalg.norm(1e6*(P-P2)) * i))
             # print(test_P(P,x0, graph))

    
