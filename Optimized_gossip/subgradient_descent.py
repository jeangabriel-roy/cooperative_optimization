import numpy as np
import matplotlib.pyplot as plt
from utilities_gossip import neighbors, generate_graph, random_walk_matrix, spectral_gap, W_of_P, W_ij
from gossip import gossip, practical_test_P


#project P on the set of stochastic matrices
def project_P(P, neighbours, graph) :
    for i in range(len(P)) :
            if P[i].sum() <= 1 :
                P[i] = P[i] + (1 - P[i].sum())/len(neighbours[i]) * graph[i] #add the missing mass to the neighbors
            else :
                #find critical index 
                pjs = sorted(P[i, neighbours[i]])
                j = 0
                while np.maximum(P[i] - pjs[j], 0).sum() > 1 :
                    j += 1
                #then apply the method from the case sum(pj) > 1
                x = (1 - np.maximum(P[i] - pjs[j], 0).sum())/(len(pjs) - j)* graph[i]
                P[i] = np.maximum(P[i] - pjs[j] + x, 0)
    return P

def subgradient_descent_step(P, neighbours, graph, n_iter, W_ijs = dict()) :
    n = len(P)
    #initialize eigenvalue search
    tol  = 1e-2
    W = W_of_P(P)
    v = np.random.rand(n)
    v2 = v
    err = 10
    while(err > tol) :
            v = W @ v2 #decentralized by nature

            #decentralized average. in the end, every agent substracts its personal average
            v = v - gossip(v, P, neighbours, tol/10, 10000000000000, W_ijs=W_ijs)[0][-1]

            #decentralized normalization
            norm = np.sqrt(gossip(np.power(v, 2), P, neighbours, tol/10, 10000000000000, W_ijs=W_ijs)[0][-1])
            for i in range(n) :
                if norm[i] > 0 :
                    v[i] = v[i]/norm[i]

            err = np.linalg.norm(v - v2)
            v2 = v
        
    #compute subgradient
    G = np.zeros((n,n))
    for i in range(n) :
            for j in neighbours[i] :
                G[i,j] = -1/(2*n) * (v[i] - v[j])**2
        
    #update P
    P = P - 1/(n_iter+1)*G

    #projet P on the feasible space
    P = project_P(P, neighbours, graph)

    return P

if __name__ == "__main__" :

    size = 5
    p = 0.05
    graph = generate_graph(size, p, seed = 1)
    P = random_walk_matrix(graph)
    x0 = np.random.rand(size)

#prepare update matrices
    n = len(P)
    Wijs = {(i,j): W_ij(i, j, n) for i in range(n) for j in range(n)}

    print(spectral_gap(P))

    n_iter = 0
    for i in range(20) :
        P = subgradient_descent_step(P, neighbors(graph), graph, n_iter, W_ijs=Wijs)
        if i%1 == 0 :
             print(spectral_gap(P))
