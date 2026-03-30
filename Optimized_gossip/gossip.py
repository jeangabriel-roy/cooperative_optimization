import numpy as np
import matplotlib.pyplot as plt
from utilities_gossip import  W_of_P, neighbors, generate_graph, random_walk_matrix, W_ij

#asynchronous gossip algorithm
def gossip(x0, P, neighbors, eps = 1e-4, N_iter = 1000, W_ijs = dict()) :
    """
    Docstring for gossip
    
    :param x0: initial values of the agents
    :param P: communication matrix (stochastic, symmetric, with zeros on the diagonal)
    :param neighbors: dictionary containing the list of neighbors of each agent
    :param eps: convergence threshold
    :param N_iter: max iter number
    """
    xs = [x0.copy()]
    x = x0.copy()
    n_iter = 0

    while n_iter < N_iter and np.linalg.norm(x - np.mean(x)) > eps :
        i = np.random.randint(len(x0)) #pick random agent
        neighbors_i = neighbors.get(i, []) #get its neighbors
        j = int(np.random.choice(neighbors_i, p = P[i][neighbors_i])) #pick random neighbor of the agent with proba P_ij
        x = W_ijs[(i,j)] @ x
        xs.append(x.copy())
        n_iter += 1

    return xs, np.linalg.norm(x - np.mean(x)), n_iter

#testing the matrix P :
#gossip algorithm
def practical_test_P(P,x0, graph, W_ijs = dict()) :
    W = W_of_P(P)
    n_iter = 0
    for i in range(100) :
        _, _, it = gossip(x0, P, neighbors(graph), 1e-5, 100000, W_ijs=W_ijs)
        n_iter += it
    return n_iter/100

if __name__ == "__main__" :
    size = 10
    p = 0.1
    graph = generate_graph(size, p, seed = 1)
    P = random_walk_matrix(graph)
    x0 = np.random.rand(size)
    n = len(P)
    Wijs = {(i,j): W_ij(i, j, n) for i in range(n) for j in range(n)}
    xs, err, n_iter = gossip(x0, P, neighbors(graph), 1e-2, 100000, W_ijs=Wijs)
    print("final error : "+str(err))
    print("number of iterations : "+str(n_iter))
    plt.plot(xs)
    plt.xlabel("Iteration")
    plt.ylabel("Values of the agents")
    plt.title("Gossip algorithm convergence")
    plt.show()