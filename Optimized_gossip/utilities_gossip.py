import numpy as np
import numpy.random as rd
import networkx as nx 
import matplotlib.pyplot as plt
#all graphs are represented as a adjacency matrix (1 if neighbor, 0 otherwise)

def get_comm_mat(graph) : #write lazy metropolis matrix for the graph
    a = graph.shape[0] #number of nodes
    degrees = np.zeros(a)
    for i in range(a) :
        degrees[i] = graph[i].sum()

    #calculate the lazy metropolis matrix
    W = np.zeros((a,a))
    for i in range(a) :
        for j in range(a) :
            if i != j and graph[i,j] > 0 :
                W[i,j] = 1/(2*max(degrees[i], degrees[j]))
    for i in range(a) :
        W[i,i] = 1 - W[i].sum()

    return W

def neighbors(graph) : #returns a dictionary that outputs neighbor nodes 
    neigh = dict()
    n = len(graph)
    for i in range(n) :
        neighs_i = []
        for j in range(n) :
            if graph[i,j] > 0 :
                neighs_i.append(j)
        neigh[i] = neighs_i
    return neigh

def generate_graph(size, p, seed = 1) : #generate random graph with a given number of agents and connectivity level
    is_connected = 0
    while is_connected >= 0 :
        #create new graph
        rd.seed(seed*is_connected)
        #pick bernouilli variables with parameter p (adjustable to obtain more or less connected graphs)
        bits = rd.binomial(1, p, size=(size*(size-1))//2)
        Mat = np.zeros((size, size))
        for i in range(size) :
            for j in range(size) :
                if j > i :
                    Mat[i,j] = bits[j+i*(size-1-i)]
                elif j < i :
                    Mat[i,j] = bits[i+j*(size-1-j)]
        is_connected +=1 #change the seed
        if is_connected%1000 == 0 :
            print("iteration "+str(is_connected))

        #check if the graph is connected
        G = nx.from_numpy_array(Mat) 
        if nx.is_connected(G) :
            print("created connected graph after "+ str(is_connected)+" iterations")
            is_connected = -1

    return Mat

def random_walk_matrix(graph) : #write random walk matrix for the graph
    a = graph.shape[0] #number of nodes
    degrees = np.zeros(a)
    for i in range(a) :
        degrees[i] = graph[i].sum()

    W = np.zeros((a,a))
    for i in range(a) :
        for j in range(a) :
            if graph[i,j] > 0 :
                W[i,j] = 1/degrees[i]

    return W

#determine average gossip matrix from P
def W_ij(i,j, n) :
    W = np.eye(n)
    W[i,j] = 0.5
    W[j,i] = 0.5
    W[i,i] = 0.5
    W[j,j] = 0.5
    return W
def W_of_P(P) :
    n = len(P)
    W = np.zeros_like(P)
    for i in range(len(P)) :
        for j in range(i+1, n) :
            if P[i,j] > 0 :
                W += P[i,j]*W_ij(i,j,n)
    return W/n

#testing the matrix P :
#second largest eigenvalue ie spectral gap 
def spectral_gap(P) :
    W = W_of_P(P)
    eigenvalues = np.linalg.eigvals(W)
    eigenvalues = np.sort(np.abs(eigenvalues))
    return 1/(1 - eigenvalues[-2])


if __name__ == "__main__":
    #test graph
    graph = np.array([[0, 1, 1, 1], 
            [1, 0, 1, 1], 
            [1, 0, 1, 1], 
            [1, 1, 1, 0]])
    random_graph = generate_graph(5, 0.05, 11181)
    print(random_graph)
    print(get_comm_mat(random_graph))

    #draw graph
    G = nx.from_numpy_array(random_graph) 
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray') 
    plt.show()