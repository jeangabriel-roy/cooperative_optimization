import numpy as np
import numpy.random as rd
import networkx as nx 
import matplotlib.pyplot as plt
#all graphs are represented as a adjacency matrix (1 if neighbor, 0 otherwise)
def Cov(x):
    m = len(x)
    Kmm = np.eye(m)
    for ii in range(m):
        for jj in range(ii+1,m):
            Kmm[ii,jj] = np.exp(-(x[ii]-x[jj])**2 )
            Kmm[jj,ii] = Kmm[ii,jj]

    return Kmm

def Cov2(x1,x2):
    m = len(x2)
    n = len(x1)
    Knm = np.zeros([n,m])
    for ii in range(n):
        for jj in range(m):
            Knm[ii, jj] = np.exp(-(x1[ii] - x2[jj]) ** 2 )
    return Knm

def get_comm_mat(graph) :
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

def generate_graph(size, p, seed = 1) :
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
        print("iteration "+str(is_connected))

        #check if the graph is connected
        G = nx.from_numpy_array(Mat) 
        if nx.is_connected(G) :
            print("created connected graph after "+ str(is_connected)+" iterations")
            is_connected = -1

    return Mat

def cost_function(alpha, x_m, x_n, y_n) :
    """
    Docstring for cost_function
    
    :param alpha: wheights
    :param x_m: points selected for nystrom approx
    :param x_n: data points
    :param y_n: data labels
    """
    cost = (1/8)*np.dot(alpha, np.dot(Cov(x_m),alpha)) + (1/2)*np.linalg.norm(alpha)
    cost += np.linalg.norm(y_n - np.dot(Cov2(x_n, x_m), alpha))
    return cost


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