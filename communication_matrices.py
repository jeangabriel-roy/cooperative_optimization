import numpy as np
import numpy.random as rd
import networkx as nx 
import matplotlib.pyplot as plt
#all graphs are represented as a adjacency matrix (1 if neighbor, 0 otherwise)

def get_comm_mat(graph) :
    a = graph.shape[0] #number of nodes
    degrees = np.zeros(a)
    for i in range(a) :
        degrees[i] = graph[i].sum()

    #calculate the lazy metropolis matrix
    W = np.zeros((a,a))
    for i in range(a) :
        for j in range(a) :
            if i != j :
                W[i,j] = 1/(2*max(degrees[i], degrees[j]))
    for i in range(a) :
        W[i,i] = 1 - W[i].sum()

    return W

def generate_graph(size, seed = 1) :
    is_connected = 0
    while is_connected >= 0 :
        #create new graph
        rd.seed(seed)
        bits = rd.randint(low=0, high=2, size=(size*(size-1))//2)
        Mat = np.zeros((size, size))
        for i in range(size) :
            for j in range(size) :
                if j > i :
                    Mat[i,j] = bits[j+i*(size-1-i)]
                elif j < i :
                    Mat[i,j] = bits[i+j*(size-1-j)]
        is_connected +=1 #change the seed

        #check if the graph is connected
        G = nx.from_numpy_array(Mat) 
        if nx.is_connected(G) :
            print("created connected graph after "+ str(is_connected)+" iterations")
            is_connected = -1

    return Mat
    
if __name__ == "__main__":
    #test graph
    graph = np.array([[0, 1, 1, 1], 
            [1, 0, 1, 1], 
            [1, 0, 1, 1], 
            [1, 1, 1, 0]])
    random_graph = generate_graph(10, 111819)
    # print(random_graph)
    # print(get_comm_mat(random_graph))

    G = nx.from_numpy_array(random_graph) 
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray') 
    plt.show()