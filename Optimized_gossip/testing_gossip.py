import numpy as np
import matplotlib.pyplot as plt
from utilities_gossip import W_ij, neighbors, generate_graph, random_walk_matrix, spectral_gap, W_of_P
from gossip import  practical_test_P
from subgradient_descent import subgradient_descent_step
from centralized_one_layer import minimal_subgradient_descent_step
from centralized_two_layer import centralized_subgradient_descent_step


if __name__ == "__main__" :

    size = 10
    p = 0.04
    graph = generate_graph(size, p, seed = 1)
    P_initial = random_walk_matrix(graph)
    x0 = np.random.rand(size)
    #prepare update matrices
    n = size
    W_ijs = {(i,j): W_ij(i, j, n) for i in range(n) for j in range(n)}

    print(spectral_gap(P_initial))

    #compute optimal P using the fully centralized algorithm
    n_iter = 0
    P = P_initial.copy()
    for i in range(15000) :
        P2 = P.copy()
        P = minimal_subgradient_descent_step(P, neighbors(graph), graph, n_iter)
        if (i+1)%1000 == 0 :
             print("apprixmate norm of subgradient"+str(i+1)+" : "+str(np.linalg.norm(1e6*(P-P2)) * i))
    P_opt = P.copy()
    lambda2_opt = spectral_gap(P_opt)
    print("final spectral gap : "+str(spectral_gap(P_opt)))

    #compute optimal P  with error using the decentralized algorithm
    n_iter = 0
    Ps = [P_initial.copy()]
    P = P_initial.copy()
    print("initial spectral gap : "+str(spectral_gap(P)))
    for i in range(50) :
        P = centralized_subgradient_descent_step(P, neighbors(graph), graph, n_iter, 1e-2)
        if (i+1)%2 == 0 :
             Ps.append(P.copy())
             print("apprixmate norm of subgradient at iteration "+str(i+1)+" : "+str(np.linalg.norm(1e6*(P-P2)) * i))
    print("final spectral gap : "+str(spectral_gap(P)))

    
    #compute P using the fully decentralized
    n_iter = 0
    Ps_practice = [P_initial.copy()]
    P_dec = P_initial.copy()
    print("initial spectral gap : "+str(spectral_gap(P_dec)))

    for i in range(50) :
        P2 = P_dec.copy()
        P_dec = subgradient_descent_step(P_dec, neighbors(graph), graph, n_iter, W_ijs=W_ijs).copy()
        if (i+1)%2 == 0 :
             Ps_practice.append(P_dec.copy())
             print("apprixmate norm of subgradient at iteration "+str(i+1)+" : "+str(np.linalg.norm(1e6*(P_dec-P2)) * i))
    print("final spectral gap : "+str(spectral_gap(P_dec)))


    eig_decentralized = [abs(spectral_gap(P_iter) - lambda2_opt) for P_iter in Ps_practice]
    plt.plot(eig_decentralized, label = "Decentralized subgradient descent convergence")
    eig_centralized = [abs(spectral_gap(P_iter) - lambda2_opt) for P_iter in Ps]
    plt.plot(eig_centralized, label = "Partially centralized subgradient descent convergence")
   
    plt.title("decentralized vs centralized subgradient descent convergence")
   
    plt.xlabel("Iteration")
    plt.ylabel("distance to optimal \lambda_2")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
