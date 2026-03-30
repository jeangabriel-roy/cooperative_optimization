import numpy as np
import matplotlib.pyplot as plt
from utilities_gossip import W_ij, neighbors, generate_graph, random_walk_matrix, spectral_gap, W_of_P
from gossip import  practical_test_P
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
    P2 = [P_initial.copy()]
    P = P_initial.copy()
    print("initial spectral gap : "+str(spectral_gap(P)))
    for i in range(15000) :
        P = centralized_subgradient_descent_step(P, neighbors(graph), graph, n_iter, 1e-2).copy()
        if (i+1)%200 == 0 :
             P2.append(P.copy())
             print("apprixmate norm of subgradient at iteration "+str(i+1)+" : "+str(np.linalg.norm(1e6*(P-P2)) * i))
    print("final spectral gap : "+str(spectral_gap(P)))


#compute optimal P  with error using the decentralized algorithm
    n_iter = 0
    P3 = [P_initial.copy()]
    P = P_initial.copy()
    print("initial spectral gap : "+str(spectral_gap(P)))
    for i in range(15000) :
        P = centralized_subgradient_descent_step(P, neighbors(graph), graph, n_iter, 1e-3).copy()
        if (i+1)%200 == 0 :
             P3.append(P.copy())
             print("apprixmate norm of subgradient at iteration "+str(i+1)+" : "+str(np.linalg.norm(1e6*(P-P2)) * i))
    print("final spectral gap : "+str(spectral_gap(P)))


#compute optimal P  with error using the decentralized algorithm
    n_iter = 0
    P4 = [P_initial.copy()]
    P = P_initial.copy()
    print("initial spectral gap : "+str(spectral_gap(P)))
    for i in range(15000) :
        P = centralized_subgradient_descent_step(P, neighbors(graph), graph, n_iter, 1e-4).copy()
        if (i+1)%200 == 0 :
             P4.append(P.copy())
             print("apprixmate norm of subgradient at iteration "+str(i+1)+" : "+str(np.linalg.norm(1e6*(P-P2)) * i))
    print("final spectral gap : "+str(spectral_gap(P)))

#compute optimal P  with error using the decentralized algorithm
    n_iter = 0
    P5 = [P_initial.copy()]
    P = P_initial.copy()
    print("initial spectral gap : "+str(spectral_gap(P)))
    for i in range(15000) :
        P = centralized_subgradient_descent_step(P, neighbors(graph), graph, n_iter, 1e-5).copy()
        if (i+1)%200 == 0 :
             P5.append(P.copy())
             print("apprixmate norm of subgradient at iteration "+str(i+1)+" : "+str(np.linalg.norm(1e6*(P-P2)) * i))
    print("final spectral gap : "+str(spectral_gap(P)))


    eig_2 = [abs(spectral_gap(P_iter) - lambda2_opt) for P_iter in P2]
    eig_3 = [abs(spectral_gap(P_iter) - lambda2_opt) for P_iter in P3]
    eig_4 = [abs(spectral_gap(P_iter) - lambda2_opt) for P_iter in P4]
    eig_5 = [abs(spectral_gap(P_iter) - lambda2_opt) for P_iter in P5]
    plt.plot(eig_2, label = "Tolerance 1e-2")
    plt.plot(eig_3, label = "Tolerance 1e-3")
    plt.plot(eig_4, label = "Tolerance 1e-4")
    plt.plot(eig_5, label = "Tolerance 1e-5")
    plt.title("Approximate subgradient descent convergence vs Tolerance")
    plt.xlabel("Iteration")
    plt.ylabel("distance to optimal \lambda_2")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
