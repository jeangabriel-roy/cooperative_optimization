import numpy as np
import matplotlib.pyplot as plt
import pickle
from utilities import generate_graph, cost_function
from DGD import DGD
from GT import GT
from dual_decomposition import dual_decomposition
from ADMM import ADMM
from centralized_solution import solve

n_agents = 5
seed = 1081170878575868
graph = generate_graph(n_agents, 0.3)

with open('first_database.pkl', 'rb') as f:
   x,y = pickle.load(f)

n = 100
m = int(np.sqrt(n))
sel = [i for i in range(n)]
ind = np.random.choice(sel, m, replace=False)
x_n = x[:n]
y_n = y[:n]
x_m = x[ind] #points used for kernel interpolation

iter = 5000
step_size = 0.005

alpha_star = solve(x_m, x_n, y_n) #reference solution

# alphas_DGD = DGD(ind, graph, iter, step_size, x_n, y_n)

# alphas_GT = GT(ind, graph, iter, step_size, x_n, y_n)

# alphas_DD = dual_decomposition(ind, graph, iter, step_size*10, x_n, y_n)

alphas_ADMM = ADMM(ind, graph, iter, 1, x_n, y_n)

#plot results
# results_DGD = np.linalg.norm(alphas_DGD - alpha_star, axis = 2)
# results_GT = np.linalg.norm(alphas_GT - alpha_star, axis = 2)
# results_DD = np.linalg.norm(alphas_DD - alpha_star, axis = 2)
results_ADMM = np.linalg.norm(alphas_ADMM - alpha_star, axis = 2)

plt.xscale('log')
plt.yscale('log')
# plt.plot(results_DGD, label = 'DGD')
# plt.plot(results_GT, label = 'GT')
# plt.plot(results_DD, label = 'DD')
plt.plot(results_ADMM, label = 'ADMM')
plt.legend()
plt.show()
