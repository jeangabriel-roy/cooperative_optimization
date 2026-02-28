import numpy as np
import matplotlib.pyplot as plt
import pickle
from utilities import generate_graph, cost_function
from DGD import DGD
from GT import GT
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

alphas_DGD = DGD(ind, graph, iter, step_size, x_n, y_n)

alphas_GT = GT(ind, graph, iter, step_size, x_n, y_n)

#plot results
# costs = [[cost_function(alpha, x_m, x_n, y_n) for alpha in alphas] for alphas in alphas_DGD]
results_DGD = np.linalg.norm(alphas_DGD - alpha_star, axis = 2)
results_GT = np.linalg.norm(alphas_GT - alpha_star, axis = 2)
print(cost_function(alpha_star, x_m, x_n, y_n))
plt.xscale('log')
plt.yscale('log')
plt.plot(results_DGD, label = 'DGD')
plt.plot(results_GT, label = 'GT')
plt.legend()
plt.show()
# plt.xscale('log')
# plt.yscale('log')
# plt.plot(costs)
# plt.show()
