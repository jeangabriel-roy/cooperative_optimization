import numpy as np
import matplotlib.pyplot as plt
import pickle
from utilities import L_smoothness, generate_graph, cost_function
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

iter = 10000

#obtain smoothness constant for each agent (for later use in step size selection)
for i in range(n_agents) :
   x_a = x[i*n//n_agents : (i+1)*n//n_agents]
   y_a = y[i*n//n_agents : (i+1)*n//n_agents]
   print("L_smoothness constant for agent " + str(i) + " : ", L_smoothness(x_m, x_a, n_agents)[0])

#obtain largest singular value of the constraint matrix for dual decomposition
A = np.zeros((n_agents*m, n_agents*m))
for i in range(n_agents):
   for j in range(n_agents):
      if graph[i][j] == 1 and i < j:
         A[i*m:(i+1)*m, i*m:(i+1)*m] += np.eye(m)
         A[i*m:(i+1)*m, j*m:(j+1)*m] -= np.eye(m)
print("Largest singular value of the constraint matrix for dual decomposition : ", np.max(abs(np.linalg.svdvals(A))))
print("strong convexity constant of the cost function : ", L_smoothness(x_m, x_n, n_agents)[1])
alpha_star = solve(x_m, x_n, y_n) #reference solution

alphas_DGD = DGD(ind, graph, iter, 0.005, x_n, y_n)

alphas_GT = GT(ind, graph, iter, 0.005, x_n, y_n)

alphas_DD = dual_decomposition(ind, graph, iter, 0.06, x_n, y_n)

alphas_ADMM = ADMM(ind, graph, iter, 1, x_n, y_n)

#plot results
results_DGD = np.linalg.norm(alphas_DGD - alpha_star, axis = 2)
results_GT = np.linalg.norm(alphas_GT - alpha_star, axis = 2)
results_DD = np.linalg.norm(alphas_DD - alpha_star, axis = 2)
results_ADMM = np.linalg.norm(alphas_ADMM - alpha_star, axis = 2)



fig1, ax1 = plt.subplots(figsize=(8, 5))
for i in range(results_DGD.shape[1]):
   curve = results_DGD[:, i]
   ax1.plot(curve, linewidth=2, color='royalblue', label='DGD' if i == 0 else None)
for i in range(results_GT.shape[1]):
   curve = results_GT[:, i]
   ax1.plot(curve, linewidth=2, color='darkorange', label='GT' if i == 0 else None)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Error')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dgd_vs_gt.pdf')
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(8, 5))
for i in range(results_GT.shape[1]):
   curve = results_GT[:, i]
   ax2.plot(curve, linewidth=2, color='darkorange', label='GT' if i == 0 else None)
for i in range(results_DD.shape[1]):
   curve = results_DD[:, i]
   ax2.plot(curve, linewidth=2, color='royalblue', label='Dual Decomposition' if i == 0 else None)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Error')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gt_vs_dd.pdf')
plt.close(fig2)

fig3, ax3 = plt.subplots(figsize=(8, 5))
for i in range(results_DD.shape[1]):
   curve = results_DD[:, i]
   ax3.plot(curve, linewidth=2, color='royalblue', label='Dual Decomposition' if i == 0 else None)
for i in range(results_ADMM.shape[1]):
   curve = results_ADMM[:, i]
   ax3.plot(curve, linewidth=2, color='darkorange', label='ADMM' if i == 0 else None)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Error')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dd_vs_admm.pdf')
plt.close(fig3)
