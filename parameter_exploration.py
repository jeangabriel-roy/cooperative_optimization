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

alpha_star = solve(x_m, x_n, y_n) #reference solution

DGD_step_sizes = 0.005 * np.logspace(-3, 0, 4) #step size schedule for DGD

#plot results
results_DGD =  []
for step in DGD_step_sizes:
   alphas_DGD = DGD(ind, graph, iter, step, x_n, y_n)
   results_DGD.append(np.linalg.norm(alphas_DGD - alpha_star, axis = 2))

results_ADMM = []
betas = np.logspace(-1, 2, 4)
for beta in betas:
   alphas_ADMM = ADMM(ind, graph, iter, beta, x_n, y_n)
   results_ADMM.append(np.linalg.norm(alphas_ADMM - alpha_star, axis = 2))

colors = ['royalblue', 'darkorange', 'green', 'red']
fig1, ax1 = plt.subplots(figsize=(8, 5))
for j in range(len(DGD_step_sizes)):
   for i in range(results_DGD[j].shape[1]):
      curve = results_DGD[j][:, i]
      ax1.plot(curve, linewidth=2, color = colors[j], label='DGD step size = ' + str(DGD_step_sizes[j]) if i == 0 else None)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Error')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dgd_step_size.pdf')
plt.close(fig1)


fig3, ax3 = plt.subplots(figsize=(8, 5))
for j in range(len(betas)):
   for i in range(results_ADMM[j].shape[1]):
      curve = results_ADMM[j][:, i]
      ax3.plot(curve, linewidth=2, color = colors[j] , label='ADMM beta = ' + str(betas[j]) if i == 0 else None)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Error')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('admm_betas.pdf')
plt.close(fig3)
