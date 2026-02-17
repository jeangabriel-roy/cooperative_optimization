import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('first_database.pkl', 'rb') as f:
   x,y = pickle.load(f)
print(x.shape)
print(y.shape)

n = 1000000
m = int(np.sqrt(n))

sel = [i for i in range(n)]
np.random.seed(1)
ind = np.random.choice(sel, m, replace=False) 
x_selected = [x[i] for i in ind]
y_selected = [y[i] for i in ind]
plt.scatter(x_selected, y_selected,)
plt.show()