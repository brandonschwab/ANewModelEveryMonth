import numpy as np
import pandas as pd
import time
import pylab
from scipy import sparse

# build toy dataset
Nt = 4
T = 4
P = 3

data = pd.DataFrame({
    'firm': np.repeat(np.arange(Nt), T),    
    'time': np.tile(np.arange(T), Nt),
    'price': np.random.normal(size=Nt*T),
    'attr1': np.random.normal(size=Nt*T),
    'attr2': np.random.normal(size=Nt*T),
    'attr3': np.random.normal(size=Nt*T),
    'attr4': np.random.normal(size=Nt*T),
    'attr5': np.random.normal(size=Nt*T)
})

data2 = pd.DataFrame({
    'firm': [1, 0, 0, 1, 1],
    'time': [0, 1, 2, 0, 1],
    'price': np.random.normal(size=5),
    'attr1': np.random.normal(size=5),
    'attr2': np.random.normal(size=5)
})

data3 = pd.DataFrame({
    'firm': [0, 0, 0, 1, 1],
    'time': [0, 1, 2, 0, 2],
    'price': np.random.normal(size=5),
    'attr1': np.random.normal(size=5),
    'attr2': np.random.normal(size=5)
})

# first impute missing time stamps with zeros
data2
data2 = impute_missing_timestamps(data2)
data2

data3
data3 = impute_missing_timestamps(data3)
data3

X_1 = create_design_matrix(data, attr=['attr1', 'attr2'])
X_2 = create_design_matrix(data2)
X_3 = create_design_matrix(data3)

data2
data2.sort_values(by=['firm', 'time'], inplace=True)


data.sort_values(by=['firm', 'time'], inplace=True)
data

df_grouped = data2.groupby('firm', sort=False)

for name, group in df_grouped:

    print(name)
    print(group)



# remove zero entries
nonzero_rows = X_2.getnnz(axis=1).ravel() > 0
X_2 = X_2[nonzero_rows, :]


##### Fused Lasso #####
import regreg.api as rr

# Y - (Nt*T) x 1
Y = np.array(data_imp.price)

# X - (Nt*T) x (T*P) [P = Number of features]
X = X_2.toarray()
X.shape

P = int(X.shape[1])
n = len(Y)

loss = rr.quadratic_loss.affine(X, -Y, coef=0.5)
loss.shape

sparsity = rr.l1norm(P, lagrange=0.1)
sparsity.shape

# D - (T*P-1) x (T*P)
D = (np.identity(P) + np.diag([-1]*(P-1),k=1))[:-1]

# every Tth row must be zero
T = data2.time.max() + 1
J = 2 # features

row_positions = np.arange(T, P, T) -1
zero_row = np.zeros((1, D.shape[1]))

D[row_positions] = zero_row

D = sparse.csr_matrix(D)

Y.shape
X.shape
D.shape

fused = rr.l1norm.linear(D, lagrange=1)

problem = rr.container(loss, sparsity, fused)

loss.shape
sparsity.shape
fused.input_shape
fused.output_shape

solver = rr.FISTA(problem)

obj_vals = solver.fit(max_its=100, tol=1e-10)

solution = solver.composite.coefs

# plot the data
import matplotlib.pyplot as plt

df_short.sort_values(by=['permno', 'time'], inplace=True)

Y_pred = np.dot(X.toarray(), solution)

df_short['prediction'] = Y_pred


# 1 row, 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

f1 = data2[data2.firm == 0]
Y_pred1 = Y_pred[:3]
ax1.plot(f1.time, f1.price, label = "true prices")
ax1.plot(f1.time, Y_pred1, label = "estimated prices")
ax1.grid(True)
ax1.legend()


f2 = data2[data2.firm == 1]
Y_pred2 = Y_pred[3:]
ax2.plot(f2.time, f2.price, label = "true prices")
ax2.plot(f2.time, Y_pred2, label = "estimated prices")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()