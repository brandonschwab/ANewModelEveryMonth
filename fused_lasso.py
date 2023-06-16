import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import inspect
import time
import regreg.api as rr
import pylab
from scipy import sparse
from sklearn.metrics import mean_squared_error as mse
from helper_functions import impute_missing_timestamps, create_design_matrix
from reg_lasso_class import reg_lasso

# load data
# Load the pickle file
with open('C:\\Users\\bs-lokal\\Desktop\\git_repos\\anmem_data\\data\\df_prep_sd.pkl', 'rb') as file:
    df = pickle.load(file)

#df = pd.read_csv('C:/Users/bs-lokal/Desktop/ANewModelEveryMonth/df_prep_mad.csv')

# reduce data
ids = df.permno.unique()[:10]
df_short = df.copy()[df.permno.isin(ids)]
df_short2 = df.copy()[df.permno.isin(ids)]
#df_short = df.copy()

# Check if there is more than one entry for a given year and a given month
df_short['date'] = pd.to_datetime(df_short['date'])
df_short['year'] = df_short['date'].dt.year
df_short['month'] = df_short['date'].dt.month

counts = df_short.groupby(['permno', 'year', 'month']).size()
counts[counts > 1]

min_year = df_short['year'].min()
min_month = df_short[df_short['year'] == min_year]['month'].min()

# Calculate the time relative to the origin
df_short['time'] = (df_short['year'] - min_year) * 12 + (df_short['month'] - min_month)
df_short.head()

df_short.sort_values(by=['permno', 'time'], inplace=True)

# impute missing values
data_imp = impute_missing_timestamps(df_short, entity = 'permno', time = 'time')
data_imp.sort_values(by=['permno', 'time'], inplace=True)

# create the design-matrix
attr = list(data_imp.columns)
attr = [col for col in attr if col not in ['permno', 'mret', 'date', 'time', 'month', 'year']]
len(attr)

tic = time.time()
X = create_design_matrix(data_imp, entity = 'permno', time = 'time', attr=attr)
# remove zero entries
nonzero_rows = X.getnnz(axis=1).ravel() > 0
X = X[nonzero_rows, :]
toc = time.time()
print(toc-tic)

# check if design matrix is constructed correctly
P = int(X.shape[1])
T = df_short.time.max() + 1
t = df_short.iloc[11].time
idx = np.arange(1, P, T) -1 + int(t)
df_short.iloc[0]

##### Fused Lasso #####
import scipy.sparse as sp

#X_transposed = X.transpose()
#XX_product = X_transposed.dot(X)
#XX_inverse = sp.sparse.linalg.inv(XX_product)

# Y - (Nt*T) x 1
Y = np.array(df_short.mret.values)

# quick dimension check
len(Y) == X.shape[0]

# X - (Nt*T) x (T*P) [P = Number of features]
X.shape

P = int(X.shape[1])
n = len(Y)

loss = rr.quadratic_loss.affine(X, -Y, coef=0.5)
loss.shape

sparsity = rr.l1norm(P, lagrange=2.26)
sparsity.shape

# every Tth row must be zero
T = df_short.time.max() + 1
J = len(attr) # features

# D - (T*P-1) x (T*P)
D = (np.identity(P) + np.diag([-1]*(P-1),k=1))[:-1]

row_positions = np.arange(T, P, T) -1
zero_row = np.zeros((1, D.shape[1]))
D[row_positions] = zero_row
D = sparse.csr_matrix(D)

Y.shape
X.shape
D.shape

fused = rr.l1norm.linear(D, lagrange=3.1)

problem = rr.container(loss, sparsity, fused)

loss.shape
sparsity.shape
fused.input_shape
fused.output_shape

solver = rr.FISTA(problem)

tic = time.time()
obj_vals = solver.fit(max_its=1000, tol=1e-5)
toc = time.time()
print(toc-tic)

plt.plot(obj_vals)
diffs = np.abs(np.diff(obj_vals[-10:]))  # Last 10 differences
print(diffs)

solution = solver.composite.coefs

betas = list(solution)
chunks = [betas[i:i+T] for i in range(0, len(betas), T)]
coefs = pd.DataFrame({'time': range(T)})

for i in range(J):
    coefs[attr[i]] = chunks[i]        

# Printing the DataFrame
print(coefs)

# plot the data
import matplotlib.pyplot as plt

Y_pred = X.dot(solution)

df_short['prediction'] = Y_pred
df_short.permno.unique()

firm_id = 10001
d = df_short[df_short.permno == firm_id]

plt.plot(d.time, d.mret, label = 'true prices')
plt.plot(d.time, d.prediction, label = 'estimated prices')
plt.grid(True)
plt.legend()
plt.show()


coefs.mean().abs().sort_values(ascending=False)*100
