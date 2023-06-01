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
with open('data\df_prep_sd.pkl', 'rb') as file:
    df = pickle.load(file)

# init instance for fused lasso
fl = reg_lasso(df)

fl.features
fl.meta_attr

fl.set_features()
fl.features

# preprocess data
fl.create_timestamps('date', entity='permno')
fl.impute_missing_timestamps(entity='permno')

# create design matrix
fl.create_design_matrix(entity='permno')

#fl.X.shape
#P = int(fl.X.shape[1])
#T = fl.df.time.max() + 1
#t = fl.df.iloc[11].time
#idx = np.arange(1, P, T) -1 + int(t)
#fl.df.iloc[11]
#fl.X[11].toarray()[0][idx]

# define the problem
fl.create_D()

problem = fl.define_problem()

# solve
tic = time.time()
fl.fit(max_its=1000, tol=1e-10)
print(time.time() - tic)

fl.check_convergence()
coefs = fl.structure_coefs()

coefs.mean() * 100

# Storing the object
#with open('design_matrix.pkl', 'wb') as file:
#    pickle.dump(DM, file)

########################################################################################

# perform cross-sectional OLS regression
import statsmodels.api as sm

df.sort_values(by=['permno', 'date'], inplace=True)
df.head()
attr = fl.features

times = np.sort(df.date.unique())

results = {}

for t in times:
    print(t)
    # extract data for this timepoint
    df_tmp = df[df.date == t]

    # simple linear regression
    lm = sm.OLS(df_tmp.mret, df_tmp[attr])
    lm_res = lm.fit()

    results[t] = lm_res.params

results_df = pd.DataFrame(results).T

with open('beta_ols.pkl', 'wb') as file:
    pickle.dump(results_df, file)



