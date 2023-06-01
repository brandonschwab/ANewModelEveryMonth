import numpy as np
import regreg.api as rr
from sklearn import linear_model
from statsmodels import regression

X = np.random.normal(0,1,50).reshape((25,2))
Y = np.random.normal(0,1,25)

# define a basic quadratic loss function
ql = rr.quadratic_loss(5, coef=1)
ql
# In this case Q is the identity matrix and beta is a 5 dimensional vector of input values
# L = 0.5 * beta' * beta 

# define a quadratic loss function with a shift
ql_shift = rr.quadratic_loss.shift(-Y, coef=1)
ql_shift
# Q is again the identity matrix
# L = 0.5 * (beta-Y)' * (beta-Y)

ql_shift_problem = rr.container(ql_shift)
ql_shift_solver = rr.FISTA(ql_shift_problem)
ql_shift_obj_vals = ql_shift_solver.fit()
ql_shift_solution = ql_shift_solver.composite.coefs

ql_shift_solution - (-Y)

##### OLS #######

# Now define a quadratic loss with an affine transformation which is then equal to the OLS Loss
# Q is again the identity matrix and can be neglected
# -Y is the offset
# X is the linear part
# L = 0.5 * (X*beta - Y)' (X*beta - Y)

# create simple OLS loss function with coef = 1/2 in front
ols_loss = rr.quadratic_loss.affine(X, -Y, coef=0.5)

# now create the final problem (simple OLS without an intercept term)
ols_problem = rr.container(ols_loss)

# select a solver
ols_solver = rr.FISTA(ols_problem)

# simply the values of the loss function
obj_vals = ols_solver.fit(max_its=100, tol=1e-5)

ols_solution = ols_solver.composite.coefs

ols_solution

xtx_inv = np.linalg.inv(np.dot(X.T, X))  
xty = np.dot(X.T, Y)
np.dot(xtx_inv, xty)

# calculate the predicted values
Y_pred = np.dot(X, ols_solution)

# calculate the loss (MSE)
ols_loss # C/2 = 0.25
0.25 * np.sum((Y - Y_pred)**2)

rr.quadratic_loss(5, coef=1)


##### Ridge Regression #####

ridge_loss = rr.quadratic_loss.affine(X, -Y, coef=0.5)
grouping = rr.quadratic_loss(2, coef=1) # shrink squared coefficients

ridge_problem = rr.container(ridge_loss, grouping)

ridge_solver = rr.FISTA(ridge_problem)

ridge_obj_vals = ridge_solver.fit(max_its=100, tol=1e-5)
ridge_solution = ridge_solver.composite.coefs
print(ridge_solution)

# compare this to the statsmodels implementation
stat_ridge = regression.linear_model.OLS(endog=Y, exog=X)
stat_ridge_fit = stat_ridge.fit_regularized(alpha=np.array([0,1]) / 25, L1_wt=0) 

print(stat_ridge_fit.params)

