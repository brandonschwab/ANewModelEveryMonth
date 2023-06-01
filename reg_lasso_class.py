import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import inspect
import time
import regreg.api as rr
import pylab
from scipy import sparse

class reg_lasso:

    def __init__(self, df):
        self.df = df.copy()
        self.df_imp = None
        self.features = []
        self.meta_attr = ['permno', 'mret', 'date', 'time', 'month', 'year']
        self.X = None
        self.target = 'mret'
        self.obj_vals = None
        self.betas = None
        self.J = len(self.features)
        self.T = None
        self.P = None
        self.D = None

    def create_timestamps(self, date_column, entity='permno') :
        # Convert the 'date' column to datetime
        self.df[date_column] = pd.to_datetime(self.df[date_column])

        # Create 'year' and 'month' columns based on the 'date' column
        self.df['year'] = self.df[date_column].dt.year
        self.df['month'] = self.df[date_column].dt.month

        min_year = self.df['year'].min()
        min_month = self.df[self.df['year'] == min_year]['month'].min()

        # Calculate the time relative to the origin
        self.df['time'] = (self.df['year'] - min_year) * 12 + (self.df['month'] - min_month)

        self.df.sort_values(by=[entity, 'time'], inplace=True)

        self.T = self.df.time.max() + 1

    def impute_missing_timestamps(self, entity='firm', time='time'):

        self.df_imp = self.df.copy()

        self.df_imp.sort_values(by=[entity, time], inplace=True)

        # Create a new multi-index DataFrame with all combinations of 'firm' and 'time'
        multi_index = pd.MultiIndex.from_product([self.df_imp[entity].unique(),
                                                    np.arange(self.df_imp[time].max() + 1)],
                                                    names=[entity, time])
        
        # Reindex the original DataFrame to align with the new multi-index
        self.df_imp.set_index([entity, time], inplace=True)
        self.df_imp = self.df_imp.reindex(multi_index)

        # Fill NaN values with zeros
        self.df_imp.fillna(0, inplace=True)

       # Reset the index to return to the original format
        self.df_imp.reset_index(inplace=True)

        self.df_imp.sort_values(by=[entity, time], inplace=True)

    def set_features(self):

        attr = list(self.df.columns)
        self.features = [col for col in attr if col not in self.meta_attr]
        print(len(self.features), 'features identified')
        self.J = len(self.features)


    def create_design_matrix(self, entity='firm', time='time', rm_zeros=True):

        # Sort values by entity and time
        self.df_imp.sort_values(by=[entity, time], inplace=True)

        # Group by entity
        grouped = self.df_imp.groupby(entity)

        # Initialize an empty list to store the individual firm matrices
        firm_matrices = []

        # Iterate over groups
        for name, group in grouped:
            # name = name of firm (0, 1)
            # group = df filtered for each name

            # Create a diagonal matrix for each attribute and stack them horizontally
            attr_matrices = [sparse.diags(group[x].values) for x in self.features]
            firm_matrix = sparse.hstack(attr_matrices)
        
            # Add the firm matrix to the list of firm matrices
            firm_matrices.append(firm_matrix)

        # Stack all the firm matrices vertically to create the design matrix
        design_matrix = sparse.vstack(firm_matrices)

        X = sparse.csr_matrix(design_matrix)

        if rm_zeros:
            # remove zero entries
            nonzero_rows = X.getnnz(axis=1).ravel() > 0
            X = X[nonzero_rows, :]

        self.X = X
        self.P = int(self.X.shape[1])

    def create_D(self):

        D = (np.identity(self.P) + np.diag([-1]*(self.P-1),k=1))[:-1]

        row_positions = np.arange(self.T, self.P, self.T) -1
        zero_row = np.zeros((1, D.shape[1]))
        D[row_positions] = zero_row
        self.D = sparse.csr_matrix(D)

    
    def define_problem(self, target=None, lambda_lasso=2.26, lambda_fused=3.1):

        if target is None: target = self.target

        # target
        Y = np.array(self.df[target].values)
        # number of coefficients
        P = int(self.X.shape[1])
        # number of targets
        n = len(Y)

        # define quadratic loss (Sum of Squared Residuals)
        loss = rr.quadratic_loss.affine(self.X, -Y, coef=0.5)

        # define lasso loss
        sparsity = rr.l1norm(self.P, lagrange=lambda_lasso)

        # create fused loss
        fused = rr.l1norm.linear(self.D, lagrange=lambda_fused)

        # define problem
        problem = rr.container(loss, sparsity, fused)

        self.problem = problem
    
    def fit(self, max_its=1000, tol=1e-5):

        solver = rr.FISTA(self.problem)
        self.obj_vals = solver.fit(max_its=max_its, tol=tol)
        self.betas = solver.composite.coefs

    def check_convergence(self):

        plt.plot(self.obj_vals)
        diffs = np.abs(np.diff(self.obj_vals[-10:]))  # Last 10 differences
        print(diffs)
    
    def structure_coefs(self, J=None, T=None):

        if J is None: J = self.J
        if T is None: T = self.T

        betas = list(self.betas)
        chunks = [betas[i:i+T] for i in range(0, len(betas), T)]
        coefs = pd.DataFrame({'time': range(T)})

        for i in range(J):
            coefs[self.features[i]] = chunks[i]

        return coefs  
        





