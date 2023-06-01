import numpy as np
import pandas as pd
import time
import pylab
from scipy import sparse

def impute_missing_timestamps(df, entity = 'firm', time = 'time'):

    df = df.copy()

    # Create a new multi-index DataFrame with all combinations of 'firm' and 'time'
    multi_index = pd.MultiIndex.from_product([df[entity].unique(), 
                                              np.arange(df[time].max() + 1)],
                                             names=[entity, time])

    # Reindex the original DataFrame to align with the new multi-index
    df.set_index([entity, time], inplace=True)
    df = df.reindex(multi_index)

    # Fill NaN values with zeros
    df.fillna(0, inplace=True)

    # Reset the index to return to the original format
    df.reset_index(inplace=True)

    return df

def create_design_matrix(df, entity = 'firm', time = 'time', attr = ['attr1', 'attr2']):

    # Sort values by entity and time
    df.sort_values(by=[entity, time], inplace=True)

    # Group by entity
    grouped = df.groupby(entity)

    # Initialize an empty list to store the individual firm matrices
    firm_matrices = []

    # Iterate over groups
    for name, group in grouped:
        # name = name of firm (0, 1)
        # group = df filtered for each name

        # Create a diagonal matrix for each attribute and stack them horizontally
        attr_matrices = [sparse.diags(group[x].values) for x in attr]
        firm_matrix = sparse.hstack(attr_matrices)
        
        # Add the firm matrix to the list of firm matrices
        firm_matrices.append(firm_matrix)

    # Stack all the firm matrices vertically to create the design matrix
    design_matrix = sparse.vstack(firm_matrices)

    return sparse.csr_matrix(design_matrix)