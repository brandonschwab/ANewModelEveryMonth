import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

attr_grouped = ['date', 'quick', 'pricedelay', 'pchquick', 'pchdepr', 'pchcurrat', 'hire', 'herf',
                'divo', 'divi', 'depr', 'currat', 'convind', 'cashdebt', 'age',
                'zerotrade', 'turn', 'std_turn', 'std_dolvol', 'retvol', 'mvel1', 'mve_ia',
                'maxret', 'ill', 'idiovol', 'dolvol', 'betasq', 'beta', 'baspread',
                'chempia', 'tang', 'rd', 'pctacc', 'acc', 'absacc', 'tb',
                'sp', 'sin', 'sgr', 'salerec', 'salecash', 'roic', 'pchsale_pchrect',
                'pchgm_pchsale', 'operprof', 'chpmia', 'gma', 'chatoia', 'cashpr', 'pchcapx_ia',
                'lgr', 'invest', 'egr', 'chinv', 'chcsho', 'agr', 'ps',
                'lev', 'ep', 'dy', 'cfp', 'cfp_ia', 'bm', 'bm_ia',
                'mom6m', 'mom36m', 'mom1m', 'mom12m', 'indmom', 'chmom']


# import the estimated coefficients and the raw dataset

with open('betas.pkl', 'rb') as file:
    betas = pickle.load(file)

with open('beta_ols.pkl', 'rb') as file:
    betas_ols = pickle.load(file)

with open('data//df_reduced.pkl', 'rb') as file:
    df = pickle.load(file)


# create dataframe for coefficents
attr = list(df.columns)
attr = [col for col in attr if col not in ['permno', 'mret', 'date', 'time', 'month', 'year']]
T = df.date.nunique()

coef_list = list(betas)
chunks = [betas[i:i+T] for i in range(0, len(betas), T)]
coefs = pd.DataFrame({'time': range(T)})
for i in range(len(attr)):
    coefs[attr[i]] = chunks[i]   

dates = np.sort(df.date.unique())
coefs.insert(1, 'date', dates)

# calculate the percentage of features (for each t) which are nonzero (>10^-11)
def calc_pct(row):
    count = np.sum(np.abs(row) >= 1e-11)
    total = float(len(row))
    return (count / total) 

non_sparsity_set = coefs[attr].apply(calc_pct, axis=1)

# plot the non-sparsity set over time
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib import colors

## raw data and 12month moving average ##

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(coefs.date, non_sparsity_set)
axs[1].plot(coefs.date, non_sparsity_set.rolling(window=12).mean())

years = mdates.YearLocator(5)
years_fmt = mdates.DateFormatter('%Y') # Year format

# apply same settings for both plots
for ax in axs:

    ax.xaxis.set_major_locator(years) # Apply the locator
    ax.xaxis.set_major_formatter(years_fmt) # Apply the format
    ax.set_ylim([0, 1])
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_title("The Nonsparsity Set Over Time")

    # List of tuples each containing start and end date of each recession
    recessions = [('1973-11-01', '1975-03-01'), 
                  ('1980-01-01', '1980-07-01'), 
                  ('1981-07-01', '1982-11-01'), 
                  ('1990-07-01', '1991-03-01'), 
                  ('2001-03-01', '2001-11-01'), 
                  ('2007-12-01', '2009-06-01')] # add more if needed

    # Create recession shades
    for start, end in recessions:
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='grey', alpha=0.5)

max_id = np.where(non_sparsity_set == non_sparsity_set.max())
coefs.date.iloc[max_id]

pd.DataFrame(non_sparsity_set).describe()

# plot the non-sparsity set over time at the category level
categories = {'past_return': 
              ['mom6m', 'mom36m', 'mom1m', 'mom12m', 'indmom', 'chmom'],
              'value':
              ['ps', 'lev', 'ep', 'dy', 'cfp', 'cfp_ia', 'bm', 'bm_ia'],
              'investment': 
              ['chatoia', 'cashpr', 'pchcapx_ia', 'lgr', 'invest', 'egr',
               'chinv', 'chcsho', 'agr'],
              'profitability':
              ['tb', 'sp', 'sin', 'sgr', 'salerec', 'salecash', 'roic',
               'pchsale_pchrect', 'pchgm_pchsale', 'operprof', 'chpmia', 'gma'],
              'intangibles':
              ['chempia', 'tang', 'rd', 'pctacc', 'acc', 'absacc'],
              'trading_friction':
              ['zerotrade', 'turn', 'std_turn', 'std_dolvol', 'retvol',
               'mvel1', 'mve_ia', 'maxret', 'ill', 'idiovol', 'dolvol',
               'betasq', 'beta', 'baspread'],
              'other':
              ['quick', 'pricedelay', 'pchquick', 'pchdepr', 'pchcurrat', 'hire',
               'herf', 'divo', 'divi', 'depr', 'currat', 'convind', 'cashdebt', 'age']}

fig, axs = plt.subplots(7, 1, figsize=(12, 16))
years = mdates.YearLocator(5)
years_fmt = mdates.DateFormatter('%Y') # Year format

for i in range(7):
    
    cat_attr = categories.values()[i]
    non_sparsity = coefs[cat_attr].apply(calc_pct, axis=1)

    axs[i].plot(coefs.date, non_sparsity.rolling(window=12).mean())
    axs[i].set_title(categories.keys()[i])


# apply same settings for both plots
for ax in axs:

    ax.xaxis.set_major_locator(years) # Apply the locator
    ax.xaxis.set_major_formatter(years_fmt) # Apply the format
    ax.set_ylim([0, 1])
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    # List of tuples each containing start and end date of each recession
    recessions = [('1973-11-01', '1975-03-01'), 
                  ('1980-01-01', '1980-07-01'), 
                  ('1981-07-01', '1982-11-01'), 
                  ('1990-07-01', '1991-03-01'), 
                  ('2001-03-01', '2001-11-01'), 
                  ('2007-12-01', '2009-06-01')] # add more if needed

    # Create recession shades
    for start, end in recessions:
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='grey', alpha=0.5)

plt.tight_layout()
plt.show()

## non-sparsity set at characteristic level ##

def encode_val(val):
    if abs(val) > 1e-11:
        return 1
    else:
        return 0

coefs_encode = coefs[attr].applymap(encode_val)
coefs_encode.insert(0, 'date', coefs.date)

# restructure the dataframe
coefs_encode = coefs_encode[attr_grouped]

# create an array with attribute data
data = coefs_encode.drop('date', axis=1).T.values

fig, ax = plt.subplots(figsize=(15,20)) 

# display the data
cax = ax.imshow(data, cmap='Greys', aspect=6)

# set the y ticks positions and labels
ax.set_yticks(np.arange(coefs_encode.shape[1]-1))
ax.set_yticklabels(coefs_encode.columns[1:])

# set the x ticks positions
idx = np.arange(0, len(dates), 60)
ax.set_xticks(idx)
ax.set_xticklabels(np.datetime_as_string(dates[idx], unit='Y'))

line_pos = [12, 26, 33, 47, 62]
for pos in line_pos:
    ax.axhline(pos+0.5, color='r', linestyle='--')

coefs_encode.mom1m.sum()/float(T)
coefs_encode.pchcurrat.sum()/float(T)

# color the values

data = coefs[attr_grouped].drop('date', axis=1)

# Set up a custom color map that goes from white to blue for negative values, and white to red for positive
cmap = colors.LinearSegmentedColormap.from_list("", ["red","white","blue"])

# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 10))

# Generate the image. Note: the aspect parameter can adjust the image's aspect ratio.
cax = ax.imshow(data.T, cmap=cmap, aspect=6, vmin=-0.02, vmax=0.02)

# Create the colorbar
cbar = plt.colorbar(cax, ax=ax)

# set the y ticks positions and labels
ax.set_yticks(np.arange(coefs.shape[1]-1))
ax.set_yticklabels(coefs.columns[1:])

# set the x ticks positions
idx = np.arange(0, len(dates), 60)
ax.set_xticks(idx)
ax.set_xticklabels(np.datetime_as_string(dates[idx], unit='Y'))

line_pos = [12, 26, 33, 47, 62]
for pos in line_pos:
    ax.axhline(pos+0.5, color='black', linestyle='--')

plt.tight_layout()
plt.show()


## Illustration of Fusion Penalization against OLS ##

attr_plot = ['mom1m', 'idiovol', 'mom12m', 'pchcurrat']

fig, axs = plt.subplots(2, 2, figsize=(12, 6))
a = 0
for i in range(2):
    for j in range(2):

        axs[i][j].plot(coefs.date, betas_ols[attr_plot[a]], color='grey')
        axs[i][j].plot(coefs.date, coefs[attr_plot[a]], color='red', linestyle='--')
        axs[i][j].set_title(attr_plot[a])
        a += 1

plt.tight_layout()
plt.show()     

## Clustering ##

import scipy.cluster.hierarchy as sch

X = coefs.drop(columns=['time', 'date'])

plt.figure(figsize=(12, 6))
l = sch.ward(X.abs().T.values)
dend = sch.dendrogram(l, labels=X.columns, leaf_rotation=90)
plt.title('Economic Importance of Characteristics - Clustering')
plt.ylabel('Euclidean distances')
plt.tight_layout()
plt.show()

## T-Tests ##
from scipy import stats

t_tests = X.apply(lambda x: stats.ttest_1samp(x, 0).statistic)
p_vals = X.apply(lambda x: stats.ttest_1samp(x, 0).pvalue)
avg_coefficient = X.apply(lambda x: np.mean(x) * 100)


t_stats = pd.DataFrame({'t_stat': abs(t_tests), 'p_val': p_vals, 'coef': avg_coefficient})
t_stats.sort_values('t_stat', ascending=False, inplace=True)
