import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect

# set the maximum number of columns to be displayed to 100
pd.set_option('display.max_columns', 100)

# Load data
firm_attr = pd.read_csv('C:\\Users\\bs-lokal\\Desktop\\git_repos\\anmem_data\\data\\datashare.csv', 
                        delimiter=',')

stock_returns = pd.read_csv('C:\\Users\\bs-lokal\\Desktop\\git_repos\\anmem_data\\data\\mth_stock_returns.csv')

#########################
## Basic Preprocessing ##
#########################

# clean column names and date format
stock_returns.head()
stock_returns.drop('Unnamed: 0', inplace=True, axis=1)
stock_returns = stock_returns.rename(columns={'KYPERMNO': 'permno', 'MCALDT': 'date'})

firm_attr.columns = firm_attr.columns.str.lower()
stock_returns.columns = stock_returns.columns.str.lower()

firm_attr['date'] = pd.to_datetime(firm_attr['date'], format='%Y%m%d')
stock_returns['date'] = pd.to_datetime(stock_returns['date'])

# Check for matches
firm_attr['permno'].isin(stock_returns['permno']).all()

firm_attr[['permno', 'date']].duplicated().any()
stock_returns[['permno', 'date']].duplicated().any()

# visualize the number of stocks per date
stocks_per_month_raw = firm_attr.groupby('date')['permno'].nunique().reset_index()
avg_stocks_per_month_raw = stocks_per_month_raw.permno.mean()

plt.plot(stocks_per_month_raw.date, stocks_per_month_raw.permno)
plt.axhline(y = avg_stocks_per_month_raw, color='r', label = str(round(avg_stocks_per_month_raw, 2)))
plt.title('Number of stocks per month (raw data)')
plt.legend()
plt.show()

## merge data ##
df = pd.merge(firm_attr, stock_returns[['permno', 'date', 'mret']], on=['permno', 'date'], how='left')

firm_attr.shape
df.shape

###################################################
## Filter accoring to 'A new model every month?' ##
###################################################

# the authors used the following characteristics
final_attr = ['absacc', 'acc', 'age', 'agr',
              'baspread', 'beta', 'betasq', 'bm', 'bm_ia',
              'cashdebt', 'cashpr', 'cfp', 'cfp_ia', 'chatoia', 'chempia',
              'chcsho', 'chinv', 'chmom', 'chpmia', 'convind', 'currat',
              'depr', 'divi', 'divo', 'dolvol', 'dy', 'egr', 'ep', 'gma', 'herf',
              'hire', 'idiovol', 'ill', 'indmom', 'invest', 'lgr',
              'lev', 'maxret', 'mom12m', 'mom1m', 'mom36m', 'mom6m', 'mvel1', 'mve_ia', 'operprof',
              'pchcapx_ia', 'pchcurrat', 'pchdepr', 'pchgm_pchsale', 'pchquick',
              'pchsale_pchrect', 'pctacc', 'pricedelay', 'ps', 'quick', 'rd', 'retvol',
              'roic', 'salecash', 'salerec', 'sgr', 'sin', 'sp',
              'std_dolvol', 'std_turn', 'tang', 'tb', 'turn', 'zerotrade']

len(final_attr)
# all columns are present in the dataframe
pd.Index(final_attr).isin(df.columns).all()

## remove firm characteristics based on relative frequency of missing values ##

# threshold: max 37.5%
threshold = 0.375

filled_values = df.count()
missing_values = len(df) - filled_values
missing_percent = (missing_values / len(df)) 

(missing_percent >= threshold).sum()

# remove columns 
cols_to_keep = missing_percent[missing_percent <= threshold].index.tolist()

# what's in df_reduced but not in the final_attr
set(cols_to_keep).difference(set(final_attr))

df_reduced = df[cols_to_keep]

# next remove columns which are not used by the authors:

# Secured debt indicator (securedind) and Standard Industrial Classificaition (sic2) are not used 
missing_percent[missing_percent.index == 'securedind']
missing_percent[missing_percent.index == 'sic2']

df_reduced = df_reduced.loc[:, ~df_reduced.columns.isin(['securedind', 'sic2'])]
set(df_reduced.columns).difference(set(final_attr))


## keep only those stocks that have data on all firm characteristics in a given month ##

df_reduced = df_reduced.dropna()

# now additionally require to have at least 1000 stocks per month
stocks_per_month_reduced = df_reduced.groupby('date')['permno'].nunique().reset_index()
filt_dates = stocks_per_month_reduced[stocks_per_month_reduced['permno'] >= 1000]

df_reduced = df_reduced[df_reduced.date.isin(filt_dates.date)]

# finally restrict the data so that June 2019 is the last date
df_reduced = df_reduced[df_reduced.date < '2019-07-01']

# visualize the number of stocks per date again
stocks_per_month_reduced = df_reduced.groupby('date')['permno'].nunique().reset_index()
avg_n_stocks_reduced = stocks_per_month_reduced.permno.mean()

plt.plot(stocks_per_month_reduced.date, stocks_per_month_reduced.permno)
plt.axhline(y = avg_n_stocks_reduced, color='r', label = str(round(avg_n_stocks_reduced, 2)))
plt.title('Number of stocks per month (filtered data)')
plt.legend()
plt.show()

# check some summary statistics:
df_reduced.shape # more than 1,600,000 firm-month observations

# timeline    
(df_reduced.date.min(), df_reduced.date.max())

# number of stocks
df_reduced.permno.nunique() # ~15,000 stocks

# avgerage number of stocks per month
avg_n_stocks_reduced # more than 2,800 stocks per month on average



df_reduced.shape
df.shape

###################
## Preprocessing ##
###################

# apply implicitly passes all the columns for each group as a DataFrame to the custom function.
# transform passes each column for each group individually as a Series to the custom function.

# cross-sectional rank transformation and normalization
def mad(x):
    return np.sum(np.abs(x - np.mean(x)))

def rank_norm_trafo(df, meta_cols, gr_cols):
    
    # make a copy of the input dataframe to avoid modifying the original
    df = df.copy()

    cols_rank = df.columns.difference(meta_cols)
    
    # rank columns by using average ranking for ties
    df[cols_rank] = df.groupby(gr_cols)[cols_rank].transform(lambda x: x.rank(method='average'))
    
    # normalize all ranks
    df[cols_rank] = df.groupby(gr_cols)[cols_rank].transform(lambda x: x / (x.max() + 1))
    
    # normalize ranke-transformed data by centering and sclaing by MAD
    #tmp = tmp.apply(lambda x: (x - x.mean()) / x.mad())

    #df[cols_rank] = df.groupby(gr_cols)[cols_rank].transform(lambda x: (x-x.mean()) / mad(x))
    df[cols_rank] = df.groupby(gr_cols)[cols_rank].transform(lambda x: (x-x.mean()) / x.std())
   
     # get the right order back
    return df.reset_index(drop=True)


df_prep_mad = rank_norm_trafo(df = df_reduced, meta_cols = ['date', 'permno', 'mret'], gr_cols = ['date'])
df_prep_sd = rank_norm_trafo(df = df_reduced, meta_cols = ['date', 'permno', 'mret'], gr_cols = ['date'])


# store the data
#df_reduced.to_csv('df_reduced.csv', index=False)
#df_prep_mad.to_csv('df_prep_mad.csv', index=False)
#df_prep_sd.to_csv('df_prep_sd.csv', index=False)


df_reduced.to_pickle('df_reduced.pkl')
df_prep_mad.to_pickle('df_prep_mad.pkl')
df_prep_sd.to_pickle('df_prep_sd.pkl')

