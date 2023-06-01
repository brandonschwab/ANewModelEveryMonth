import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# import data
ts = pd.read_csv('C:\\Users\\bs-lokal\\Desktop\\Python_Stuff\\demo_ts.csv',
                 sep=';', index_col=False)

ts.drop('Unnamed: 0', inplace=True, axis=1)

# convert date column to proper date format
ts['date'] = pd.to_datetime(ts['date']).dt.date

# convert numeric columns properly
num_cols = pd.Index(ts.columns).drop('date')

for col in num_cols: 
    ts[col] = ts[col].str.replace(',', '.').astype(float)

# plot the time series
plt.plot(ts['date'], ts['Consumption'], label='Consumption')
plt.plot(ts['date'], ts['Income'], label='Income')
plt.xlabel('Date')
plt.ylabel('%Change')
plt.legend()
plt.grid(True)
plt.show()

# simple linear regression consumption ~ income
lm = sm.OLS(ts.Consumption, sm.add_constant(ts.Income))
lm_res = lm.fit()

print(lm_res.summary())


# plot the data with the regression line
reg_line = lm_res.params[0] + lm_res.params[1] * ts['Income']

plt.scatter(ts.Income, ts.Consumption)
plt.plot(ts.Income, reg_line, c='r', label='OLS')
plt.legend()
plt.xlabel('Income')
plt.ylabel('Consumption')
plt.show()





