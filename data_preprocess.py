import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

path_to_file_prefix = 'data_options/spx_eod_2022/spx_eod_2022'

calls = pd.DataFrame()

for i in range(1,13):
    if i < 10:
        path_to_file =  path_to_file_prefix + '0' + str(i) + '.csv'
    else:
        path_to_file = path_to_file_prefix + str(i) + '.csv'
    #print(path_to_file)

    monthly_eod = pd.read_csv(path_to_file)
    monthly_eod.dropna(inplace=True)

    calls_monthly = monthly_eod.filter(items=['[QUOTE_UNIXTIME]', '[UNDERLYING_LAST]', '[STRIKE]', '[STRIKE_DISTANCE]', '[STRIKE_DISTANCE_PCT]', '[EXPIRE_UNIX]', '[DTE]', '[C_DELTA]', '[C_GAMMA]', '[C_VEGA]', '[C_THETA]', '[C_RHO]', '[C_IV]', '[C_VOLUME]', '[C_LAST]', '[C_SIZE]', '[C_BID]', '[C_ASK]'])
    
    calls = pd.concat([calls, calls_monthly], ignore_index=True)
    
    print(calls_monthly.shape)
    #print(calls_df.dtypes)
    #print(path_to_file[-20:])

# Converting DTE to integer
calls['[DTE]'] = calls['[DTE]'].round()
calls['[DTE]'] = calls['[DTE]'].astype(np.int64)

# Converting STRIKE and VOLUME to integer
# Drop Strike Distance because Strike is there.
calls = calls.drop('[STRIKE_DISTANCE]', axis=1)
calls['[STRIKE]'] = calls['[STRIKE]'].astype(np.int64)
calls['[C_VOLUME]'] = calls['[C_VOLUME]'].astype(np.int64)

# Splitting SIZE attribute and converting to int.
calls[['[C_SIZE1]', '[C_SIZE2]']] = calls['[C_SIZE]'].str.split(' x ', expand=True)
calls['[C_SIZE1]'] = calls['[C_SIZE1]'].astype(np.int64)
calls['[C_SIZE2]'] = calls['[C_SIZE2]'].astype(np.int64)
# Drop Size because Size has been split into 2 columns above.
calls = calls.drop('[C_SIZE]', axis=1)



# Drop redundant colunmns. DTE can serve the same purpose 
#calls = calls.drop('[QUOTE_UNIXTIME]', axis=1)
#calls = calls.drop('[EXPIRE_UNIX]', axis=1)

print(calls.shape)
print(calls.columns)

cols = ['[UNDERLYING_LAST]', '[STRIKE]', '[STRIKE_DISTANCE_PCT]', '[DTE]', '[C_DELTA]',
       '[C_GAMMA]', '[C_VEGA]', '[C_THETA]', '[C_RHO]', '[C_VOLUME]', '[C_SIZE1]', '[C_SIZE2]',
       '[C_BID]', '[C_ASK]', '[QUOTE_UNIXTIME]', '[EXPIRE_UNIX]', '[C_LAST]', '[C_IV]']
calls = calls[cols]

#### Converting to X as training data
# y=C_IV
# z=C_LAST_df for later evaluation
X_num = calls.iloc[:,:-1]
y = calls.iloc[:,-1:]

X_num_train, X_num_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=1)
X_num_train, X_num_val, y_train, y_val = train_test_split(X_num_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

z_train = X_num_train.iloc[:, -3:]
z_test = X_num_test.iloc[:, -3:]
z_val = X_num_val.iloc[:, -3:]

X_num_train = X_num_train.iloc[:, :-3]
X_num_test = X_num_test.iloc[:, :-3]
X_num_val = X_num_val.iloc[:, :-3]

print(X_num_test.dtypes)

# Coverting Dataframes to .npy for better storage.
X_num_train = X_num_train.to_numpy()
X_num_test = X_num_test.to_numpy()
X_num_val = X_num_val.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
y_val = y_val.to_numpy()

z_train = z_train.to_numpy()
z_test = z_test.to_numpy()
z_val = z_val.to_numpy()

# Save the Numpy arrays for training.
with open('tab-ddpm/tab-ddpm-main/data/options/X_num_train.npy', 'wb') as f:
    np.save(f, X_num_train)
with open('tab-ddpm/tab-ddpm-main/data/options/X_num_test.npy', 'wb') as f:
    np.save(f, X_num_test)
with open('tab-ddpm/tab-ddpm-main/data/options/X_num_val.npy', 'wb') as f:
    np.save(f, X_num_val)

with open('tab-ddpm/tab-ddpm-main/data/options/y_train.npy', 'wb') as f:
    np.save(f, y_train)
with open('tab-ddpm/tab-ddpm-main/data/options/y_test.npy', 'wb') as f:
    np.save(f, y_test)
with open('tab-ddpm/tab-ddpm-main/data/options/y_val.npy', 'wb') as f:
    np.save(f, y_val)

with open('tab-ddpm/tab-ddpm-main/data/options/z_train.npy', 'wb') as f:
    np.save(f, z_train)
with open('tab-ddpm/tab-ddpm-main/data/options/z_test.npy', 'wb') as f:
    np.save(f, z_test)
with open('tab-ddpm/tab-ddpm-main/data/options/z_val.npy', 'wb') as f:
    np.save(f, z_val)

print('X y and z are saved in data/options', X_num_train.shape, z_train.shape)

