def var_over_time(process: np.array) -> np.array:
    var_func = []
    
    for i in range(len(process)):
        var_func.append(np.var(process[:i]))
    
    return var_func

def mean_over_time(process: np.array) -> np.array:
    mean_func = []
    
    for i in range(len(process)):
        mean_func.append(np.mean(process[:i]))
    
    return mean_func


# ----
# stationarity
from statsmodels.tsa.stattools import adfuller
ADF_result = adfuller(df['values'])

print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

#differencing

df_diff = np.diff(df['values'], n=1)
#then do adf on it

#-----
