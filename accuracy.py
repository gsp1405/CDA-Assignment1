import numpy as np
import statistics

print('Numpy imported')

def accuracy(y_hat, y_val):
    dev = (y_val - y_hat)/y_val
    # Exclude infinite (division by 0 and NaN)
    dev = list(filter(lambda dev: (dev != -np.inf) & (dev != np.inf) & (~np.isnan(dev)), dev))

    acc = 1 - np.abs(dev)
    
    return statistics.mean(acc)