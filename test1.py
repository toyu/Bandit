from joblib import Parallel, delayed
import numpy as np


def test():
    return np.random.rand()

data = Parallel(n_jobs=2)([delayed(test)() for i in range(2)])

print(data)