from joblib import Parallel, delayed
import numpy as np

job_num = 10

def test():
    a = np.zeros(1000)
    for i in range(1000):
        # print(np.random.rand())
        a[i] = np.random.rand()
    return a


data = Parallel(n_jobs=job_num)([delayed(test)() for i in range(job_num)])
data = np.array(data).flatten()
print(data.shape)
count = 0
for i in range(len(data)):
    for j in range(i + 1, len(data)):
        if data[i] == data[j]:
            count += 1
            break

print(count)
#
# a = 0.4958462
# b = 0.4958461
#
# if a == b:
#     print("same")
