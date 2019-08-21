import numpy as np
import pandas as pd

a = np.array([1,2,0,3,0,4])
for i in np.where(a==0)[0]:
    a[i] =6
print(a)
