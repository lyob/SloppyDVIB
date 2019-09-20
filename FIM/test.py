
import sys
import os
own_name = sys.argv[0]
tup = os.path.splitext(own_name)
out = tup[0][3:4]
print(out)


filepath = os.path.dirname(os.path.abspath(__file__))
if 'post' in filepath:
    print('fim is in filepath')
else:
    print('fim is not in filepath')

#%%
import numpy as np
a = np.array([])
b = np.array([[3,4,5], [7,7,7]])
c = np.array([1,2,3])

a = np.vstack((b, c))

print(a)

if a.size > 0:
    print('yes')
else:
    print('no')
#%%
