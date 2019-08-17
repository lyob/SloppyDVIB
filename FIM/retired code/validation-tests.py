#%%
for i in range(10,20):
    print(i)

#%%
def test(number):
    if number in range(10,20):
        indx = number - 10
    print(indx)


#%%
# assigning isn't the same as pointer referencing!
mat = {}
mat['x1y1'] = 12
mat['x2y2'] = mat['x1y1']
mat['x1y1'] *= 2
print(mat['x2y2'])


#%%

import numpy as np
def convert(layer_index, yrange):
    xval = int(np.floor(layer_index / yrange))
    yval = layer_index - yrange * xval
    return xval, yval

print(convert(3199, 32))
testarr =  []
