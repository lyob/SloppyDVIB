## libraries
import numpy as np
import timing
import os
import sys

### global variables ###
# parameters -----------
own_name = os.path.splitext(sys.argv[0])
own_name = own_name[0][14:]
print('file name is {}'.format(own_name))
beta  = own_name[6:]

filepath = os.path.dirname(os.path.abspath(__file__))
basepath = filepath[:-11]
model = basepath[-8:] # l7Xindep

# paths
loadpath = '/dali/sepalmer/blyo/'+model+'/data-fim/'
outpath = loadpath

def load_from_npy(filename):
    object = np.load( loadpath + filename )
    return object

def add(x,y):
    return np.add(x,y)

def export_fim(matrix):
    np.save(outpath+'fim-b{}.npy'.format(beta), matrix)
    print("The FIM matrix has been exported as fim-b{}.npy".format(beta))


#-----------------------------------------------------------------------------------------
if __name__ == '__main__':


    firstfim = load_from_npy('b{}-fim1.npy'.format(beta))
    secondfim = load_from_npy('b{}-fim2.npy'.format(beta))
    fimtotal = add(firstfim, secondfim)
    print("The first and second FIM matrices have been added together")

    export_fim(fimtotal)
    print('The fim matrix has been exported.')

    
