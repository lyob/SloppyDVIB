## libraries
import numpy as np
import timing



### global variables ###
# parameters -----------
beta = 7

# paths
outpath = '/project/sepalmer/blyo/fim-output/beta{}/'.format(beta)
npypath = outpath


def load_from_npy(filename):
    object = np.load( npypath + filename )
    return object


def add(x,y):
    return np.add(x,y)

def export_fim(matrix):
    np.save(npypath+'fim-b{}.npy'.format(beta), matrix)
    print("The FIM matrix has been exported as fim-b{}.npy".format(beta))


#-----------------------------------------------------------------------------------------
if __name__ == '__main__':

    firstfim = load_from_npy('fim1-b{}.npy'.format(beta))
    secondfim = load_from_npy('fim2-b{}.npy'.format(beta))
    fimtotal = add(firstfim, secondfim)
    print("The first and second FIM matrices have been added together")

    export_fim(fimtotal)

    
