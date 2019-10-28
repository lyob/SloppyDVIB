## libraries
import numpy as np
import timing

### global variables ###
# parameters -----------
# beta = 0

# paths
loadpath = '/project2/sepalmer/blyo/data-fim/'
outpath = loadpath

def load_from_npy(filename):
    object = np.load( loadpath + filename )
    return object

def add(x,y):
    return np.add(x,y)

def export_fim(matrix, beta):
    np.save(outpath+'fim-b{}.npy'.format(beta), matrix)
    print("The FIM matrix has been exported as fim-b{}.npy".format(beta))


#-----------------------------------------------------------------------------------------
if __name__ == '__main__':

    for i in range(13):
        beta = i
        print('New beta = {}'.format(beta))
        firstfim = load_from_npy('b{}-fim1.npy'.format(beta))
        secondfim = load_from_npy('b{}-fim2.npy'.format(beta))
        fimtotal = add(firstfim, secondfim)
        print("The first and second FIM matrices have been added together")

        export_fim(fimtotal, beta)
        print('End of beta = {}'.format(beta))
    
