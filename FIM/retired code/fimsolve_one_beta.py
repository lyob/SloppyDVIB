# This code takes in the FIM matrix and outputs its eigenvectors and eigenvalues as npy files

#%%
## libraries
import numpy as np
import scipy
from memory_profiler import profile
#%%
# paths
beta = 4
savePath = './output/beta{}/'

# loading matrix array
def load_matrix():
    fim = np.load( savePath + 'fim.npy' )
    return fim


# calculating the eigenvalues and eigenvectors of the symmetric square matrix
@profile
def diagonalise(fim):
    from scipy.linalg import eigh
    print('------ calculating the eigenvectors and eigenvalues of the FIM ------')
    w, v = eigh(fim)
    return w, v
# results = {"eigenvalues": w, 'eigenvectors': v}

def export_npy(output_name, array):
    np.save(savePath + output_name, array)
    print("The array has been exported as {}".format(output_name))

def export_csv(output_name, array):
    np.savetxt(savePath + output_name, array, delimiter=',')
    print("The array has been exported as {}".format(output_name))

#%%
if __name__ == "__main__":
    fim = load_matrix()
    w, v = diagonalise(fim)
    print( '------ The matrix has been successfully diagonalised ------')

    export_npy('b{}-eigenvalues.npy'.format(beta), w)
    export_npy('b{}-eigenvectors.npy'.format(beta), v)

    export_csv('b{}-eigenvalues.csv'.format(beta), w)
    export_csv('b{}-eigenvectors.csv'.format(beta), v)

    print( '------ The eigenvectors and eigenvalues of the FIM matrix have been saved ------' )

#%%
