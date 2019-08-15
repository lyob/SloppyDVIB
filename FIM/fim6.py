#%%
## libraries
import numpy as np
import scipy
from memory_profiler import profile
#%%
# paths
savePath = './output/'

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

    export_npy('eigenvalues.npy', w)
    export_npy('eigenvectors.npy', v)

    export_csv('eigenvalues.csv', w)
    export_csv('eigenvectors.csv', v)

    print( '------ The eigenvectors and eigenvalues of the FIM matrix have been saved ------' )

#%%
