# This code takes in the FIM matrix and outputs its eigenvectors and eigenvalues as npy files

## libraries
import numpy as np
import scipy
import os
import sys

# /home/blyo/python_tests/l7Xindep/code-solve
filepath = os.path.dirname(os.path.abspath(__file__))

# /home/blyo/python_tests/l7Xindep
basepath = filepath[:-11]

# l7Xindep
model = basepath[-8:]


#%%
# paths
loadpath = '/dali/sepalmer/blyo/'+model+'/data-fim/'
outpath = '/dali/sepalmer/blyo/'+model+'/data-eigs/'

# loading matrix array
def load_matrix():
    fim = np.load( loadpath + 'fim-b{}.npy'.format(beta) )
    return fim

# calculating the eigenvalues and eigenvectors of the symmetric square matrix
# @profile
def diagonalise(fim):
    from scipy.linalg import eigh
    print('------ calculating the eigenvectors and eigenvalues of the FIM ------')
    w, v = eigh(fim)
    return w, v
# results = {"eigenvalues": w, 'eigenvectors': v}

def export_npy(output_name, array):
    np.save(outpath + output_name, array)
    print("The array has been exported as {}".format(output_name))

def export_csv(output_name, array):
    np.savetxt(outpath + output_name, array, delimiter=',')
    print("The array has been exported as {}".format(output_name))

#%%
if __name__ == "__main__":

    global beta

    for b in range(13):
        beta = b
        print('---------- New beta = {} ----------'.format(beta))
        
        fim = load_matrix()
        w, v = diagonalise(fim)
        print('------ The matrix has been successfully diagonalised ------')

        export_npy('b{}-evalues.npy'.format(beta), w)
        export_npy('b{}-vectors.npy'.format(beta), v)
        print('------ The eigenvectors and eigenvalues have been exported ------')

#%%
