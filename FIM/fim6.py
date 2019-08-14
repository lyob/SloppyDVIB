## libraries
import numpy as np
import scipy
import pickle

# paths
savePath = './output/'

# loading pickle files
mfim = pickle.load( open('fim5.p', 'rb') )

#%%
# calculating the eigenvalues and eigenvectors of the matrix
from scipy.linalg import eigh
print('----------------calculating the eigenvectors and eigenvalues of the FIM---------------')
w, v = eigh(mfim)

results = {"eigenvalues": w, 'eigenvectors': v}

#%%
# writes output to a file
w_file = savePath+'eigenvalues.csv'
v_file = savePath+'eigenvectors.csv'

np.savetxt(w_file, w, delimiter=',')
np.savetxt(v_file, v, delimiter=',')

pickle.dump( results, open('fim6.p', 'wb') )
print( '------ The eigenvectors and eigenvalues of the FIM matrix have been pickled as fim6.p ------' )