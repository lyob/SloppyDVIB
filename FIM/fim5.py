## libraries
import numpy as np
import sys
import pickle

# paths
savePath = './output/'

# importing pickle file
fim = pickle.load( open('fim4.py', 'rb') )

# converting the dictionary into a matrix
def dict_to_matrix(d):
    print('-----------------converting the dictionary FIM into an array-----------------')
    mtx = np.zeros((20002,20002))
    for i in range(20002):
        for j in range(20002):
            mtx[i][j] = d["i"+str(i)+"j"+str(j)]
    return mtx

mfim = dict_to_matrix(fim)

mfim_file = savePath+'fim.csv'
np.savetxt(mfim_file, mfim, delimiter=',')

pickle.dump( mfim, open('fim5.py', 'wb') )
print('------ The FIM (array) has been pickled as fim5.p ------')