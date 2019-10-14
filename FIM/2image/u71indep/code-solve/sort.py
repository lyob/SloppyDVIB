# sorting the eigenvalues from smallest to largest 
import numpy as np
import os
import timing

# /home/blyo/python_tests/l7Xindep/code-solve
filepath = os.path.dirname(os.path.abspath(__file__))
basepath = filepath[:-11] # /home/blyo/python_tests/l7Xindep
model = basepath[-8:] # l7Xindep

# Paths
dalipath = '/dali/sepalmer/blyo/2image/'
sortpath = dalipath+model+'/data-eigs/'

def import_array(b):
    return np.load(sortpath + 'b{}-evalues.npy'.format(b))

def sort(arr):
    return np.sort(arr, kind='heapsort')

def export(filename, array):
    np.save( open(sortpath + filename, 'wb'), array)

def loop():
    for i in range(13):
        beta = i
        print('new beta = {}'.format(beta))
        eigs = import_array(beta)
        print(eigs.shape)
        s_eigs = sort(eigs)
        export('sorted-b{}.npy'.format(beta), s_eigs)

if __name__ == "__main__":
    loop()
