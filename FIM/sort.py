# sorting the eigenvalues from smallest to largest 

# Paths
sortpath = '/project/sepalmer/blyo/sort-output/'


# global parameters
beta = 0

def import_array(b):
    return np.load('/project/sepalmer/blyo/fim-output/beta{}/b{}-eigenvalues.npy'.format(b, b))

def sort(arr):
    return np.sort(arr, kind='heapsort')

def export(filename, array):
    np.save( open(sortpath + filename, 'wb'), array)

def loop():
    for i in range(13):
        beta = i
        eigs = import_array(beta)
        s_eigs = sort(eigs)
        export('b{}sorted'.format(beta), s_eigs)



if __name__ == "__main__":
    loop()
