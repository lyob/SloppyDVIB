import numpy as np
import matplotlib.pyplot as plt 
import itertools as it

# global parameters
beta = 0

# Paths - import the eigenvalues that are already sorted with sort.py
inPath = './eigenvalues-sorted/'
outPath = './plots/'

def import_array(b):
    return np.load(inPath + 'b{}sorted.npy'.format(b))

def sort(arr):
    return np.sort(arr, kind='heapsort')

def remove_negativity(x):
    return x[x > 0]

def logify(array):
    return np.log(array)

def plot_spectrum(eigs):
    ytop = np.array([])
    ybot = np.array([])

    x = [0.2, 0.8]
    
    for e in it.islice(eigs, eigs.size-10, None):
        y = [e, e]
        plt.plot(x,y,color='red')
        ytop = np.append(ytop, e)
    for e in it.islice(eigs, 0, eigs.size-10):
        y = [e, e]
        plt.plot(x,y,color='green')
        ybot = np.append(ybot,e)

    plt.ylabel('eigenvalues')

    # find max and min values of y
    ymax = np.amax(eigs)
    ymin = np.amin(eigs)
    buffer = (ymax - ymin) * 0.1

    # set x and y range
    plt.xlim([0,1])
    plt.ylim([ymin - buffer, ymax + buffer])

    # remove x-axis label
    plt.gca().xaxis.set_major_locator(plt.NullLocator())

    # save plot
    plt.savefig(outPath + 'b{}.png'.format(beta)) 


    print("Number of elements in ytop:", ytop.size)
    print("Number of elements in ybot:", ybot.size)

    # show plot
    # plt.show()
    

if __name__ == "__main__":
    eigs = import_array(beta)
    # eigs = sort(eigs)
    eigs = remove_negativity(eigs)
    eigs = logify(eigs)
    plot_spectrum(eigs)

