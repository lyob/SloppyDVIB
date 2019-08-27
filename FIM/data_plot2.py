import numpy as np
import matplotlib.pyplot as plt 
import itertools as it

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

def plot_spectrum(eigs, beta, *args):
    ytop = np.array([])
    ybot = np.array([])

    # total x axis
    t = linspace(0, 13)

    # set x value dependent on beta
    x = [beta + 0.2, beta + 0.8]
    
    for e in it.islice(eigs, eigs.size-10, None):
        y = [e, e]
        plt.plot(x,y,color='red')
        ytop = np.append(ytop, e)
    for e in it.islice(eigs, 0, eigs.size-10):
        plt.plot(x,y,color='green')
        ybot = np.append(ybot,e)

    plt.ylabel('eigenvalues')

    # find max and min values of y
    if (ymax == None & ymin == None):
        # initial set
        ymax = np.amax(eigs)
        ymin = np.amin(eigs)
        buffer = (ymax - ymin) * 0.1
    else:
        # set new ymax and ymin if you find larger/smaller values
        if ymax < np.amax(eigs):
            ymax = np.amax(eigs)
        if ymin > np.amin(eigs):
            ymin = np.amin(eigs)

    # set x and y range
    # plt.xlim([0,1])
    plt.ylim([ymin - buffer, ymax + buffer])

    # remove x-axis label
    plt.gca().xaxis.set_major_locator(plt.NullLocator())

    # save plot
    plt.savefig(outPath + 'b{}.png'.format(beta)) 

    # print("Number of elements in ytop:", ytop.size)
    # print("Number of elements in ybot:", ybot.size)

    # show plot
    # plt.show()
    return ymin, ymax
    
def plot_all_thirteen():
    ymin, ymax = None, None
    for i in range(13):
        beta = i
        eigs = import_array(beta)
        # eigs = sort(eigs)
        eigs = remove_negativity(eigs)
        eigs = logify(eigs)
        ymin, ymax = plot_spectrum(eigs, beta, ymin, ymax)

if __name__ == "__main__":
    plot_all_thirteen()

