import numpy as np
import matplotlib.pyplot as plt 
import itertools as it
import timing

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

def plot_spectrum(num_betas):
    ytop = np.array([])

    # total x axis
    t = np.linspace(0, num_betas+1)

    ymax = None
    ymin = None

    for i in range(num_betas+1):
        beta = i
        # set x value dependent on beta
        x = [beta + 0.2, beta + 0.8]
        eigs = import_array(beta)
        # eigs = sort(eigs)
        eigs = remove_negativity(eigs)
        eigs = logify(eigs)
    
        for e in it.islice(eigs, eigs.size-10, None):
            y = [e, e]
            plt.plot(x,y,color='red')
            ytop = np.append(ytop, e)
        for e in it.islice(eigs, 0, eigs.size-10):
            plt.plot(x,y,color='green')

        # find max and min values of y
        if (ymax == None and ymin == None):
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

        np.save( open(outPath+'10largest-b{}.npy'.format(beta), 'wb'), ytop )
        print('the 10 largest eigenvalues have been saved for beta = {}'.format(beta))

    print('ymax for all {} betas is {}'.format(num_betas, ymax))
    print('ymin for all {} betas is {}'.format(num_betas, ymin))

    # set x and y range
    # plt.xlim([0,1])
    plt.ylim([0, ymax + buffer])

    plt.ylabel('eigenvalues')

    # remove x-axis label
    plt.gca().xaxis.set_major_locator(plt.NullLocator())

    # save plot
    plt.savefig(outPath + 'b{}.png'.format(beta))
    print('the plot has been saved as b{}'.format(beta))

    # print("Number of elements in ytop:", ytop.size)
    # print("Number of elements in ybot:", ybot.size)

    # show plot
    # plt.show()

if __name__ == "__main__":
    # max beta num is 13
    num_betas = 13 
    plot_spectrum(num_betas)

