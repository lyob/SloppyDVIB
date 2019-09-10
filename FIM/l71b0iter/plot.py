import numpy as np
import matplotlib.pyplot as plt 
import itertools as it
import timing

# Paths - import the eigenvalues that are already sorted with sort.py
basePath = '.'
inPath = basePath+'/data-eigs/'
outPath = basePath+'/data-plots/'

def import_array(b):
    return np.load(inPath + 'sorted-b{}.npy'.format(b))

def remove_negativity(x):
    return x[x > 0]

def logify(array):
    return np.log(array)

def plot_spectrum(num_betas):

    # total x axis
    t = np.linspace(0, num_betas)

    ymax = None
    ymin = None
    ytopmin = None

    for i in range(num_betas):
        # set beta
        beta = i

        # array to hold dominant eig values
        ytop = np.array([])

        # x position dependent on beta
        x = [beta + 0.2, beta + 0.8]
        
        # import sorted eigenvalues
        eigs = import_array(beta)
        
        # remove all negative eigs
        eigs = remove_negativity(eigs)
        
        # log values
        eigs = logify(eigs)

        # get the largest X values for each beta
        num_eigs = 11
        for e in it.islice(eigs, eigs.size-num_eigs, None):
            ytop = np.append(ytop, e)

        ytopmax = np.amax(ytop)
        for e in ytop:
            # normalise the y values s.t. max(y) = 1.
            y = [e/ytopmax, e/ytopmax]
            plt.plot(x,y,color='red')
            
        #for e in it.islice(eigs, 0, eigs.size-10):
        #    plt.plot(x,y,color='green')

        # find max and min values of y
        if (ymax == None and ymin == None):
            # initial set
            ymax = np.amax(eigs)
            ymin = np.amin(eigs)
        else:
            # set new ymax and ymin if you find larger/smaller values
            if ymax < np.amax(eigs):
                ymax = np.amax(eigs)
            if ymin > np.amin(eigs):
                ymin = np.amin(eigs)
        if ytopmin != None:
            if ytopmin > np.amin(ytop):
                ytopmin = np.amin(ytop)
        else:
            ytopmin = np.amin(ytop)
        
    # the amount to have above/below the max/min y value
    # buffer = (ymax - ytopmin) * 0.1
    buffer = (1 - ytopmin) * 0.1
    print('ymax for all {} betas is {}'.format(num_betas, ymax))
    print('ymin for all {} betas is {}'.format(num_betas, ymin))

    # set x and y range
    # plt.ylim([ytopmin - buffer, ymax + buffer])
    plt.ylim([ytopmin - buffer, 1 + buffer])
    print('buffer = {}'.format(buffer))

    plt.ylabel('eigenvalues')

    # remove x-axis label
    plt.gca().xaxis.set_major_locator(plt.NullLocator())

    # save plot
    plt.savefig(outPath + 'allbeta-normalised.png')
    print('the plot has been saved as allbeta-normalised.png')

    # print("Number of elements in ytop:", ytop.size)
    # print("Number of elements in ybot:", ybot.size)

    # show plot
    # plt.show()

if __name__ == "__main__":
    # max beta num is 13
    num_betas = 13
    plot_spectrum(num_betas)

