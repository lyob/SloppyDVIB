import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as ticker
import itertools as it
import timing

# Paths - import the eigenvalues that are already sorted with sort.py
basePath = '.'
inPath = basePath+'/eigenvalues-sorted/'
outPath = basePath+'/data-plots-l71/'

def import_array(b):
    return np.load(inPath + 'b{}sorted.npy'.format(b))

def remove_negativity(x):
    return x[x > 0]

def logify(array):
    return np.log(array)

def plot_spectrum(num_betas):

    # total x axis
    t = np.linspace(0, num_betas)
    fig, ax = plt.subplots()

    ymax = None
    ymin = None
    ytopmin = None

    for i in range(num_betas):
        # set beta
        beta = i

        # array to hold dominant eig values
        ytop = np.array([])
        ytopnorm = np.array([])

        # x position dependent on beta
        x = [beta - 0.3, beta + 0.3]
        
        # import sorted eigenvalues
        eigs = import_array(beta)
        
        # remove all negative eigs
        eigs = remove_negativity(eigs)
        
        # log values
        eigs = logify(eigs)

        # get the largest X values for each beta
        num_eigs = 20
        for e in it.islice(eigs, eigs.size-num_eigs, None):
            ytop = np.append(ytop, e)

        ytopmax = np.amax(ytop)
        for e in ytop:
            # normalise the y values s.t. max(y) = 1.
            y = [e/ytopmax, e/ytopmax]
            ax.plot(x,y,color='red')
            ytopnorm = np.append(ytopnorm, y)

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

        # find min values of ytopnorm
        if ytopmin != None:
            if ytopmin > np.amin(ytopnorm):
                ytopmin = np.amin(ytopnorm)
        else:
            ytopmin = np.amin(ytopnorm)
        
    # the amount to have above/below the max/min y value
    # buffer = (ymax - ytopmin) * 0.1
    buffer = (1 - ytopmin) * 0.1
    print('ymax for all {} betas is {}'.format(num_betas, ymax))
    print('ymin for all {} betas is {}'.format(num_betas, ymin))
    print('ytopmin for all betas is {}'.format(ytopmin))

    # set x and y range
    # plt.ylim([ytopmin - buffer, ymax + buffer])
    ax.set_ylim([ytopmin - buffer, 1 + buffer])
    print('buffer = {}'.format(buffer))

    ax.set_ylabel('eigenvalues')
    ax.set_xlabel('betas (1e-x)')
    ax.set_title('Normalised eigenspectrum for {} largest eigenvalues\n Model independently trained on label = 7'.format(num_eigs))

    # x axis ticks
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # ax.xaxis.set_major_formatter(ticker.NullFormatter())
    # xlist = [i+0.5 for i in range(13)]
    # xvals = [str(i) for i in range(13)]
    # ax.xaxis.set_minor_locator(ticker.FixedLocator(xlist))
    # ax.xaxis.set_minor_formatter(ticker.FixedFormatter(xvals))

    # save plot
    plt.savefig(outPath + 'top{}-normalised.png'.format(num_eigs))
    print('the plot has been saved as top{}-normalised.png'.format(num_eigs))

    # print("Number of elements in ytop:", ytop.size)
    # print("Number of elements in ybot:", ybot.size)

    # show plot
    # plt.show()

if __name__ == "__main__":
    # max beta num is 13
    num_betas = 13
    plot_spectrum(num_betas)

