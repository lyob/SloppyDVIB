import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as ticker
import itertools as it
import os
import sys

own_name = os.path.splitext(sys.argv[0]) # ../code-plot/plotnorm10.py
own_name = own_name[0][13:] # plotnorm10
print('file name is {}'.format(own_name))
etype = 'h'

num_eigs = 10
num_betas = 13

outPath ='/dali/sepalmer/blyo/l7avgindep/data-plots/'

def import_array(b):
    return np.load(inPath + 'sorted-b{}.npy'.format(b))

def remove_negativity(x):
    return x[x > 0]

def logify(array):
    return np.log(array)

def plot_spectrum(yvals):

    # total x axis
    t = np.linspace(0, num_betas)
    fig, ax = plt.subplots()

    ymin = None

    for i in range(num_betas):
        # set beta
        beta = i
        yval = yvals[beta]

        # array to hold dominant eig values
        ytop = np.array([])

        # x position dependent on beta
        x = [beta - 0.3, beta + 0.3]
        
        for e in yval:
            y = [e, e]
            ax.plot(x,y,color='red')

        # find max and min values of y
        if ymin == None:
            ymin = np.amin(yval)
        else:
            # set new ymin if you find smaller values
            if ymin > np.amin(yval):
                ymin = np.amin(yval)
        
    # the amount to have above/below the max/min y value
    # buffer = (ymax - ytopmin) * 0.1
    buffer = (1 - ymin) * 0.1

    # set x and y range
    ax.set_ylim([0.2, 1 + buffer])

    ax.set_ylabel('eigenvalues')
    ax.set_xlabel('betas (1e-x)')
    ax.set_title('Normalised eigenspectrum for {} largest eigenvalues\n Model trained on label = 7, epoch = 10000'.format(num_eigs))

    # x axis ticks
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # save plot
    plt.savefig(outPath + '{}7avgindep-top{}.png'.format(etype, num_eigs))
    print('the plot has been saved as {}7avgindep-top{}.png'.format(etype, num_eigs))


if __name__ == "__main__":

    global inPath

    # max beta num is 13
    ytopavg = np.array([])
    for b in range(13): # beta runs from b = 0 to 12
        # array to hold dominant eig values
        
        mrange = 9

        ytopnorm = np.zeros(num_eigs)
        for m in range(1,1+mrange): # models run from h71 to h79
            model = '{}7{}indep'.format(etype, m)
            # Paths - import the eigenvalues that are already sorted with sort.py 
            inPath = '/dali/sepalmer/blyo/{}/data-eigs/'.format(model)

            ytop = np.array([])

            # imports l7Mindep/sorted-bB
            eigs = import_array(b)
            # remove all negative eigs
            eigs = remove_negativity(eigs)
            # log values
            eigs = logify(eigs)

            # get the largest X values for each beta
            for e in it.islice(eigs, eigs.size-num_eigs, None):
                ytop = np.append(ytop, e)

            # normalise the y values s.t. max(y) = 1.
            ytopmax = np.amax(ytop)
            for i, e in enumerate(ytop):
                ytopnorm[i] += e/ytopmax

        if ytopavg.size == 0:
            ytopavg = np.divide(ytopnorm, mrange)
        else:
            ytopavg = np.vstack((ytopavg, np.divide(ytopnorm, mrange)))

    print(ytopavg)

    plot_spectrum(ytopavg)









