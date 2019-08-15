import numpy as np
import matplotlib.pyplot as plt 

# Paths
savePath = './output/'
outPath = './plots/'

def import_array():
    array = np.load(savePath + 'eigenvalues.npy')
    return array

def remove_negativity(x):
    positive = x[x > 0]
    return positive

def logify(array):
    logarr = np.log(array)
    return logarr

def plotter(eigs, filename):
    # add label to y-axis
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

    # save plot in a eps file
    plt.savefig(outPath + filename + '.png') 

def plot_spectrum(eigs):
    ytop = np.array([])
    ybot = np.array([])

    x = [0.2, 0.8]
    
    for e in np.nditer(eigs):
        y = [e, e]
        if e > 10:
            plt.plot(x,y,color='red')
            ytop = np.append(ytop, e)
        else:
            plt.plot(x,y,color='green')
            ybot = np.append(ybot, e)
    plotter(eigs, 'espectrum')

    print("Number of elements in ytop:", ytop.size)
    print("Number of elements in ybot:", ybot.size)

    # show plot
    plt.show()

if __name__ == "__main__":
    eigs = import_array()
    eigs = remove_negativity(eigs)
    eigs = logify(eigs)
    plot_spectrum(eigs)

