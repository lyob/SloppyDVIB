# testing the order of argsort

inpath = '/dali/sepalmer/blyo/centralised/hb0/h702/'

sindex = np.load(inpath+'sindex-b0.npy')
eigval = np.load(inpath+'b0-evalues.npy')

for i in range(10):
    print('sorted index value number {} is {}'.format(i, sindex[-i-1]))
    print('eigenvalue {} is {}'.format(i, eigval[sindex[-i-1]]))
    print('----------')

