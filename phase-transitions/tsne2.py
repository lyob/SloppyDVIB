# tsne on the FIM eigenvectors

# first import libraries
import numpy as np
import pandas as pd
import time
import pylab
import sklearn
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# global vars
beta = 2
maxbeta = 12
num_images = 15
num_labels = 20

# functions
def load_sorted_eigval_index(b):
    return np.load(inpath + 'sindex-b{}.npy'.format(b))

def load_eigenvectors(b):
    return np.load(inpath + 'b{}-vectors.npy'.format(b))

def build_dataset():
    print('building dataset...')
    # first order is by beta, then by image*label. each beta has 15 images. each image has 20002 dims
    all_vectors = np.ndarray(shape=(num_images * num_labels, 20002))
    all_labels = np.ndarray(shape=(num_images * num_labels))

    for i in range(num_images):
        i += 1
        print('building for image number ' + str(i))
        if int(i) < 10:
            i = '0{}'.format(i)
        model = 'h7/{}'.format(i)
        global inpath
        inpath = '/dali/sepalmer/blyo/centralised-v2/{}/data-eigs/'.format(model)

        # import sorted index
        sindex = load_sorted_eigval_index(beta)
        sindex = sindex[-num_labels:] # 10 largest eigenvalues

        # import eigenvectors
        vector = load_eigenvectors(beta)

        # extract the vectors corresponding to the 10 largest eigvalues ()
        # for r in range(10):
        #     val = (int(i)-1)*10 + r
        #     all_vectors[val] = vector[sindex[r]]
        #     all_labels[val] = str(r)

        # top ten only -- which is why there's 10 instead of 20002
        for r in range(num_labels):
            val = (int(i) - 1) * num_labels + r
            all_vectors[val] = vector[:,sindex[r]]
            all_labels[val] = str(int(r))

    return all_vectors, all_labels

def convert_to_dframe(X, Y):
    print('converting dataset to pandas dataframe...')
    dim_cols = ['dimension'+str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X,columns=dim_cols)
    df['label'] = Y
    df['labels'] = df['label'].apply(lambda i: str(i))
    return df, dim_cols

def randomise(df):
    print('randomising dataset...')
    np.random.seed(42) # for reproducibility
    rndperm = np.random.permutation(df.shape[0])
    df = df.loc[rndperm[:], :]
    return df

def run_tsne(df, dim_cols):
    print('running t-sne...')
    data = df[dim_cols].values

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)  # higher iterations maybe??
    tsne_results = tsne.fit_transform(data)
    print("t-sne complete! Time elapsed: {} seconds".format(time.time()-time_start))
    return tsne_results

def plot(results):
    outpath = '/dali/sepalmer/blyo/centralised-v2/plots/data-tsne/'
    df['tsne-2d-x'] = results[:,0]
    df['tsne-2d-y'] = results[:,1]
    plt.figure(figsize=(16,10))
    plot = sns.scatterplot(
            x="tsne-2d-x", y="tsne-2d-y",
            hue = "labels",
            palette=sns.color_palette("hls", num_labels),
            data = df,
            legend="full",
            alpha=0.3
            )
    fig = plot.get_figure()
    fig.savefig(outpath + "b{}-tsne.png".format(beta))


if __name__ == "__main__":
    vectors, labels = build_dataset()
    df, dim_cols = convert_to_dframe(vectors, labels)
    vectors, labels = None, None
    df = randomise(df)
    tsne_results = run_tsne(df, dim_cols)
    plot(tsne_results)


# 13 betas --- plots
# 15 images of 7 --- data point per label
# 10 largest eigvalues // eigenvectors --- each corresponding to a label
# 20002 dims

# MNIST:
# 1 plot
# 10000 samples (i.e. 15*10)
# 784 dims

# The question is, does it matter which order the data is fed into TSNE?