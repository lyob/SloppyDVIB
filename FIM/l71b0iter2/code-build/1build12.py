## libraries
import numpy as np
import pickle
import ray
import nujson as ujson
import timing



### global variables ###
# parameters -----------
size = 20002 # size of network is 20002
beta = 12

# paths
loadpath = '/home/blyo/python_tests/2FIM/data-scores/'
outpath = '/project2/sepalmer/blyo/data-fim/'
jsonpath = outpath


#----------------------fim1.py------------------------------------------------
def load_from_json(filename):
    with open(loadpath+filename, 'r') as f:
        object = ujson.load(f)
    return object

def load_from_pickle(filename):
    object = pickle.load( open(loadpath+filename, 'rb') )
    return object

def export_fim(matrix):
    np.save(outpath+'b{}-fim1.npy'.format(beta), matrix)
    print("The FIM matrix has been exported as b{}-fim1.npy".format(beta))


#----------------------fim3.py------------------------------------------------
# build the matrix by calculating score(i, label) * score(j, label) for all pairs of parameters i,j
# using structured numpy arrays instead of dictionaries to reduce memory usage

@ray.remote(num_cpus=1)
class Pairing:
    def __init__(self, scores, label, softmax):
        self.score_pairs = np.zeros((size, size))
        self.scores = scores
        self.label = label
        self.softmax = softmax

    def calc_score_pairs(self):
        for i in range(size):
            for j in range(size):
                if self.score_pairs[j][i] != 0:
                    self.score_pairs[i][j] = self.score_pairs[j][i]
                else:
                    self.score_pairs[i][j] = self.scores['l{}p{}'.format(self.label, i)] \
                                             * self.scores['l{}p{}'.format(self.label, j)] \
                                             * self.softmax[0][self.label]
        print('The size of the array for label={} is {} bytes'.format(self.label, self.score_pairs.nbytes))
        return self.score_pairs

@ray.remote(num_cpus=1)
def add(x,y):
    return np.add(x,y)

def calculate_fim(scores, softmax, start, end):
    print("****** Calculating the Fisher Information Matrix ******")

    # combining score pairs (weighted by softmax label values) to get the FIM 
    class_ids = [Pairing.remote(scores, i, softmax) for i in range(start, end)] # there should be 5 classes
    obj_ids = [c.calc_score_pairs.remote() for c in class_ids[start:end]] # there should be 5 instances of calcs (one per label)
    ready_ids, _ = ray.wait(obj_ids, num_returns=end-start, timeout=None) # there should be 5 results

    print('starting to add the 5 matrices...')
    # add the np arrays in log time using a tree structured pattern
    while len(ready_ids) > 1:
        ready_ids = ready_ids[2:] + [add.remote(ready_ids[0], ready_ids[1])]

    print('------ The matrices for l = {} to {} have been element-wise added ------'.format(start, end-1))

    return ray.get(ready_ids[0])





#-----------------------------------------------------------------------------------------
if __name__ == '__main__':

    # fim2.py
    ray.init(object_store_memory=40 * 1000 * 1000 * 1000)

    print('------ New beta value = {} ------'.format(beta))
    # import orig_network-b*.p and scores_all_labels-b*.json
    scores = load_from_json('scores_all_labels-b{}.json'.format(beta))
    out = load_from_pickle('orig_network-b{}.p'.format(beta))

    fim = calculate_fim(scores, out['softmax'], 0, 5)
    export_fim(fim)