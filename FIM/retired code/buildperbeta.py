## libraries
import numpy as np
import pickle
import ray
import nujson as ujson
import timing



### global variables ###
# parameters -----------
size = 20002 # size of network is 20002

# paths
basepath = '/home/blyo/python_tests/2FIM'   
scorepath = basepath+'/data-scores/' # where all the .json and .p score files are kept
outpath = '/project2/sepalmer/blyo/data-fim/'  # where to save the built fim matrices

#----------------------fim1.py------------------------------------------------
def save_to_json(filename, object):
    with open(scorepath+filename, 'w') as f:
        ujson.dump(object, f)
    print("The data has been jsonified and saved as {}.".format(filename))

def load_from_json(filename):
    with open(scorepath+filename, 'r') as f:
        object = ujson.load(f)
    return object

def load_from_pickle(filename):
    object = pickle.load( open(scorepath+filename, 'rb') )
    return object

def load_from_npy(filename):
    object = np.load( outpath + filename )
    print('The FIM matrix {} has been loaded.'.format(filename))
    return object

def export_fim(matrix, filename):
    np.save(outpath+filename, matrix)
    print("The FIM matrix has been exported as {}".format(filename))

#----------------------fim3.py------------------------------------------------
# build the matrix by calculating score(i, label) * score(j, label) for all pairs of parameters i,j
# using structured numpy arrays instead of dictionaries to reduce memory usage

@ray.remote(num_cpus=1)
class Pairing:
    def __init__(self, scores, label, softmax):
        self.score_pairs = np.zeros((size, size))
        # self.tracker = np.zeros((size, size), dtype=np.uint8)
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

def add2(x,y):
    return np.add(x,y)

def calculate_fim(scores, softmax, start, end):
    print("********************** Calculating the Fisher Information Matrix *****************************")

    # combining score pairs (weighted by softmax label values) to get the FIM 
    class_ids = [Pairing.remote(scores, i, softmax) for i in range(10)]
    obj_ids = [c.calc_score_pairs.remote() for c in class_ids[start:end]]
    ready_ids, _ = ray.wait(obj_ids, num_returns=end-start, timeout=None)

    # add the np arrays in log time using a tree structured pattern
    while len(ready_ids) > 1:
        ready_ids = ready_ids[2:] + [add.remote(ready_ids[0], ready_ids[1])]

    print('--------- The matrices for l = {} to {} have been added element-wise together --------'.format(start, end-1))
    return ray.get(ready_ids[0])


#-----------------------------------------------------------------------------------------
if __name__ == '__main__':

    # fim2.py
    ray.init(object_store_memory=35*1000000000)

    global beta

    for b in range(0,3):
        beta = b
        print('------- New beta value = {} -------'.format(beta))

        # import orig_network-b*.p and scores_all_labels-b*.json
        scores = load_from_json('scores_all_labels-b{}.json'.format(beta))
        out = load_from_pickle('orig_network-b{}.p'.format(beta))

        fim = calculate_fim(scores, out['softmax'], 0, 5)
        export_fim(fim, 'b{}-fim1.npy'.format(beta))
        del fim
        
        fim = calculate_fim(scores, out['softmax'], 5, 10)
        export_fim(fim, 'b{}-fim2.npy'.format(beta))

        firstfim = load_from_npy('b{}-fim1.npy'.format(beta))
        # secondfim= load_from_npy('fim2-b{}.npy'.format(beta))
        fimtotal = add(firstfim, fim)
        print('The first and second FIM matrices have been added together')
        export_fim(fimtotal, 'total-fim-b{}.npy'.format(beta))
        del firstfim
        del fim
        del fimtotal



    
