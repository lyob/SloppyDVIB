#%%
## libraries
import numpy as np
import pickle
import multiprocessing as mp
import psutil

#%%
# import all pickle files
def import_common_files():
    original = pickle.load( open('./pickle/fim1.p', 'rb') )
    softmax = original['softmax']
    return softmax

def import_per_label_files(label):
    ls = pickle.load( open("./pickle/scores{}.p".format(label), "rb") )
    # a dictionary of scores that looks like:
    # scores = {"label0":{"l0p0":X, "l0p1":X, ...}, "label1": {"l1p0":X, "l1p1":X, ...}, ... }
    # where X is a value
    return ls

# build the matrix by calculating score(i, label) * score(j, label) for all pairs of parameters i,j
# using structured numpy arrays instead of dictionaries to reduce memory usage
def calc_score_pairs(softmax, label):
    size = 100
    scores = import_per_label_files(label)
    print('----- calculating score pairs for label = {} -----'.format(label))
    score_pairs = np.zeros((size, size))
    tracker = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if tracker[j][i] == 1:
                score_pairs[i][j] = score_pairs[j][i]
            else:
                score_pairs[i][j] = scores['l{}p{}'.format(label, i)] * scores['l{}p{}'.format(label, j)] * softmax[0][label]
                tracker[i][j] = 1
    print('The size of the array for label={} is {} bytes'.format(label, score_pairs.nbytes))
    return score_pairs
# we have the if cond because sp(i,j|l) = sp(j,i|l) -- this would remove 10*20002 unnecessary calculations

# def multi():
#     for i in range(10):
#         p = mp.Process(target=calc_score_pairs, args=(scores, i))
#         p.start()
#         p.join()

#     print("done!")

def export_fim(matrix):
    np.save('./output/fim.npy', matrix)
    print("The FIM matrix has been exported as fim.npy")

if __name__ == '__main__':
    num_cores = mp.cpu_count()
    print('This kernel has {} cores.'.format(num_cores))
    print(psutil.virtual_memory())

    softmax = import_common_files()
    
    fim = calc_score_pairs(softmax, 0)
    # I want to parallelise this process and also keep memory use low by using subroutines
    for i in range(1,10):
        mat = calc_score_pairs(softmax, i)
        fim = np.add(fim, mat)
        print(psutil.virtual_memory())
    export_fim(fim)
    print("fim[0][0] =", fim[0][0])

