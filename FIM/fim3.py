#%%
## libraries
import pickle
import numpy as np

#%%
# import all 10 pickle files (which are dictionaries of scores)
scores = {}
for label in range(10):
    ls = pickle.load( open("scores{}.p".format(label), "rb") )
    scores.update( {"label{}".format(label) : ls} )

# now we have a dictionary of scores that looks like:
# scores = {"label0":{"l0p0":X, "l0p1":X, ...}, "label1": {"l1p0":X, "l1p1":X, ...}, ... }
# where X is a value


#%%
# build the matrix by calculating score(i, label) * score(j, label) for all pairs of parameters i,j
def calc_score_pairs(scores):
    print('------------------calculating score pairs-------------------')
    score_pairs = np.array([{},{},{},{},{},{},{},{},{},{}])
    for label in range(10):
        print('-------label: {}/9--------'.format(label))
        for i in range(20002):
            for j in range(20002):
                if 'i{}j{}'.format(j, i) in score_pairs[label]:
                    score_pairs[label]['i{}j{}'.format(i,j)] = score_pairs[label]['i{}j{}'.format(j,i)]
                else:
                    score_pairs[label].update( {'i{}j{}'.format(i,j) : scores["label{}".format(label)]['l{}p{}'.format(label, i)] * scores["label{}".format(label)]['l{}p{}'.format(label, j)]} )
    return score_pairs
# we have the if cond because sp(i,j|l) = sp(j,i|l) -- this would remove 10*20002 unnecessary calculations, but it 
# this is fine

score_pairs = calc_score_pairs(scores)

pickle.dump( score_pairs, open("score_pairs.p", "wb") )
print( "-------- Score pairs have been pickled as score_pairs.p --------" )

#%%

#%%
'''
dict1 = {"label0": {"l0p0": 23.3, "l0p1": 45.5}, "label1": {"l1p0":2.3, "l1p1":6.7}}
print(dict1["label1"]["l1p0"])
'''
#%%
