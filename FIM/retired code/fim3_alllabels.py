#%%
## libraries
import pickle
import numpy as np

# parameter
l = 0

#%%
# import all 10 pickle files (which are dictionaries of scores)
scores = {}
for label in range(10):
    ls = pickle.load( open("./pickle/scores{}.p".format(label), "rb") )
    scores.update( {"label{}".format(label) : ls} )

# now we have a dictionary of scores that looks like:
# scores = {"label0":{"l0p0":X, "l0p1":X, ...}, "label1": {"l1p0":X, "l1p1":X, ...}, ... }
# where X is a value

# print(scores["label0"]["l0p20001"])

#%%
# build the matrix by calculating score(i, label) * score(j, label) for all pairs of parameters i,j
def calc_score_pairs(scores, label):
    print('------------------calculating score pairs for label = {}-------------------'.format(l))
    score_pairs = {}
    for i in range(20002):
        if i%100==0: print(i)
        for j in range(20002):
            if 'i{}j{}'.format(j, i) in score_pairs:
                score_pairs['i{}j{}'.format(i,j)] = score_pairs['i{}j{}'.format(j,i)]
            else:
                score_pairs.update( {'i{}j{}'.format(i,j) : scores["label{}".format(label)]['l{}p{}'.format(label, i)] * scores["label{}".format(label)]['l{}p{}'.format(label, j)]} )
    return score_pairs
# we have the if cond because sp(i,j|l) = sp(j,i|l) -- this would remove 10*20002 unnecessary calculations, but it 
# this is fine

score_pairs = calc_score_pairs(scores, l)


