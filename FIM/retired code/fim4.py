#%%
## libraries
import pickle


# import pickle files
score_pairs = pickle.load( open("fim3.p", "rb") )
original = pickle.load( open('fim1.p', 'rb') )
softmax = original['softmax']

def calc_fim(score_pairs):
    print('-----------------------calculating the FIM---------------------------')
    fim = {} # this will be a 20002 * 20002 matrix
    for i in range(20002):
        for j in range(20002):
            if 'i{}j{}'.format(j,i) in fim:
                fim['i{}j{}'.format(i,j)] = fim['i{}j{}'.format(j,i)]
            else:
                sum = 0
                for label in range(10):
                    sum += softmax[0][label] * score_pairs[label]['i{}j{}'.format(i,j)]
                fim['i{}j{}'.format(i,j)] = sum
    return fim
# we can use the same trick we used in score_pairs above: if fim(ji|l) already exists then copy that value to fim(ij|l) 

fim = calc_fim(score_pairs)

pickle.dump( fim, open("fim4.p", "wb") )
print('----- fim matrix (dictionary) has been pickled as fim4.p -----')