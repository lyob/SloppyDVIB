#%%
import pickle

test = pickle.load( open( "fim1.p", "rb" ))

print(test['softmax'])

#%%
print(test['logits'])

#%%
