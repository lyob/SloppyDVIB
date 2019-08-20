#%%
from sklearn import datasets
import pickle

def save_dataset():
    # data from the scikit-learn package
    digits = datasets.load_digits() # 8x8 images of digits
    pickle.dump(digits, open('./DATA/dataset.p', "wb"))
    
save_dataset()

#%%
