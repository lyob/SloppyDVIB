#%%
# renaming

import os

path = "./NMNN/DATA/train/"

for filename in os.listdir(path):
    if filename[0:4] == "NMNN":
        filename_without_ext = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]

        filename_without_ext = list(filename_without_ext)

        new_file_name = filename_without_ext[:-11] # keep everything but the last 11 elements
        print(new_file_name)
        # new_file_name = filename_without_ext[-11:] # keep only the last 11 elements
        new_file_name = ''.join(str(e) for e in new_file_name)


        new_file_name = new_file_name + "train-20000"
        new_file_name_with_ext = new_file_name + extension
        print(new_file_name_with_ext)
        os.rename(os.path.join(path,filename), os.path.join(path, new_file_name_with_ext))


#%%
