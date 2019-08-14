#%%
# renaming

import os

path = "./DATA/train/"

for filename in os.listdir(path):
    if filename[0:5] == "h1024":
        # splitting the file name and the extension
        filename_without_ext = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]

        # turning filename into a list
        filename_without_ext = list(filename_without_ext)

        # new_file_name = filename_without_ext[:-11] # keep everything but the last 11 elements
        # new_file_name = filename_without_ext[-11:] # keep only the last 11 elements
        new_file_name = filename_without_ext[5:] # keep everything but the first 5 elements
        print(new_file_name)
        new_file_name = ''.join(str(e) for e in new_file_name)


        new_file_name = "h512" + new_file_name
        new_file_name_with_ext = new_file_name + extension
        print(new_file_name_with_ext)

        os.rename(os.path.join(path,filename), os.path.join(path, new_file_name_with_ext))


#%%
