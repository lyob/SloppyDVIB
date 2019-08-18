#%%
# multiprocessing practice
import random
import string
import multiprocessing as mp
output = mp.Queue()

def rand_string(length, output):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    rand_str = ''.join(random.choice(
                        string.ascii_lowercase
                        + string.ascii_uppercase
                        + string.digits)
                   for i in range(length))
    output.put(rand_str)

processes = [mp.Process(target=rand_string, args=(5, output)) for x in range(4)]

for p in processes:
    p.start()

for p in processes:
    p.join()

results = [output.get() for p in processes]

print(results)



#%%
