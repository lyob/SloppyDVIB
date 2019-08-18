# mp using Pool.apply and Pool.map

import multiprocessing as mp

def cube(x):
    return x**3

pool = mp.Pool(processes=4)
results = [pool.apply(cube, args=(x,)) for x in range(1, 7)]
print(results)


# alternatively:
pool = mp.Pool(processes=4)
results = [pool.map(cube, range(1, 7))]


'''
What is the difference between Pool.map and Pool.apply?
Pool.map applies the same function to many arguments, while Pool.apply 
can be used to call a number of different functions.

What is the difference between Pool.apply and Pool.apply_async?
In pool.apply the function call is blocked until the function is completed.
In apply_sync, the call returns immediately instead of waiting for the result. 
You can call its get() method to retrieve the results of the function call.
The get() methods blocks until the function is completed. 

Answered here:
https://stackoverflow.com/questions/8533318/multiprocessing-pool-when-to-use-apply-apply-async-or-map
'''