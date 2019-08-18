# mp using Pool.apply_async

import multiprocessing as mp

def powers(dict, l):
    dict2 = {}
    if l == 1:
        dict2.update({'p1': dict['value']**1})
    elif l == 2:
        dict2.update({'p2': dict['value']**2})
    elif l == 3:
        dict2.update({'p3': dict['value']**3})
    return dict2

def run_loop():
    dict = {
            'value' : 2
    }

    pool = mp.Pool(processes=3)
    all = {}
    for i in range(1, 4):
        result = pool.apply_async(powers, args=(dict, i))
        output = result.get()
        all = {**all, **output}
    print(all)

run_loop()




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