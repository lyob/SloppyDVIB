# This code takes the pre-trained network and calculates the scores. 
# The FIM can then be calculated from them using build_fim.py and build2_fim.py 


## libraries
import numpy as np
import tensorflow as tf
import pickle
import ray
import nujson as ujson
import timing
import os
import sys

own_name = sys.argv[0]
print(own_name)

# should be /home/blyo/python_tests/l75indep/code-gather
filepath = os.path.dirname(os.path.abspath(__file__)) 
filepath = filepath[:10]
print(filepath)


### global variables ###
# parameters -----------
weight_change = 0.01
size = 20002 # size of network is 20002
basepath='/home/blyo/python_tests/l74indep'


#-----------------------------------------------------------------------------------------
if __name__ == '__main__':

    global beta

    for b in range(0, 3):
        beta = b
        out = calc_original()
        scores = calc_scores_all_labels(out)
        


