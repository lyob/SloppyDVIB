# This code takes the pre-trained network and calculates the scores. 
# The FIM can then be calculated from them using build_fim.py and build2_fim.py 


## libraries
import os
import sys

# should be ..../gather_scores1.py
own_name = os.path.splitext(sys.argv[0])
own_name = own_name[0][-14:]
print(own_name) # gather_scores1

# should be /home/blyo/python_tests/l75indep/code-gather
filepath = os.path.dirname(os.path.abspath(__file__)) 
basepath = filepath[:-12]
print(basepath) # /home/blyo/python_tests/l75indep


#-----------------------------------------------------------------------------------------
if __name__ == '__main__':

    global beta

    if '1' in own_name:
        for b in range(0, 3):
            beta = b
            print(b)
    elif '2' in own_name:
        for b in range(3, 6):
            beta = b
            print(b)
    elif '3' in own_name:
        for b in range(6, 9):
            beta = b
            print(b)


