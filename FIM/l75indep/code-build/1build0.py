## libraries
import os
import sys


# dynamic parsing of name and path
own_name = os.path.splitext(sys.argv[0])
own_name = own_name[0][-7:]
print('file name is {}'.format(own_name)) # 1build0

# gets rid of /code_build
filepath = os.path.dirname(os.path.abspath(__file__))
basepath = filepath[:-11]
model = basepath[-8:]
print('base path is {}'.format(basepath)) # /home/blyo/python_tests/l7Xindep
print('model version is {}'.format(model)) # l7Xindep


beta = own_name[-1] # the bit at the end
section = int(own_name[0])  # the bit at the front
type(section)

# paths
loadpath = basepath+'/data-scores/'
outpath = '/dali/sepalmer/blyo/'+model+'/data-fim/'
jsonpath = outpath

print('loadpath is {}'.format(loadpath))
print('outpath is {}'.format(outpath))




#-----------------------------------------------------------------------------------------
if __name__ == '__main__':

    if section == 1:
        print('section is 1')
    elif section == 2:
        print('section is 2')
    
