
import sys
import os
own_name = sys.argv[0]
tup = os.path.splitext(own_name)
out = tup[0][3:4]
print(out)


filepath = os.path.dirname(os.path.abspath(__file__))
if 'post' in filepath:
    print('fim is in filepath')
else:
    print('fim is not in filepath')