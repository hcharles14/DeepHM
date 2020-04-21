import sys
import numpy as np
infile=sys.argv[1]
infor=sys.argv[2]

loss_list=[]
with open(infile) as f:
    for line in f:
        l=line.rstrip().split('\t')
        loss_list.append(float(l[1]))
min_value=np.min(loss_list)
min_index=loss_list.index(min_value)+1
print(infor+'-epoch'+str(min_index), ' has the smallest loss of ', min_value)
