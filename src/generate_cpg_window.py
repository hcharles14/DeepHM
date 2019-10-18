import sys
import numpy as np
feature_file=sys.argv[1]
out_file=sys.argv[2]
outobject=open(out_file,'w')

dist_range=[-1000,-500,-250,-50,0,50,250,500,1000]
with open(feature_file) as f:
	for line in f:
		l=line.rstrip().split()
		coord_range=[max(0,x+int(l[1])) for x in dist_range]
		for i in range(len(coord_range)-1):
			print(l[0],coord_range[i],coord_range[i+1],sep='\t',file=outobject)
outobject.close()
