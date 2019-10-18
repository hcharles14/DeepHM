import sys
infile=sys.argv[1]
outfile=sys.argv[2]
outobject=open(outfile,'w')

with open (infile) as f:
	for line in f:
		l=line.rstrip().split()	
		print(l[0],l[1],'+','CpG',l[3],l[4],sep='\t',file=outobject)
outobject.close()


