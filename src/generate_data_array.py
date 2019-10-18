#This script generate x, y data for training or testing, which can be directly loaded.
import sys
import numpy as np
outfile=sys.argv[1]
infile=sys.argv[2]
x2_file=sys.argv[3] #dna feature
line_start=int(sys.argv[4]) #which line to start including(first line is 1)
line_end=int(sys.argv[5]) #which line to finishing including (This line is included)
 #medip/mre/hmcSeal feature
num_feature=int(sys.argv[6]) #number of features for each cpg (including itslef and neighboring cpgs)

#create dictionary to covert chromosome name into integer
chr_list=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chrX', 'chrY', 'chrM']
chr_encode=list(range(1,23))
chr_dict={}
for x,y in zip(chr_list,chr_encode):
        chr_dict[x]=y


infile_hander=open(infile)
x2_file_hander=open(x2_file)
num_line=line_end-line_start+1

count=0
data_x1=np.zeros((num_line,num_feature))
data_x2=np.zeros((num_line,3)) # 3 represents chr, start, end
data_y1=np.zeros((num_line,1))
data_y2=np.zeros((num_line,1))
i=0
while count<line_end:
	if count>=(line_start-1):
		theList=infile_hander.readline().rstrip().split()
		dnaList=x2_file_hander.readline().rstrip().split()
		data_x1[i]=[float(y) for y in theList[:-2]]
		data_x2[i]=[chr_dict[dnaList[0]],int(dnaList[1]),int(dnaList[2])]
		data_y1[i]=float(theList[-2])
		data_y2[i]=float(theList[-1])
		i=i+1
	else:
		next(infile_hander)
		next(x2_file_hander)
	count=count+1

np.savez(outfile,x1=data_x1,x2=data_x2,y1=data_y1,y2=data_y2)