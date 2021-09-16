def cal_mean_feature(feature,each_coord,chr_size): #cal mean feature in each consecutive window
	dist_range=[-1000,-500,-250,-50,0,50,250,500,1000]
	dist_range=[x+each_coord for x in dist_range]
	mean_feature=[feature[each_coord]]
	for i in range(len(dist_range)-1):
		start=max(0,dist_range[i])
		end=min(dist_range[i+1],chr_size)
		if end<=0 or start>=chr_size:
			mean_feature.append(0)
		else:
			mean_feature.append(round(np.mean(feature[start:end]),3))
	return mean_feature


import sys
import numpy as np
feature_file=sys.argv[1]
coord_file=sys.argv[2]
chr_size_file=sys.argv[3]
out_file=sys.argv[4]
outobject=open(out_file,'w')

#extract cpg coord for each chr
coord_chr={}
with open(coord_file) as f:
	for line in f:
		l=line.rstrip().split()
		if l[0] in coord_chr:
			coord_chr[l[0]].append(int(l[1]))
		else:
			coord_chr[l[0]]=[int(l[1])]

#extract chr size
chr_size_dict={}
with open(chr_size_file) as f:
	for line in f:
		l=line.rstrip().split()
		chr_size_dict[l[0]]=int(l[1])

#read feature data for each chr and cal mean feature for the corresponding chr
feature=[]
exist_chr=[]
pre_coord=[]
with open(feature_file) as f:
	for line in f:
		l=line.rstrip().split()
		if l[0] in exist_chr: #same chr
			for x in range(pre_coord,int(l[1])):#for between previous coord and current coord, give a feature of 0
				feature.append(0)
			for x in range(int(l[1]),int(l[2])): #for current coord
				feature.append(float(l[3]))
			pre_coord=int(l[2])
		else:			
			exist_chr.append(l[0])
			if l[0]=='chr1':
				for x in range(int(l[1])-1): #for upstream of the first coord, give a feature of 0
					feature.append(0)
				for x in range(int(l[1]),int(l[2])): #for current coord
					feature.append(float(l[3]))
				pre_coord=int(l[2])
			else:
				#cal avearage feature for cpgs in this chr
				pre_chr=exist_chr[-2]
				for x in range(pre_coord,chr_size_dict[pre_chr]): #for downstream of last coord, 0
					feature.append(0)
				if pre_chr in coord_chr: #cpg coord file contains that chromosome in feature file (fix bug 4-13-2020)
					for each_coord in coord_chr[pre_chr]: #for each cpg in one chr
						mean_feature=cal_mean_feature(feature,each_coord,chr_size_dict[pre_chr])
						print(pre_chr+'\t'+str(each_coord)+'\t'+str(each_coord+2)+'\t'+ ''.join([str(x)+'\t' for x in mean_feature[:-1]])+str(mean_feature[-1]),file=outobject)
				#rest feature after one chr is finished and add data
				feature=[]
				for x in range(int(l[1])-1): #for upstream of the first coord, give a feature of 0
					feature.append(0)
				for x in range(int(l[1]),int(l[2])): #for current coord
					feature.append(float(l[3]))
				pre_coord=int(l[2])
#for last chr
pre_chr=exist_chr[-1]
for x in range(pre_coord,chr_size_dict[pre_chr]): #for downstream of last coord, 0
	feature.append(0)
if pre_chr in coord_chr: 
	for each_coord in coord_chr[pre_chr]: #for each cpg in one chr
		mean_feature=cal_mean_feature(feature,each_coord,chr_size_dict[pre_chr])
		print(pre_chr+'\t'+str(each_coord)+'\t'+str(each_coord+2)+'\t'+ ''.join([str(x)+'\t' for x in mean_feature[:-1]])+str(mean_feature[-1]),file=outobject)



