import sys
import numpy as np
feature_file=sys.argv[1]
out_file=sys.argv[2]
outobject=open(out_file,'w')

pre=['chr0','0']
cgi_top=-1
cpg_notCgi=[]
with open(feature_file) as f:
	for line in f:
		l=line.rstrip().split()
		if l[0]!=pre[0]: #diff chro
			#print(l,cpg_notCgi)
			for ele in cpg_notCgi:
				if cgi_top>0:
					print(ele-cgi_top,file=outobject)
				else:
					print(1000000,file=outobject)
			cpg_notCgi=[]
			cgi_top=-1

		if l[-1]=='2': #is in cgi
			if len(cpg_notCgi)>0: #previous cpg is not in cgi
				cgi_bottom=int(l[1]) #update cgi_bottom
				for ele in cpg_notCgi:
					if cgi_top==-1:
						print(cgi_bottom-ele,file=outobject)
					else:
						print(min(ele-cgi_top,cgi_bottom-ele),file=outobject)

				cpg_notCgi=[]
			cgi_top=int(l[1]) #update cgi_top
			print(0,file=outobject)
		else:
			cpg_notCgi.append(int(l[1]))
		pre=l[0:2]

#for last cgp_notCgi in last chro
for ele in cpg_notCgi:
	#print(cpg_notCgi)
	if cgi_top>0:
		print(ele-cgi_top,file=outobject)
	else:
		print(1000000,file=outobject)
outobject.close()
