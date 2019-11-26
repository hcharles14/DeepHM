import sys
import numpy as np
DNA_file=sys.argv[1]
out_file=sys.argv[2]

#create nucleotide encode dict and chromosome list
nucleotide_dict={'a':'1000','t':'0100','c':'0010','g':'0001','n':'0000','A':'1000','T':'0100','C':'0010','G':'0001','N':'0000'}

dnadict={}
chr_name=''
all_encoded=[]
with open(DNA_file) as f:
    for line in f:
        l=line.rstrip()
        if len(l)>0:
            if l.startswith('>'):
                if chr_name=='':
                    all_encoded=[]
                    chr_name=l[1:]
                    print('process:',chr_name)
                else:
                    dnadict[chr_name]=all_encoded
                    all_encoded=[]
                    chr_name=l[1:]
                    print('process:',chr_name)
            else:
                for base in l:
                    for x in nucleotide_dict[base]:
                        all_encoded.append(int(x))
#for last chr
dnadict[chr_name]=all_encoded

#print size
for key in dnadict:
    print(key,len(dnadict[key]))

np.savez(out_file, chr1=dnadict['chr1'], chr2=dnadict['chr2'], chr3=dnadict['chr3'], chr4=dnadict['chr4'], \
chr5=dnadict['chr5'], chr6=dnadict['chr6'], \
chr7=dnadict['chr7'], chr8=dnadict['chr8'], chr9=dnadict['chr9'], chr10=dnadict['chr10'], chr11=dnadict['chr11'], \
chr12=dnadict['chr12'], chr13=dnadict['chr13'], chr14=dnadict['chr14'], chr15=dnadict['chr15'], chr16=dnadict['chr16'], \
chr17=dnadict['chr17'], chr18=dnadict['chr18'], chr19=dnadict['chr19'], chrX=dnadict['chrX'], chrY=dnadict['chrY'])

print('done')




