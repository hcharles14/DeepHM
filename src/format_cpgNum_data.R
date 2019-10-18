args=commandArgs(TRUE)
infile=args[1]
outfile=args[2]

data=read.table(infile,header=F)
data_mat=matrix(data[,1],ncol=8,byrow=T)
write.table(data_mat,outfile,row.names=F,col.names=F,quote=F,sep='\t')
