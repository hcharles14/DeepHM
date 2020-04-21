args=commandArgs(TRUE)
infile=args[1]
outfile=args[2]
tab_cov=as.numeric(args[3])
wgbs_cov=as.numeric(args[4])

data=read.table(infile,header=F)
logic=data$V7>=tab_cov &data$V6>=wgbs_cov
write.table(cbind(data[logic,c(1:3,5)],data[logic,4]),outfile,row.names=F,col.names=F,quote=F,sep='\t')

