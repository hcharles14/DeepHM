args=commandArgs(TRUE)
infile=args[1]
outfile=args[2]

data=read.table(infile,header=F)
logic=data$V7>=25 &data$V6>=20
write.table(cbind(data[logic,c(1:3,5)],data[logic,4]),outfile,row.names=F,col.names=F,quote=F,sep='\t')

