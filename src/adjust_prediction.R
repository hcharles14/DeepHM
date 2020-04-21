args=commandArgs(TRUE)
infile=args[1]
outfile1=args[2]
outfile2=args[3]

data=read.table(infile,header=F)
data[data$V4>1,4]=1
data[data$V4<0,4]=0
data[data$V5>1,5]=1
data[data$V5<0,5]=0

write.table(cbind(data[,c(1:3,4)],rep(20,nrow(data))),outfile1,row.names=F,col.names=F,quote=F,sep='\t')
write.table(cbind(data[,c(1:3,5)],rep(20,nrow(data))),outfile2,row.names=F,col.names=F,quote=F,sep='\t')

