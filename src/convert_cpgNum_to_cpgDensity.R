args=commandArgs(TRUE)
infile=args[1]
outfile=args[2]

data=read.table(infile,header=F,sep='\t')
distance_list=c(500,250,200,50,50,200,250,500)
for (i in 1:length(distance_list)){
	data[,9+i]=round((data[,9+i]/distance_list[i])/(data[,i]/200)^2,2)
	data[is.na(data[,9+i]),9+i]=0
	data[is.infinite(data[,9+i]),9+i]=0 #gc density can be 0, which will generate na or inf
}
write.table(data,outfile,row.names=F,col.names=F,quote=F,sep='\t')
