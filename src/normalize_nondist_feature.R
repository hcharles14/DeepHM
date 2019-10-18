args=commandArgs(TRUE)
infile=args[1]
outfile=args[2]
normDataFile=args[3]
data_coord_file=args[4]

data_coord=read.table(data_coord_file,header=F)
data=read.table(infile,header=F)
normlize_column=c(1:26,28:45) #do not normalize mre and gc percent
normData=matrix(,ncol=4,nrow=length(normlize_column))

#get rpkm for medip and hmcSeal
data[,18:26]=round(data[,18:26]/48.3,2) #total medip read
data[,28:36]=round(data[,28:36]/36.7,2) #total mre read
data[,37:45]=round(data[,37:45]/71.6,2) #total hmcSeal read


#get gc percent by dividng by 100
data[,1:8]=round(data[,1:8]/100,2)

#set max cpg_to_cgi distance to 100k
logic_temp=data[,9]>=100000
data[logic_temp,9]=100000

#rm outlier by setting max signal to be 50 for medip,mre,hmcSeal and cpg_num
max_multiple=apply(data[,10:45],1,max)
logic=max_multiple<50
data_filter=data[logic,]


#get the normalize information
for (i in 1:length(normlize_column)){
	normData[i,1]=mean(data_filter[,normlize_column[i]])
	normData[i,2]=sd(data_filter[,normlize_column[i]])
	Q1=quantile(data_filter[,normlize_column[i]],0.25)
	Q3=quantile(data_filter[,normlize_column[i]],0.75)
	IQR=Q3-Q1
	normData[i,3]=Q1-IQR*1.5
	normData[i,4]=Q3+IQR*1.5	
}
write.table(normData,normDataFile,row.names=F,col.names=F,quote=F,sep='\t')

#normalize data. data has the same column as original data
normalize_data_nonDist=function(data,normData,normlize_column){
	for (i in 1:length(normlize_column)){
			data[,normlize_column[i]]=(data[,normlize_column[i]]-normData[i,1])/normData[i,2]
	}
	return (data)
}

data_norm=normalize_data_nonDist(data_filter,normData,normlize_column)
write.table(round(data_norm,4),outfile,row.names=F,col.names=F,quote=F,sep='\t')

write.table(data_coord[logic,],paste(outfile,'_coord',sep=''),row.names=F,col.names=F,quote=F,sep='\t')
