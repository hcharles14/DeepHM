args=commandArgs(TRUE)
infile=args[1]
normDataFile=args[2]
outfile=args[3]

data=read.table(infile,header=F)
normData=read.table(normDataFile,header=F)

normlize_column=c(1:26,28:45) #do not normalize mre and gc percent
#get rpkm for medip and hmcSeal
data[,18:26]=round(data[,18:26]/48.3,2) #total medip read
data[,28:36]=round(data[,28:36]/36.7,2) #total mre read
data[,37:45]=round(data[,37:45]/71.6,2) #total hmcSeal read
#get gc percent by dividng by 100
data[,1:8]=round(data[,1:8]/100,2)
#set max cpg_to_cgi distance to 100k
logic_temp=data[,9]>=100000
data[logic_temp,9]=100000

#setting max signal to be 50 for medip,mre,hmcSeal and cpg_num
data[,c(10:45)][data[,c(10:45)]>50]=50

#normalize data. data has the same column as original data
normalize_data_nonDist=function(data,normData,normlize_column){
	for (i in 1:length(normlize_column)){
			data[,normlize_column[i]]=(data[,normlize_column[i]]-normData[i,1])/normData[i,2]
	}
	return (data)
}

data_norm=normalize_data_nonDist(data,normData,normlize_column)
fake_label=cbind(rep(-1,nrow(data_norm)),rep(-1,nrow(data_norm)))
write.table(cbind(round(data_norm,4),fake_label),outfile,row.names=F,col.names=F,quote=F,sep='\t')


