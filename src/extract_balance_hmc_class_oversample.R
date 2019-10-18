args=commandArgs(TRUE)
infile=args[1]
class_file=args[2]
outfile=args[3]

data=read.table(infile,header=F)
ncol=ncol(data)
class_data=read.table(class_file,header=F)
nclass=nrow(class_data)-1

#get sample_num
num_list=rep(0,nclass)
for (i in 1:nclass){
        if (i!=nclass){
                logic=data[,ncol]>=class_data[i,1] & data[,ncol]<class_data[i+1,1]          
        }else{
                logic=data[,ncol]>=class_data[i,1] & data[,ncol]<=class_data[i+1,1]         
        }
        num_list[i]=sum(logic)
}
print(num_list)
sample_num=median(num_list)


#get the coordinates of sampleed cpg
combine=c()
for (i in 1:nclass){
        if (i!=nclass){
                logic=data[,ncol]>=class_data[i,1] & data[,ncol]<class_data[i+1,1]          
        }else{
                logic=data[,ncol]>=class_data[i,1] & data[,ncol]<=class_data[i+1,1]         
        }

        data_filter=data[logic,]
        if (nrow(data_filter)>0){
                if (nrow(data_filter)>=sample_num){
                	rand_sample=sample(rownames(data_filter),sample_num)
                }else{
                	rand_sample=c(rownames(data_filter),sample(rownames(data_filter),sample_num-nrow(data_filter),replace=TRUE))
                }              
                combine=c(combine,as.numeric(rand_sample))       
        }
}

#remove cpgs that does not have 5 + or - neighbor
combine_sort=sort(combine)
#print(combine_sort)
write.table(data[combine_sort,1:3],outfile,row.names=F,col.names=F,quote=F,sep='\t')
summary(data[combine_sort,4])



