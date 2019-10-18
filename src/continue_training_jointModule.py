import numpy as np
import tensorflow as tf
import time
import random
import sys
import tensorflow as tf
import os

num_feature=45
batch_size = 500
num_epochs=100
diff_threshold=0.00001 #0.00001
num_steps = 1 
num_feature_dna=4*1000 #4 nucleotide * 100 neighbors
num_base_extend=1000

pred_file1=sys.argv[1]
pred_file2=sys.argv[2]
out_model=sys.argv[3]
feat_file=sys.argv[4]
label_file=sys.argv[5]
out_model_rnn=sys.argv[6]
out_model_cnn=sys.argv[7]
dna_seq_file=sys.argv[8]
pred_object1=open(pred_file1,'w')
pred_object2=open(pred_file2,'w')
cwd = os.getcwd()

#create dictionary to covert chromosome name into integer
chr_list=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chrX', 'chrY']
chr_encode=list(range(1,(len(chr_list)+1)))
chr_dict={}
for x,y in zip(chr_list,chr_encode):
  chr_dict[y]=x

#read dna seq dictionary, whose key is chromosome name and answer is a list
data_dna=np.load(dna_seq_file) #add 1-12
#extracted dictionary
dna_dict={}
for chr_name in chr_list:  ##
  print('extract:',chr_name)
  dna_dict[chr_name]=data_dna[chr_name]
del data_dna

def convert_coord_to_seq(data_array):
    all_list=[]
    array_shape=data_array.shape
    for i in range(array_shape[0]):
        one_list=[0]*array_shape[1]
        for j in range(int(array_shape[1]/3)):
            chr_name=chr_dict[data_array[i,j*3]]
            start=int((data_array[i,j*3+1]-num_base_extend/2)*4) #window size is num_base_extend
            end = start+4*num_base_extend
            if start<0:
                extra=[0]*(0-start)
                one_list[4*num_base_extend*j:4*num_base_extend*(j+1)]=extra+list(dna_dict[chr_name][0:end])
            elif end >len(dna_dict[chr_name]):
                extra=[0]*(end-len(dna_dict[chr_name]))
                one_list[4*num_base_extend*j:4*num_base_extend*(j+1)]=list(dna_dict[chr_name][start:len(dna_dict[chr_name])])+extra
            else:
                one_list[4*num_base_extend*j:4*num_base_extend*(j+1)]=dna_dict[chr_name][start:end]
        all_list.append(one_list)
    return np.array(all_list)

class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc,collectName,inputName):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta')
            saver.restore(self.sess, loc)
            # Get activation function from saved collection
            self.input=tf.get_collection(inputName)[0]
            self.activation = tf.get_collection(collectName)[0]

    def run(self, data):
        """ Running the activation function previously imported """
        return self.sess.run(self.activation, feed_dict={self.input: data})

def get_collection_rnn_state(name):
    layers = []
    coll = tf.get_collection(name)
    for i in range(0, len(coll), 2):
        state = tf.nn.rnn_cell.LSTMStateTuple(coll[i], coll[i+1])
        layers.append(state)
    return tuple(layers)

def main(unused_argv): 
    print('load data:')
    data_train = np.load(feat_file)
    len_train=len(data_train['y1'])
    data_test = np.load(label_file)
    len_test=len(data_test['y1'])
    print('total number of data in train set is: ',len_train)
    print('total number of data in val set is: ',len_test)
    print('\n')

    rnn_model = ImportGraph(cwd+'/'+out_model_rnn,"rnn_outputs","x1")
    cnn_model = ImportGraph(cwd+'/'+out_model_cnn,"cnn_outputs","x2")

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(cwd+'/'+out_model+'.meta')
        new_saver.restore(sess, cwd+'/'+out_model)
        rnn_x1 = tf.get_collection("rnn_x1")[0]
        cnn_x2 = tf.get_collection("cnn_x2")[0]
        y1 = tf.get_collection("y1")[0]
        y2 = tf.get_collection("y2")[0]
        joint_loss = tf.get_collection("joint_loss")[0]
        train_step = tf.get_collection("train_step")[0]
        diff1 = tf.get_collection("diff1")[0]
        diff2 = tf.get_collection("diff2")[0]
        y1_reshaped = tf.get_collection("y1_reshaped")[0]
        y2_reshaped = tf.get_collection("y2_reshaped")[0]
        y1_pred = tf.get_collection("y1_pred")[0]
        y2_pred = tf.get_collection("y2_pred")[0]

        data_test_x1=data_test['x1']
        data_test_x2=data_test['x2']
        data_test_y1=data_test['y1']
        data_test_y2=data_test['y2']
        del data_test
        data_train_x1=data_train['x1']
        print('shape:',data_train_x1.shape)
        data_train_x2=data_train['x2']
        data_train_y1=data_train['y1']        
        data_train_y2=data_train['y2']
        del data_train


        previous_test_error=1
        for epoch in range(num_epochs):
            print('epoch:',epoch)
            index_shuffle=random.sample(range(len_train),len_train)
            data_train_x1=data_train_x1[index_shuffle,:]
            data_train_x2=data_train_x2[index_shuffle,:]
            data_train_y1=data_train_y1[index_shuffle,:]
            data_train_y2=data_train_y2[index_shuffle,:]            
            epoch_average_loss=[]
            epoch_average_concordance1=[]
            epoch_average_concordance2=[]
            for i in range(int(len_train/batch_size)): #each batch
                #print('batch:',i)
                rand_sample=range(i*batch_size,(i+1)*batch_size)
                data_train_x2_seq=convert_coord_to_seq(data_train_x2[rand_sample,:])
                average_loss=[]
                average_concordance1=[]
                average_concordance2=[]
                for j in range(1): #each num_steps
                    rnn_outputs_pred=rnn_model.run(data_train_x1[rand_sample,(j*num_steps*num_feature):((j+1)*num_steps*num_feature)])
                    cnn_outputs_pred=cnn_model.run(data_train_x2_seq[:,(j*num_steps*num_feature_dna):((j+1)*num_steps*num_feature_dna)])
                    feed_dict={rnn_x1: rnn_outputs_pred, cnn_x2: cnn_outputs_pred, y1: data_train_y1[rand_sample,(j*num_steps):((j+1)*num_steps)], y2: data_train_y2[rand_sample,(j*num_steps):((j+1)*num_steps)]}

                    train_diff1,train_diff2,training_loss_, _ = sess.run([diff1,diff2,joint_loss,
                                                              train_step],
                                                                     feed_dict)
                    concordance1=round(sum([1 for x in train_diff1 if x<=0.1])/len(train_diff1),2)
                    concordance2=round(sum([1 for x in train_diff2 if x<=0.1])/len(train_diff2),2)
                    average_loss.append(training_loss_)
                    average_concordance1.append(concordance1)
                    average_concordance2.append(concordance2)
              
                epoch_average_loss.append(np.mean(average_loss))
                epoch_average_concordance1.append(np.mean(average_concordance1))
                epoch_average_concordance2.append(np.mean(average_concordance2))
                if 1:                    
                    print('batch:',i,"average training loss", np.mean(average_loss), 'label1 concordance',np.mean(average_concordance1), 'label2 concordance',np.mean(average_concordance2),sep='\t')
            print('epoch:',epoch,"average training loss", np.mean(epoch_average_loss), 'label1 concordance',np.mean(epoch_average_concordance1),'label2 concordance',np.mean(epoch_average_concordance2),sep='\t')

            if epoch>0:
                #for test data
                print('\n','epoch:',epoch,' evaluate for test data.')
                # data_test_x1=data_test['x1']
                # data_test_x2=data_test['x2']
                # data_test_y1=data_test['y1']
                # data_test_y2=data_test['y2']
                epoch_average_loss=[]
                epoch_average_concordance1=[]
                epoch_average_concordance2=[]
                for i in range(int(len_test/batch_size)): #each batch
                    average_loss=[]
                    average_concordance1=[]
                    average_concordance2=[]
                    sample=range(i*batch_size,(i+1)*batch_size)
                    data_test_x2_seq=convert_coord_to_seq(data_test_x2[sample,:])
                    for j in range(1): #each num_steps
                        rnn_outputs_pred=rnn_model.run(data_test_x1[sample,(j*num_steps*num_feature):((j+1)*num_steps*num_feature)])
                        cnn_outputs_pred=cnn_model.run(data_test_x2_seq[:,(j*num_steps*num_feature_dna):((j+1)*num_steps*num_feature_dna)])
                        feed_dict={rnn_x1: rnn_outputs_pred, cnn_x2: cnn_outputs_pred, y1: data_test_y1[sample,(j*num_steps):((j+1)*num_steps)], y2: data_test_y2[sample,(j*num_steps):((j+1)*num_steps)]}
                        test_diff1,test_diff2,test_loss_ = sess.run([diff1,diff2,joint_loss],
                                                                                     feed_dict) 
                        concordance1=round(sum([1 for x in test_diff1 if x<=0.1])/len(test_diff1),2)
                        concordance2=round(sum([1 for x in test_diff2 if x<=0.1])/len(test_diff2),2)
                        average_loss.append(test_loss_)
                        average_concordance1.append(concordance1)
                        average_concordance2.append(concordance2)
                    #print(average_loss)
                    print("test average loss", np.mean(average_loss), 'label1 concordance',np.mean(average_concordance1),'label2 concordance',np.mean(average_concordance2),sep='\t')
                    epoch_average_loss.append(np.mean(average_loss))
                    epoch_average_concordance1.append(np.mean(average_concordance1))
                    epoch_average_concordance2.append(np.mean(average_concordance2))
                print("final average test loss", np.mean(epoch_average_loss), 'label1 concordance',np.mean(epoch_average_concordance1),'label2 concordance',np.mean(epoch_average_concordance2),sep='\t')
                print()
                current_test_error=np.mean(epoch_average_loss)
                if (previous_test_error-current_test_error)<=diff_threshold and epoch>=150:
                    print('Test error is not decreasing in 10 epochs. Stop training.')
                    break
                else:
                    previous_test_error=current_test_error
            # Creates a saver.
            saver0 = tf.train.Saver()
            saver0.save(sess, cwd+'/'+out_model+'-epoch'+str(epoch))
            # Generates MetaGraphDef.
            saver0.export_meta_graph(cwd+'/'+out_model+'-epoch'+str(epoch)+'.meta')

        #for test data
        print('\n','evaluate for test data.')
        # data_test_x1=data_test['x1']
        # data_test_x2=data_test['x2']
        # data_test_y1=data_test['y1']
        # data_test_y2=data_test['y2']
        epoch_average_loss=[]
        epoch_average_concordance1=[]
        epoch_average_concordance2=[]
        for i in range(int(len_test/batch_size)): #each batch
            average_loss=[]
            average_concordance1=[]
            average_concordance2=[]
            sample=range(i*batch_size,(i+1)*batch_size)
            data_test_x2_seq=convert_coord_to_seq(data_test_x2[sample,:])
            for j in range(1): #each num_steps
                rnn_outputs_pred=rnn_model.run(data_test_x1[sample,(j*num_steps*num_feature):((j+1)*num_steps*num_feature)])
                cnn_outputs_pred=cnn_model.run(data_test_x2_seq[:,(j*num_steps*num_feature_dna):((j+1)*num_steps*num_feature_dna)])
                feed_dict={rnn_x1: rnn_outputs_pred, cnn_x2: cnn_outputs_pred, y1: data_test_y1[sample,(j*num_steps):((j+1)*num_steps)], y2: data_test_y2[sample,(j*num_steps):((j+1)*num_steps)]}
                y1_pred_test,y1_reshaped_test,test_diff1,y2_pred_test,y2_reshaped_test,test_diff2,test_loss_ = sess.run([y1_pred,y1_reshaped,diff1,y2_pred,y2_reshaped,diff2,joint_loss],
                                                                             feed_dict) 
                concordance1=round(sum([1 for x in test_diff1 if x<=0.1])/len(test_diff1),2)
                concordance2=round(sum([1 for x in test_diff2 if x<=0.1])/len(test_diff2),2)
                average_loss.append(test_loss_)
                average_concordance1.append(concordance1)
                average_concordance2.append(concordance2)
                for k in range(len(y1_pred_test)):
                    print(y1_reshaped_test[k],y1_pred_test[k],sep='\t',file=pred_object1)
                    print(y2_reshaped_test[k],y2_pred_test[k],sep='\t',file=pred_object2)
            #print(average_loss)
            print("test average loss", np.mean(average_loss), 'label1 concordance',np.mean(average_concordance1),'label2 concordance',np.mean(average_concordance2),sep='\t')
            epoch_average_loss.append(np.mean(average_loss))
            epoch_average_concordance1.append(np.mean(average_concordance1))
            epoch_average_concordance2.append(np.mean(average_concordance2))
        print("final average test loss", np.mean(epoch_average_loss), 'label1 concordance',np.mean(epoch_average_concordance1),'label2 concordance',np.mean(epoch_average_concordance2),sep='\t')
        pred_object1.close()  
        pred_object2.close()              

        # # Creates a saver.
        # saver0 = tf.train.Saver()
        # saver0.save(sess, cwd+'/'+out_model+str(epoch))
        # # Generates MetaGraphDef.
        # saver0.export_meta_graph(cwd+'/'+out_model++str(epoch)+'.meta')

if __name__ == "__main__":
  tf.app.run()


