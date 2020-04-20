"""
Imports
"""
import numpy as np
import tensorflow as tf
import time
import random
import sys
import os

num_feature=45 #modified 1-6-2018 Yu #4 neighbor
num_epochs=5
batch_size =500
num_steps = 1 
learning_rate = 1e-4
diff_threshold=0.00001 #0.00001


#cnn parameter
num_feature_dna=4*1000 #4 nucleotide * 100 neighbors
#before conv
num_base_extend=1000
num_nucleotide=4
#conv layer1
field_size_layer1=10
num_filter_layer1=120
#conv layer2
field_size_layer2=10
num_filter_layer2=240
#full connect layer
num_feat_after_conv_layers=int(num_base_extend/5/5) #two pool layer
num_feat_cnn_output=100  ###100

#connect cnn/rnn output fully
num_feat_full_connect_layer=100 ##300


pred_file1=sys.argv[1]
pred_file2=sys.argv[2]
out_model=sys.argv[3]
feat_file=sys.argv[4]
label_file=sys.argv[5]
dna_seq_file=sys.argv[6]
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
for chr_name in chr_list: #example chr
  print('extract:',chr_name)
  dna_dict[chr_name]=data_dna[chr_name]
del data_dna


#for cnn
def conv_x2(x, W):
  """conv_x2 returns a 2d convolution layer with full stride."""
  return tf.nn.conv1d(x, W, stride=1, padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.layers.max_pooling1d(x, pool_size=5,
                        strides=5, padding='SAME')  #chagne from 2 to 5. 11-17

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def cnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, num_base_extend, num_nucleotide])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([field_size_layer1, num_nucleotide, num_filter_layer1])
    b_conv1 = bias_variable([num_filter_layer1])
    h_conv1 = tf.nn.tanh(conv_x2(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([field_size_layer2, num_filter_layer1, num_filter_layer2])
    b_conv2 = bias_variable([num_filter_layer2])
    h_conv2 = tf.nn.tanh(conv_x2(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([num_feat_after_conv_layers * num_filter_layer2, num_feat_cnn_output])
    b_fc1 = bias_variable([num_feat_cnn_output])

    h_pool2_flat = tf.reshape(h_pool2, [-1, num_feat_after_conv_layers * num_filter_layer2])
    h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # # Dropout - controls the complexity of the model, prevents co-adaptation of
  # # features.
  # with tf.name_scope('dropout'):
  #   keep_prob = tf.placeholder(tf.float32)
  #   h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # with tf.name_scope('fc2'):
  #   W_fc2 = weight_variable([500, 200])
  #   b_fc2 = bias_variable([200])
  #   h_fc2 = tf.nn.tanh(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  # Map the 1024 features to 10 classes, one for each digit
  # with tf.name_scope('fc3'):
  #   W_fc3 = weight_variable([500, 1])
  #   b_fc3 = bias_variable([1])

  #   y_conv = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)
  return h_fc1
#end cnn


def conv_x1(x, W):
  """conv_x1 returns a 2d convolution layer with full stride."""
  return tf.nn.conv1d(x, W, stride=num_feature, padding='VALID')

def add_to_collection_rnn_state(name, rnn_state):
    # store the name of each cell type in a different collection
    coll_of_names = name + '__names__'
    for layer in rnn_state:
        n = layer.__class__.__name__
        tf.add_to_collection(coll_of_names, n)
        try:
            for l in layer:
                tf.add_to_collection(name, l)
        except TypeError:
            # layer is not iterable so just add it directly
            tf.add_to_collection(name, layer)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

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

def train_network(g, num_epochs, min_test_loss_local, num_steps = 1, batch_size = 50, verbose = True):
    print('load data:')
    data_train = np.load(feat_file)
    len_train=len(data_train['y1'])
    data_test = np.load(label_file)
    len_test=len(data_test['y1'])
    print('total number of data in train set is: ',len_train)
    print('total number of data in val set is: ',len_test)
    print('\n')

    #add variables to tf.collection
    tf.add_to_collection("x1", g['x1'])
    tf.add_to_collection("x2", g['x2'])
    tf.add_to_collection("y1", g['y1'])
    tf.add_to_collection("y2", g['y2'])
    tf.add_to_collection("joint_loss", g['joint_loss'])
    tf.add_to_collection("train_step", g['train_step'])
    tf.add_to_collection("diff1", g['diff1'])
    tf.add_to_collection("diff2", g['diff2'])
    tf.add_to_collection("y1_reshaped", g['y1_reshaped'])
    tf.add_to_collection("y1_pred", g['y1_pred'])
    tf.add_to_collection("y2_reshaped", g['y2_reshaped'])
    tf.add_to_collection("y2_pred", g['y2_pred'])
    tf.add_to_collection("cnn_outputs", g['cnn_outputs'])

    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        data_test_x1=data_test['x1']
        data_test_x2=data_test['x2']
        data_test_y1=data_test['y1']
        data_test_y2=data_test['y2']
        del data_test
        data_train_x1=data_train['x1']
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
                    
                    feed_dict={g['x1']: data_train_x1[rand_sample,(j*num_steps*num_feature):((j+1)*num_steps*num_feature)],g['x2']: data_train_x2_seq[:,(j*num_steps*num_feature_dna):((j+1)*num_steps*num_feature_dna)], g['y1']: data_train_y1[rand_sample,(j*num_steps):((j+1)*num_steps)], g['y2']: data_train_y2[rand_sample,(j*num_steps):((j+1)*num_steps)]}

                    train_diff1,train_diff2,training_loss_, _ = sess.run([g['diff1'],g['diff2'],g['joint_loss'],
                                                              g['train_step']],
                                                                     feed_dict)
                    concordance1=round(sum([1 for x in train_diff1 if x<=0.1])/len(train_diff1),2)
                    concordance2=round(sum([1 for x in train_diff2 if x<=0.1])/len(train_diff2),2)
                    average_loss.append(training_loss_)
                    average_concordance1.append(concordance1)
                    average_concordance2.append(concordance2)
              
                epoch_average_loss.append(np.mean(average_loss))
                epoch_average_concordance1.append(np.mean(average_concordance1))
                epoch_average_concordance2.append(np.mean(average_concordance2))
                # if verbose:                    
                #     print('batch:',i,"average training loss", np.mean(average_loss), 'label1 concordance',np.mean(average_concordance1), 'label2 concordance',np.mean(average_concordance2),sep='\t')
            print('epoch:',epoch,"average training loss", np.mean(epoch_average_loss), 'label1 concordance',np.mean(epoch_average_concordance1),'label2 concordance',np.mean(epoch_average_concordance2),sep='\t')

            # if epoch%10==0:
            #     #for test data
            #     print('\n','epoch:',epoch,' evaluate for test data.')
            #     # data_test_x1=data_test['x1']
            #     # data_test_x2=data_test['x2']
            #     # data_test_y1=data_test['y1']
            #     # data_test_y2=data_test['y2']
            #     epoch_average_loss=[]
            #     epoch_average_concordance1=[]
            #     epoch_average_concordance2=[]
            #     for i in range(int(len_test/batch_size)): #each batch
            #         average_loss=[]
            #         average_concordance1=[]
            #         average_concordance2=[]
            #         sample=range(i*batch_size,(i+1)*batch_size)
            #         data_test_x2_seq=convert_coord_to_seq(data_test_x2[sample,:])
            #         for j in range(1): #each num_steps
                        
            #             feed_dict={g['x1']: data_test_x1[sample,(j*num_steps*num_feature):((j+1)*num_steps*num_feature)],g['x2']: data_test_x2_seq[:,(j*num_steps*num_feature_dna):((j+1)*num_steps*num_feature_dna)], g['y1']: data_test_y1[sample,(j*num_steps):((j+1)*num_steps)], g['y2']: data_test_y2[sample,(j*num_steps):((j+1)*num_steps)]}
            #             test_diff1,test_diff2,test_loss_ = sess.run([g['diff1'],g['diff2'],g['joint_loss']],
            #                                                                          feed_dict) 
            #             concordance1=round(sum([1 for x in test_diff1 if x<=0.1])/len(test_diff1),2)
            #             concordance2=round(sum([1 for x in test_diff2 if x<=0.1])/len(test_diff2),2)
            #             average_loss.append(test_loss_)
            #             average_concordance1.append(concordance1)
            #             average_concordance2.append(concordance2)
            #         #print(average_loss)
            #         print("test average loss", np.mean(average_loss), 'label1 concordance',np.mean(average_concordance1),'label2 concordance',np.mean(average_concordance2),sep='\t')
            #         epoch_average_loss.append(np.mean(average_loss))
            #         epoch_average_concordance1.append(np.mean(average_concordance1))
            #         epoch_average_concordance2.append(np.mean(average_concordance2))
            #     print("final average test loss", np.mean(epoch_average_loss), 'label1 concordance',np.mean(epoch_average_concordance1),'label2 concordance',np.mean(epoch_average_concordance2),sep='\t')
            #     print()
            #     current_test_error=np.mean(epoch_average_loss)
            #     if (previous_test_error-current_test_error)<=diff_threshold and epoch >=200:
            #         print('Test error is not decreasing in 10 epochs. Stop training.')
            #         break
            #     else:
            #         previous_test_error=current_test_error


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
                
                feed_dict={g['x1']: data_test_x1[sample,(j*num_steps*num_feature):((j+1)*num_steps*num_feature)],g['x2']: data_test_x2_seq[:,(j*num_steps*num_feature_dna):((j+1)*num_steps*num_feature_dna)], g['y1']: data_test_y1[sample,(j*num_steps):((j+1)*num_steps)], g['y2']: data_test_y2[sample,(j*num_steps):((j+1)*num_steps)]}
                y1_pred,y1_reshaped,test_diff1,y2_pred,y2_reshaped,test_diff2,test_loss_ = sess.run([g['y1_pred'],g['y1_reshaped'],g['diff1'],g['y2_pred'],g['y2_reshaped'],g['diff2'],g['joint_loss']],
                                                                             feed_dict) 
                concordance1=round(sum([1 for x in test_diff1 if x<=0.1])/len(test_diff1),2)
                concordance2=round(sum([1 for x in test_diff2 if x<=0.1])/len(test_diff2),2)
                average_loss.append(test_loss_)
                average_concordance1.append(concordance1)
                average_concordance2.append(concordance2)
                for k in range(len(y1_pred)):
                    print(y1_reshaped[k],y1_pred[k],sep='\t',file=pred_object1)
                    print(y2_reshaped[k],y2_pred[k],sep='\t',file=pred_object2)
            #print(average_loss)
            #print("test average loss", np.mean(average_loss), 'label1 concordance',np.mean(average_concordance1),'label2 concordance',np.mean(average_concordance2),sep='\t')
            epoch_average_loss.append(np.mean(average_loss))
            epoch_average_concordance1.append(np.mean(average_concordance1))
            epoch_average_concordance2.append(np.mean(average_concordance2))
        print("final average test loss", np.mean(epoch_average_loss), 'label1 concordance',np.mean(epoch_average_concordance1),'label2 concordance',np.mean(epoch_average_concordance2),sep='\t')


        #add 4-17-2020 YU He 
        # Creates a saver.
        if np.mean(epoch_average_loss)<min_test_loss_local:
            min_test_loss_local=np.mean(epoch_average_loss)
            min_test_result='final average test loss:' + str(np.mean(epoch_average_loss)) +'\t' +'label1 concordance:' + str(np.mean(epoch_average_concordance1)) +'\t' + 'label2 concordance:' + str(np.mean(epoch_average_concordance2))
            saver0 = tf.train.Saver()
            saver0.save(sess, cwd+'/'+out_model)
            # Generates MetaGraphDef.
            saver0.export_meta_graph(cwd+'/'+out_model+'.meta')
        return min_test_loss_local

def build_multilayer_lstm_graph_with_dynamic_rnn(
    batch_size = 50,
    num_steps = 1,
    learning_rate = 1e-4):

    reset_graph()

    x1 = tf.placeholder(tf.float32, [batch_size, num_steps*num_feature], name='input1_placeholder')
    x2 = tf.placeholder(tf.float32, [batch_size, num_steps*num_feature_dna], name='input2_placeholder')
    y1 = tf.placeholder(tf.float32, [batch_size, num_steps], name='label1_placeholder')
    y2 = tf.placeholder(tf.float32, [batch_size, num_steps], name='label2_placeholder')

    #cnn layer
    cnn_outputs=cnn(x2)


    y1_reshaped = tf.reshape(y1, [-1])
    y2_reshaped = tf.reshape(y2, [-1])

    #combine cnn and rnn output
    #rnn_outputs=tf.concat([x1, cnn_outputs], 1) 
    rnn_outputs=cnn_outputs

   # 1 Fully connected layer for label1
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([num_feat_cnn_output, num_feat_full_connect_layer])
        b_fc1 = bias_variable([num_feat_full_connect_layer])
        h_fc1 = tf.nn.relu(tf.matmul(rnn_outputs, W_fc1) + b_fc1)

    # 1 Fully connected layer for label2
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([num_feat_cnn_output, num_feat_full_connect_layer])
        b_fc2 = bias_variable([num_feat_full_connect_layer])
        h_fc2 = tf.nn.relu(tf.matmul(rnn_outputs, W_fc2) + b_fc2)

   # 2 Fully connected layer for label1
    with tf.name_scope('fc12'):
        W_fc12 = weight_variable([num_feat_full_connect_layer, num_feat_full_connect_layer])
        b_fc12 = bias_variable([num_feat_full_connect_layer])
        h_fc12 = tf.nn.relu(tf.matmul(h_fc1, W_fc12) + b_fc12)

    # 2 Fully connected layer for label2
    with tf.name_scope('fc22'):
        W_fc22 = weight_variable([num_feat_full_connect_layer, num_feat_full_connect_layer])
        b_fc22 = bias_variable([num_feat_full_connect_layer])
        h_fc22 = tf.nn.relu(tf.matmul(h_fc2, W_fc22) + b_fc22)

    with tf.variable_scope('label1'):
        W1 = tf.get_variable('W1', [num_feat_full_connect_layer, 1])
        b1 = tf.get_variable('b1', [1], initializer=tf.constant_initializer(0.0))
    with tf.variable_scope('label2'):
        W2 = tf.get_variable('W2', [num_feat_full_connect_layer, 1])
        b2 = tf.get_variable('b2', [1], initializer=tf.constant_initializer(0.0))

    #y1
    y1_pred = tf.nn.relu(tf.matmul(h_fc12, W1) + b1)
    y1_pred=tf.squeeze(y1_pred)
    squared_deltas1 = tf.square(y1_pred - y1_reshaped)
    train_loss1 = tf.reduce_mean(squared_deltas1)
    diff1=tf.abs(y1_pred - y1_reshaped)
    #y2
    y2_pred = tf.nn.relu(tf.matmul(h_fc22, W2) + b2)
    y2_pred=tf.squeeze(y2_pred)
    squared_deltas2 = tf.square(y2_pred - y2_reshaped)
    train_loss2 = tf.reduce_mean(squared_deltas2)
    diff2=tf.abs(y2_pred - y2_reshaped)
    #joint loss
    joint_loss=train_loss1 + 0.5*train_loss2 #for balance
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(joint_loss)


    return dict(
        x1 = x1,
        x2 = x2,
        y1 = y1,
        y2 = y2,
        joint_loss = joint_loss,
        train_step = train_step,
        diff1=diff1,
        diff2=diff2,
        y1_reshaped=y1_reshaped,
        y1_pred=y1_pred,
        y2_reshaped=y2_reshaped,
        y2_pred=y2_pred,
        cnn_outputs=cnn_outputs
    )

# t = time.time()
# build_multilayer_lstm_graph_with_dynamic_rnn()
# print("It took", time.time() - t, "seconds to build the graph.")

#add codes to train 5 times and pick model that have best performance. 4-15-2020 yu
min_test_loss=1000000
for num_try in range(5):
    print("\n\n\n")
    print("Try training:"+str(num_try+1))
    g = build_multilayer_lstm_graph_with_dynamic_rnn(batch_size,num_steps, learning_rate)
    min_test_loss=train_network(g, num_epochs, min_test_loss,num_steps , batch_size)
print('The minimal test loss from 5 training is: '+str(min_test_loss))
pred_object1.close()  
pred_object2.close()



