import numpy as np
import random
from random import shuffle
import tensorflow as tf
import os
import csv
#import data_processing_util as dlu

def build_multi_channel_lstm_network(params):
    sequence_length = params['sequence_length']
    input_length = params['input_length']

    num_hidden = params['num_hidden']

    input_length = int(num_cols*time_laps*sampling_rate/float(sequence_length))

    data = tf.placeholder(tf.float32, [None, sequence_length,input_length]) #Number of examples, number of input, dimension of each input
    target = tf.placeholder(tf.float32)
    #target = tf.placeholder(tf.float32, [None, 3])

    cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
    val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    weight = tf.Variable(tf.truncated_normal([num_hidden, 1]))
    bias = tf.Variable(tf.constant(0.1, shape=[1]))

    prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
    cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)

    mistakes = tf.not_equal(target, prediction)
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)

    return (sess,minimize,error,data,target)

def run_lstm_model(tensorflow_data,data_input,data_output,train=True, batch_size=72, epoch=1000):

    sess,minimize,error,data,target,accuracy_class_1,accuracy_class_2,accuracy_class_3 = tensorflow_data

    no_of_batches = int(len(data_input)) / batch_size
    saver = tf.train.Saver()
    if train:
        #saver = tf.train.Saver()        
        saved_model = None
        for i in range(epoch):
            ptr = 0
            for j in range(no_of_batches):
                inp, out = data_input[ptr:ptr+batch_size], data_output[ptr:ptr+batch_size]
                ptr+=batch_size
                sess.run(minimize,{data: inp, target: out})
                #incorrect = sess.run(error,{data: train_input, target: train_output})
                #print('Epoch {:2d} loss {:3.1f}'.format(i + 1, incorrect))
            #print "Epoch ",str(i)
            incorrect = sess.run(error,{data: data_input, target: data_output})
            print('Epoch {:2d} loss {:3.5f}'.format(i + 1, incorrect))
            
            if (i+1)%50 == 0:
                saved_model = saver.save(sess,'./Saved_Models/lstm_model',global_step = i+1)   
    else:
        ckpt = tf.train.get_checkpoint_state('./Saved_Models')
        if ckpt and ckpt.model_checkpoint_path:
            print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print '\nError!!! first train the model, then test it\n'
        
        incorrect = sess.run(error,{data: data_input, target: data_output})

        print('Accuracy {:3.5f}%'.format(100-100 * incorrect))


def main(params,train_input,train_output,val_input,val_output,test_input,test_output,train=True):

    #train_input,train_output,val_input,val_output,test_input,test_output = dlu.main()
    train_input+=val_input
    train_output+=val_output

    print np.array(train_input).shape
    print np.array(test_input).shape
    tensorflow_data =  build_lstm_network(params = params)
    if train:
        run_lstm_model(tensorflow_data,train_input,train_output)
    run_lstm_model(tensorflow_data,test_input,test_output,train = False)

if __name__ == '__main__':
    main()

