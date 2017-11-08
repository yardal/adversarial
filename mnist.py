import tensorflow as tf
import argparse
import os
import sys
from tensorflow.examples.tutorials.mnist import input_data

data_path  = "data/"
model_dir  = "model/"
saved_model_path = os.path.join(model_dir,"model.ckpt")

sample_size = 28

def cnn(input,prob):

	z = [16,32]
	
	network = tf.reshape(input,[-1,28,28,1])
	network = tf.layers.conv2d(network,z[0],5,(2,2),padding='SAME',activation=tf.nn.relu)
	network = tf.layers.conv2d(network,z[1],3,(2,2),padding='SAME',activation=tf.nn.relu)
	
	dim = (network.get_shape().as_list()[1]**2*z[-1])
	
	network = tf.reshape(network,[-1,dim])
	network = tf.layers.dense(network,64,activation=tf.nn.relu)
	network = tf.layers.dropout(network,prob)
	network = tf.layers.dense(network,10,activation=None)
	return network
	
def main():

	x    = tf.placeholder(tf.float32,shape = (None,784))
	prob = tf.placeholder(tf.float32,shape = ())
	lr = tf.placeholder(tf.float32,shape = ())
	y    = tf.placeholder(tf.float32,shape = (None,10))
	mnist = input_data.read_data_sets(data_path, one_hot=True)
	net = cnn(x,prob)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = net))
	opt = tf.train.AdamOptimizer(lr).minimize(loss)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,axis=1),tf.argmax(net,axis=1)),tf.float32))

	lr_ = 1e-3
	with tf.Session() as sess: 
		sess.run(tf.global_variables_initializer())
		for i in range(50000):
			
			x_,y_ =   mnist.train.next_batch(256)
			sess.run(opt,feed_dict={x:x_,y:y_,prob:0.6,lr:lr_})
			if (i%200 == 0):
				x_,y_ =  mnist.test.next_batch(10000)
				acc,loss_ = sess.run([accuracy,loss],feed_dict={x:x_,y:y_,prob:1.0})
				print(loss_)
			if (i%4000 == 0):
				lr_ = lr_ * 0.1
if __name__ == "__main__":

	main()
