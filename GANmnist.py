import tensorflow as tf
import numpy as np
import os,math
import cv2
import time
from tensorflow.examples.tutorials.mnist import input_data
from utils import *

data_path  = "data/"

sample_size = 28
data_path  = "data/"
model_dir  = "GANmodel/"
saved_model_path = os.path.join(model_dir,"model.ckpt")

def D(input,reuse):
	
	network = tf.reshape(input,[-1,28,28,1])
	network = tf.layers.conv2d(network,8,3,2 ,padding="SAME",activation=tf.nn.elu,name="D1",reuse=reuse)
	network = tf.layers.conv2d(network,16,3,2,padding="SAME",activation=tf.nn.elu,name="D2",reuse=reuse)
	network = tf.reshape(network,[-1,7*7*16])
	network = tf.layers.dense(network,16,activation=tf.nn.elu   ,name="D3",reuse=reuse)
	network = tf.layers.dense(network,2 ,activation=None      ,name="D4",reuse=reuse)
	return network

def G(input,reuse):

	N = 7 * 7 * 4
	network = tf.layers.dense(input,   N,activation=tf.nn.elu,name="G1",reuse=reuse)
	network = tf.layers.dense(input, 2*N,activation=tf.nn.elu,name="G4",reuse=reuse)
	network = tf.reshape(network,[-1,7,7,8])
	network = tf.layers.conv2d_transpose(network,16,(3,3),(2,2),padding='SAME',name="G2",reuse=reuse,activation=tf.nn.elu)
	network = tf.layers.conv2d_transpose(network,1 ,(3,3),(2,2),padding='SAME',name="G3",reuse=reuse)
	network = tf.reshape(network,[-1,784])
	return tf.nn.sigmoid(network)

	
def main():

	half_batch = 32
	latent = 100
	D_in  = tf.placeholder(tf.float32,shape = (None,784))
	G_in  = tf.placeholder(tf.float32,shape = (None,latent))
	D_out1 = tf.placeholder(tf.float32,shape = (None,2))
	D_out2 = tf.placeholder(tf.float32,shape = (None,2))
	
	discriminator = D(D_in,None)
	generator 	  = G(G_in,None)
	chain = D(G(G_in,True),True)
	mnist = input_data.read_data_sets(data_path, one_hot=True)
	
	D_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = D_out1,logits = discriminator))
	G_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = D_out2,logits = chain))

	D_opt = tf.train.AdamOptimizer(1e-3).minimize(D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "D"))
	G_opt = tf.train.AdamOptimizer(1e-3).minimize(G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "G"))
	
	print_var_list()
	
	real_in  = tf.placeholder(tf.float32,shape = (None,784))
	fake_in  = tf.placeholder(tf.float32,shape = (None,784))
	concat = tf.concat([real_in,fake_in],axis=0)
	labels1 = np.tile([1,0],[half_batch,1])
	labels2 = np.tile([0,1],[half_batch,1])
	labels = np.concatenate([labels1,labels2],axis=0)
	saver = tf.train.Saver()
	
					
	if (os.path.isfile(saved_model_path + ".index")):
		sess  = load_session(saved_model_path)		
	else:
		sess = tf.Session() 
		sess.run(tf.global_variables_initializer())
		
	iter = 0
	while True:	
		x1,_ =   mnist.train.next_batch(half_batch)
		seed = np.random.randn(half_batch,latent)
		x2	  = sess.run(generator,feed_dict = {G_in:seed})
		batch = sess.run(concat,feed_dict = {real_in:x1,fake_in:x2})

		if (iter%20 == 0):	
			dloss,gloss = sess.run([D_loss,G_loss],feed_dict = {D_out1:labels,D_in:batch,D_out2:labels1,G_in:seed})
			train_d = dloss > gloss or iter < 20
		if (train_d):
			sess.run(D_opt,feed_dict = {D_out1:labels,D_in:batch})
		else:
			sess.run(G_opt,feed_dict = {D_out2:labels1,G_in:seed})

		if (iter%500 == 0):
			print("G loss: " + str(gloss) + " D loss: " + str(dloss))
			show_batch(batch)
			saver.save(sess,saved_model_path)
		iter = iter + 1
	
if __name__ == "__main__":

	main()
