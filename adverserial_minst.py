import tensorflow as tf
import numpy as np
import cv2
import math
import argparse
import os,sys,argparse
from tensorflow.examples.tutorials.mnist import input_data
from utils import *

data_path  = "data/"
model_dir  = "model/"
saved_model_path = os.path.join(model_dir,"model.ckpt")
size = 28
height = size**2
num_classes = 10

def cnn(input,prob,reuse=None):

	z = [16,32]
	network = input
	if len(input.get_shape()) <= 2:
		network = tf.reshape(network,[-1,size,size,1])
	network = tf.layers.conv2d(network,z[0],5,(2,2),padding='SAME',activation=tf.nn.relu,reuse=reuse,name="C1")
	network = tf.layers.conv2d(network,z[1],3,(2,2),padding='SAME',activation=tf.nn.relu,reuse=reuse,name="C2")
	
	dim = (network.get_shape().as_list()[1]**2*z[-1])
	network = tf.reshape(network,[-1,dim])
	network = tf.layers.dense(network,64,activation=tf.nn.relu,reuse=reuse,name="C3")
	network = tf.layers.dropout(network,prob)
	network = tf.layers.dense(network,10,activation=None,reuse=reuse,name="C4")
	return network

def get_network_endpoints():

	endpoints = {}
	x = tf.placeholder(tf.float32,shape = (None,height))
	keep_prob = tf.placeholder(tf.float32,shape = ())
	lr = tf.placeholder(tf.float32,shape = ())
	y = tf.placeholder(tf.float32,shape = (None,num_classes))
	net = cnn(x,keep_prob)	
	prob = tf.nn.softmax(net)
	predict = tf.argmax(net,axis=1)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = net))
	opt = tf.train.AdamOptimizer(lr).minimize(loss)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,axis=1),tf.argmax(net,axis=1)),tf.float32))

	endpoints['x']         = x           
	endpoints['y']         = y           
	endpoints['net']       = net         
	endpoints['keep_prob'] = keep_prob   
	endpoints['lr']        = lr          
	endpoints['loss']      = loss        
	endpoints['prob']      = prob        
	endpoints['predict']   = predict     
	endpoints['opt']       = opt         
	endpoints['accuracy']  = accuracy    

	return endpoints

def get_adverserial_endpoints(network_endpoints):

	endpoints = {}
	grad_image = tf.placeholder(tf.float32,shape = (None,height))
	adverserial = cnn(grad_image,1.0,True)
	predict = tf.argmax(adverserial,axis=1)

	loss_adv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = network_endpoints['y'],logits = adverserial))
	grad = tf.gradients(loss_adv,grad_image)
	#grad = tf.sign(tf.reshape(grad[0],[1,size,size,1]))
	grad = (tf.reshape(grad[0],[1,size,size,1]))
	
	new_label = tf.placeholder(tf.int32,shape = ())
	old_label = tf.placeholder(tf.int32,shape = ())

	new_label_vec  = tf.one_hot(new_label,num_classes)
	new_label_vec  = tf.reshape(new_label_vec,[1,num_classes])
	old_label_vec  = tf.one_hot(old_label,num_classes)
	
	endpoints['grad_image'] = grad_image
	endpoints['adverserial'] = adverserial
	endpoints['loss_adv'] = loss_adv
	endpoints['grad'] = grad
	endpoints['new_label'] = new_label
	endpoints['old_label'] = old_label
	endpoints['new_label_vec'] = new_label_vec
	endpoints['old_label_vec'] = old_label_vec

	return endpoints

def train_cnn(sess,net,mnist,saver):

	lr_ = 1e-3
	acc = 0.0
	required_acc = 0.98
	i = 0
	checkpoint = 100
	best_acc = 0
	
	sess.run(tf.global_variables_initializer())
	while (acc < required_acc):
		i = i + 1
		x_,y_ =   mnist.train.next_batch(256)
		sess.run(net['opt'],feed_dict={net['x']:x_,net['y']:y_,net['keep_prob']:0.75,net['lr']:lr_})
		if (i%checkpoint == 0):
			x_,y_ =  mnist.test.next_batch(10000)
			acc = sess.run(net['accuracy'],feed_dict={net['x']:x_,net['y']:y_,net['keep_prob']:1.0})
			if (acc > best_acc):
				best_acc = acc
				print(str(acc) + " Saving")
				saver.save(sess,saved_model_path)
			else:
				print(acc)
	return sess

def main():

	net = get_network_endpoints()
	adv = get_adverserial_endpoints(net)
	mnist = input_data.read_data_sets(data_path, one_hot=True)
	
	saver = tf.train.Saver()
	if (os.path.isfile(saved_model_path + ".index")):
		sess  = load_session(saved_model_path)
		x_,y_ =  mnist.test.next_batch(10000)
		acc = sess.run(net['accuracy'],feed_dict={net['x']:x_,net['y']:y_,net['keep_prob']:1.0})
		print("model accuracy " + str(acc))
	else:
		sess = tf.Session()
		train_cnn(sess,net,mnist,saver)
	
	old_label = 1
	new_label = 2
	equal = False
	
	old, new = sess.run([adv['old_label_vec'],adv['new_label_vec']],feed_dict = {adv['new_label']:new_label,adv['old_label']:old_label})
	while not equal:
		x_,y_ =   mnist.train.next_batch(1)
		equal = np.array_equal(old,y_[0])
	
	J  = sess.run(adv['grad'],feed_dict = {net['y']:new,adv['grad_image']:x_})

	x_ = np.reshape(x_,[1,size,size,1])
	steps = 100
	for frac in range(steps+1):
		sum1 = x_ + J*(frac/steps)
		image = sum1 
		image = np.reshape(image,[1,height])
		predict_sum = sess.run(net['predict'],feed_dict={net['x']:image})
		predict_J   = sess.run(net['predict'],feed_dict={net['x']:np.reshape(J ,[1,height])})
		predict_x   = sess.run(net['predict'],feed_dict={net['x']:np.reshape(x_,[1,height])})
		print("sum " + str(predict_sum) + " predict grad " + str(predict_J) + " predict input : " + str(predict_x))
		show_batch(np.concatenate([x_,sum1,J],axis=0),factor=3)

if __name__ == "__main__":
	main()
