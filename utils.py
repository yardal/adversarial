import tensorflow as tf
import numpy as np
import math
import cv2

def print_var_list():
	
	var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	total = 0
	for i in var_list:
		val_total = 1
		for j in i.get_shape().as_list():
			val_total = val_total * j
		num_param_in_var = val_total
		strr = i.name + "\tParams: " + str(num_param_in_var)
		print(strr.expandtabs(27))
		total = total + num_param_in_var
	print("Total: " + str(total))

def load_session(saved_model_path):

	try:
		sess = tf.Session()
		saver = tf.train.Saver()
		saver.restore(sess, saved_model_path)
		return sess

	except:
		print("Unable to load model from:\t" + str(saved_model_path))
		print("Run with --train")
		sys.exit(0)
		

def find_grid(N):
	
	grid_x = 1
	for i in range(1,int(math.sqrt(N))+1):
		if (N%i == 0):
			grid_x = i

	grid_y = N/grid_x
	return int(grid_x),int(grid_y)
	
def show_batch(batch,factor=1,delay=100):

	x_range, y_range = find_grid(batch.shape[0])
	size = 28
	batch = np.reshape( batch,(x_range*y_range,size,size,1))
	batch = batch.transpose((1,2,3,0))
	batch = batch - np.amin(batch)
	batch = batch/np.amax(batch)
	border = 2
	new_image = np.zeros((x_range * (size + border),y_range * (size + border),3))
	for x in range(x_range):
		for y in range(y_range):
			image_index = x*y_range + y
			temp_patch = batch[:,:,:,image_index]
			new_image[x * (size+border):x * (size+border) + size, y * (size+border):y * (size+border) + size,:]= temp_patch
	
	new_image = np.kron(new_image, np.ones((factor,factor,1)))
	cv2.imshow("A",new_image)
	cv2.waitKey(delay)
