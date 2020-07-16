from os import walk
import os
import numpy as np
import tensorflow as tf

def save_dataset(path):


	f = []
	for (dirpath, dirnames, filenames) in walk(path):
	    f.extend(filenames)
	    break

	np_dir = "data50_np"
	save_path = path.replace(os.path.dirname(path), np_dir)
	print("save path", save_path)

	for img in f:

		print("saving image ", img)
		full_path = path + "/" + img
		pil_img = tf.keras.preprocessing.image.load_img(full_path)
		np_array = tf.keras.preprocessing.image.img_to_array(pil_img)

		#print("save path ", save_path + "/" + img.split(".")[0] + ".npy")
		np.save(save_path + "/" + img.split(".")[0], np_array)


def load_dataset(path):
	
	f = []
	for (dirpath, dirnames, filenames) in walk(path):
	    f.extend(filenames)
	    break

	dataset = []

	#np_array = np.load(path + "/" + f[0])
	#dataset.append(np.array)

	for img in f:

		np_array = np.load(path + "/" + img)
		dataset.append(np_array)


	dataset = np.asarray(dataset)
	print(type(dataset))
	return dataset

#save_dataset("data50/2")


