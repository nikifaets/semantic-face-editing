from os import walk
import os
import numpy as np

def save_dataset(path):

	import tensorflow as tf

	f = []
	for (dirpath, dirnames, filenames) in walk(path):
		f.extend(filenames)
		print("Extend ", filenames)
		break

	
	np_dir = "../data/faces_np"
	save_path = np_dir
	print("save path", save_path)

	for img in f:

		print("saving image ", img)	
		full_path = path + "/" + img
		print("full path ", full_path)
		print("save path ", save_path)
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


if __name__ == '__main__':

	save_dataset("../data/faces")
