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

	
	np_dir = "../data/test_cut_np"
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

def normalize_pos(data):

	#get data in range 0-1
	data += abs(np.min(data))
	data /= np.max(data)

def normalize_tanh(data):

	data -= np.median(data)
	data /= np.max(abs(data))


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
	return dataset

def shuffle_with_labels(dataset, labels):

	mapper = np.arange(dataset.shape[0])
    
	np.random.shuffle(mapper)
	dataset = dataset[mapper]
	labels = labels[mapper]
	
def shuffle(dataset):

	mapper = np.arange(dataset.shape[0])
	dataset = dataset[mapper]


if __name__ == '__main__':

	save_dataset("../data/test_cut")
