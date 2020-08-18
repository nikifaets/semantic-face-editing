'''import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"  '''

from autoencoder import Autoencoder
from autogan import AutoGAN
import numpy as np
from dataset_utils import load_dataset

import tensorflow as tf

#x_train = load_dataset("../data/faces_np")[:15000]/255
#print(x_train.shape)
#ae = Autoencoder(128, 128, x_train, x_train)
#ae.create_model()

positive = load_dataset("../data/faces_np")[:15000]/255
negative = load_dataset("../data/cut_n_paste_np")[:15000]/255

positive_test = load_dataset("../data/test_np")[:2000]/255
negative_test = load_dataset("../data/test_cut_np")[:2000]/255

train = (positive, negative)
test = (positive_test, negative_test)

autogan = AutoGAN(128, 128, train, test)
autogan.create_model()
autogan_model = autogan.get_autogan_model()

autogan.train(10, 60)
#ae.load_model("hehehe.h5")

#ae.train_model(80, 25, 1)
#ae.save_model("weights")