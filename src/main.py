#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

from Autoencoder import Autoencoder
import numpy as np
from dataset_utils import load_dataset


x_train = load_dataset("../data/faces_np")[:15000]/255
print(x_train.shape)
ae = Autoencoder(128, 128, x_train, x_train)
ae.create_model()

#ae.load_model("hehehe.h5")

ae.train_model(80, 25, 1)
ae.save_model("weights")