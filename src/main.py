from Autoencoder import Autoencoder
import numpy as np
from dataset_utils import load_dataset


#x_train = load_dataset("../data/data_np50/1")
x_train = load_dataset("../data/data100_np/small")
print(x_train.shape)
ae = Autoencoder(100, 100, x_train, x_train)
ae.create_model()

#ae.load_model("hehehe.h5")

ae.train_model(32, 20, 1)




