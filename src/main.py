# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from autoencoder import Autoencoder
from autogan import AutoGAN
import numpy as np
from dataset_utils import load_dataset, normalize_tanh

import tensorflow as tf


#positive = load_dataset("../data/faces_dummy_np")[:600]
#negative = load_dataset("../data/cut_n_paste_dummy")[:600]
positive = load_dataset("../data/faces_np")[:15000]
negative = load_dataset("../data/cut_n_paste_np")[:15000]

normalize_tanh(positive)
normalize_tanh(negative)

print("normalized range", np.min(positive), np.max(positive))



train = (positive, negative)

autogan = AutoGAN(128, 128, train, training=True)
autogan.create_model()
autogan_model = autogan.get_autogan_model()
autogan.load_trained("../checkpoints_autogan/epoch20")
autogan.train(150, 80)
