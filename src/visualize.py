import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
from dataset_utils import load_dataset, normalize_pos, normalize_tanh
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from mpl_toolkits.axes_grid1 import ImageGrid

def plot(dataset, start, end):

    dataset = dataset[start:end]
    normalize_pos(dataset)

    fig = plt.figure(figsize=(4., 4.))

    rows = 4
    cols = 4
    for pos in range(1, min(rows*cols+1, dataset.shape[0]+1)):

        img = dataset[pos-1]
        fig.add_subplot(rows, cols, pos)
        plt.imshow(img)

    return plt

def load_model(filename):

    from autogan import AutoGAN

    autogan = AutoGAN(128,128, training=False)
    autogan_model = autogan.get_autogan_model()
    autogan_model.load_weights(filename)

    return autogan_model

def test_autoencoder_output(model, data):

    autogan_model = model
    ae_model = autogan_model.get_layer("autoencoder")
    print(ae_model)
    pred = ae_model.predict(data)

    plt_input = plot(data,20, 36)
    plt_pred = plot(pred, 20, 36)
    plt_input.show()
    plt_pred.show()
    #ae_in = ae_model.get_layer("autoencoder_input")
    #ae_out =  ae_model.get_layer("autoencoder_output")

    
def test_discriminator(model, negative, positive):

    autogan_model = model
    disc_model = autogan_model.get_layer("discriminator")
    #disc_in = disc_model.get_layer("discriminator_input")
    #disc_out = disc_model.get_layer("discriminator_output")
    pred_neg = disc_model.predict(negative)
    pred_pos = disc_model.predict(positive)
    

    print("neg range", np.min(pred_neg), np.max(pred_neg))
    print("pos range", np.min(pred_pos), np.max(pred_pos))

    print("negatives", negative.shape[0])
    print("correct negatives", pred_neg[(pred_neg<=0.5)].shape[0])

    print("positives", positive.shape[0])
    print("correct positives", pred_pos[(pred_pos>=0.5)].shape[0])

def test_autogan(model, negative):

    autogan_model = model
    ae_model = autogan_model.get_layer("autoencoder")
    batch = negative[50:66]
    ae_out = ae_model.predict(batch)
    
    normalize_tanh(ae_out)

    disc_model = autogan_model.get_layer("discriminator")
    disc_out = disc_model.predict(ae_out)
    print("discriminator output ", disc_out)

if __name__ == '__main__':

    positive = load_dataset("../data/test_np")[:100]
    negative = load_dataset("../data/test_cut_np")[:100]

    normalize_tanh(positive)
    normalize_tanh(negative)

    filename = "../checkpoints_autogan/epoch56"
    autogan_model = load_model(filename)
    #test autogan
    test_autogan(autogan_model, negative)

    #test autoencoder
    test_autoencoder_output(autogan_model, negative)

    #test discriminator
    test_discriminator(autogan_model, negative, positive)
