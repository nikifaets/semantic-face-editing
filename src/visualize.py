import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
from dataset_utils import load_dataset

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

from mpl_toolkits.axes_grid1 import ImageGrid

def plot(dataset, start, end):

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

    autogan = AutoGAN(128,128)
    autogan_model = autogan.get_autogan_model()
    autogan_model.load_weights(filename)

    return autogan_model

def test_autoencoder_output(filename, data):

    autogan_model = load_model(filename)
    ae_model = autogan_model.get_layer("autoencoder")
    print(ae_model)
    pred = ae_model.predict(data)
    pred /= np.max(pred)

    plt_input = plot(data,0, 16)
    plt_pred = plot(pred, 0, 16)
    plt_input.show()
    plt_pred.show()
    #ae_in = ae_model.get_layer("autoencoder_input")
    #ae_out =  ae_model.get_layer("autoencoder_output")

    
def test_discriminator(filename, negative, positive):

    autogan_model = load_model(filename)
    disc_model = autogan_model.get_layer("discriminator")
    #disc_in = disc_model.get_layer("discriminator_input")
    #disc_out = disc_model.get_layer("discriminator_output")
    pred_neg = disc_model.predict(negative)
    pred_pos = disc_model.predict(positive)
    
    print("neg range", np.min(pred_neg), np.max(pred_neg))
    print("pos range", np.min(pred_pos), np.max(pred_pos))

    np.floor(pred_neg + 0.5)
    np.floor(pred_pos + 0.5)

    print("negatives", negative.shape[0])
    print("correct negatives", pred_neg[(pred_neg<1)].shape[0])

    print("positives", positive.shape[0])
    print("correct positives", pred_pos[(pred_pos>=0.99)].shape[0])


if __name__ == '__main__':

    #test autoencoder
    data = load_dataset("../data/test_cut_np")/255#[500:516]/255
    print(data.shape)
    print(np.min(data), np.max(data))
    #test_autoencoder_output("../checkpoints_autogan/epoch3", data)

    #test discriminator
    positive = load_dataset("../data/test_np")[:2000]/255
    negative = load_dataset("../data/test_cut_np")[:2000]/255
    test_discriminator("../checkpoints_autogan/epoch9", negative, positive)

    
    #load model
    '''model = load_model("../checkpoints/weights.25")
    predictions = model.predict(data)
    predictions /= np.max(predictions)
    print("Predictions shape ", predictions.shape)
    print("Predictions range", np.min(predictions), np.max(predictions))'''

    '''#plot input
    plt_input = plot(data, 0, 16)

    #plot predictions
    plt_predictions = plot(predictions, 0, 16)

    #show figures
    plt_predictions.show()
    plt_input.show()'''


