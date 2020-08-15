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

    from Autoencoder import Autoencoder
    
    ae = Autoencoder(128,128)
    ae.load_model(filename)

    return ae.get_model()

#load  dataset
data = load_dataset("../data/test_cropped_np")/255#[500:516]/255
print(data.shape)
print(np.min(data), np.max(data))

#load model
model = load_model("../checkpoints/weights.25")
predictions = model.predict(data)
predictions /= np.max(predictions)
print("Predictions shape ", predictions.shape)
print("Predictions range", np.min(predictions), np.max(predictions))

#plot input
plt_input = plot(data, 0, 16)

#plot predictions
plt_predictions = plot(predictions, 0, 16)

#show figures
plt_predictions.show()
plt_input.show()


