import numpy as np 
import dataset_utils


def shuffle_and_save(positives, negatives):

    pos_labels = np.ones(shape=(positives.shape[0]))
    neg_labels = np.zeros(shape=(negatives.shape[0]))

    print("positives", positives)
    print("negatives", negatives)

    data = np.concatenate((negatives, positives))
    labels = np.concatenate((neg_labels, pos_labels))

    print("concatenated data ", data)

    mapper = np.arange(data.shape[0])
    
    np.random.shuffle(mapper)

    data = data[mapper]
    labels = labels[mapper]

    print("shuffled data", data)
    print("labels ", labels)

num_samples = 10

negatives = np.arange(10, 20)
positives = np.arange(0, 10)

shuffle_and_save(positives, negatives)