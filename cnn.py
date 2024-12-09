import preprocessing_kyle as preprocessing
import cnn
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import pandas as pd
import knn
from PIL import Image

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time
import shutil
from scipy.fft import fft, ifft
from scipy import interpolate
     
def main():
    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("You are using device: %s" % device)

    ### Start of program ###
    preprocessing.load_Mnist_MU_dataset()
    labels, eeg_data = preprocessing.get_data("datasets/MindBigData2022_MNIST_MU_train.csv")

    labels[labels != -1] = 1
    labels[labels == -1] = 0

    eeg_data, labels = preprocessing.downsample_data(eeg_data, labels, 2, 1000)
    preprocessing.get_class_distribution(labels)


    ### Uncomment block below to apply filtering ###
    
    # eeg_data = preprocessing.reshape_eeg_data(eeg_data, channels=4, time=440)
    # for i in range(eeg_data.shape[0]):
    #     eeg_data[i] = preprocessing.remove_frequency(eeg_data[i], 8, 30)
    # eeg_data = eeg_data.transpose(0, 2, 1)
    # eeg_data = eeg_data.reshape(eeg_data.shape[0], eeg_data.shape[1] * eeg_data.shape[2])


    # Produce image transformation per sample
    starting_freq = 1
    end_freq = 6
    num_frequencies = 10

    times = np.linspace(0,2,1760)
    nData = eeg_data.shape[1]
    cmwX, nKern, frex = preprocessing.get_cmwX(nData, srate=220, freqrange=[starting_freq, end_freq], numfrex=num_frequencies)
    tf = preprocessing.time_frequency(eeg_data, cmwX, nKern)

    preprocessing.plot_sample_cmwt(labels, tf, times, frex) # Saves 9 random samples to cwt_sample.png
    preprocessing.save_cmwt(labels, tf, times, frex)
    
    print("Running CNN")
    cnn.train_cnn(labels, device)


if __name__ == "__main__":
    main()