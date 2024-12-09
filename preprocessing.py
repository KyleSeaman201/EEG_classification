import pandas as pd
import os
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA, create_eog_epochs
from sklearn.decomposition import FastICA
from sklearn.utils import resample
from scipy.signal import cwt, morlet
from scipy.signal import butter, filtfilt
import numpy as np
import pywt
from scipy.interpolate import interp1d
import random
import os
import time
import shutil
from scipy.fft import fft, ifft
from scipy import interpolate
from PIL import Image

def load_Cap64_dataset():
    splits = {'train': 'train.csv', 'test': 'test.csv'}

    # Download train data
    if not os.path.exists("datasets/MindBigData2022_VisMNIST_Cap64_train.csv"):
        os.makedirs("datasets", exist_ok=True)
        print("downloading MindBigData2022_VisMNIST_Cap64_train ...")
        MindBigData2022_VisMNIST_Cap64_train_train = pd.read_csv("hf://datasets/DavidVivancos/MindBigData2022_VisMNIST_Cap64/" + splits["train"])
        MindBigData2022_VisMNIST_Cap64_train_train.to_csv('datasets/MindBigData2022_VisMNIST_Cap64_train.csv', index=False)
        print("MindBigData2022_VisMNIST_Cap64_train downloaded")
    else:
        print("MindBigData2022_VisMNIST_Cap64_train already downloaded")
    
    # Download test data
    if not os.path.exists("datasets/MindBigData2022_VisMNIST_Cap64_test.csv"):
        os.makedirs("datasets", exist_ok=True)
        print("downloading MindBigData2022_VisMNIST_Cap64_test ...")
        MindBigData2022_VisMNIST_Cap64_train_test = pd.read_csv("hf://datasets/DavidVivancos/MindBigData2022_VisMNIST_Cap64/" + splits["test"])
        MindBigData2022_VisMNIST_Cap64_train_test.to_csv('datasets/MindBigData2022_VisMNIST_Cap64_test.csv', index=False)
        print("MindBigData2022_VisMNIST_Cap64_test downloaded")
    else:
        print("MindBigData2022_VisMNIST_Cap64_test already downloaded")

def load_Muse2_dataset():
    splits = {'train': 'train.csv', 'test': 'test.csv'}

    # Download train data
    if not os.path.exists("datasets/MindBigData2022_VisMNIST_MU2_train.csv"):
        os.makedirs("datasets", exist_ok=True)
        print("downloading MindBigData2022_VisMNIST_MU2_train ...")
        MindBigData2022_VisMNIST_Cap64_train_train = pd.read_csv("hf://datasets/DavidVivancos/MindBigData2022_VisMNIST_MU2/" + splits["train"])
        MindBigData2022_VisMNIST_Cap64_train_train.to_csv('datasets/MindBigData2022_VisMNIST_MU2_train.csv', index=False)
        print("MindBigData2022_VisMNIST_MU2_train downloaded")
    else:
        print("MindBigData2022_VisMNIST_MU2_train already downloaded")
    
    # Download test data
    if not os.path.exists("datasets/MindBigData2022_VisMNIST_MU2_test.csv"):
        os.makedirs("datasets", exist_ok=True)
        print("downloading MindBigData2022_VisMNIST_MU2_test ...")
        MindBigData2022_VisMNIST_Cap64_train_test = pd.read_csv("hf://datasets/DavidVivancos/MindBigData2022_VisMNIST_MU2/" + splits["test"])
        MindBigData2022_VisMNIST_Cap64_train_test.to_csv('datasets/MindBigData2022_VisMNIST_MU2_test.csv', index=False)
        print("MindBigData2022_VisMNIST_MU2_test downloaded")
    else:
        print("MindBigData2022_VisMNIST_MU2_test already downloaded")

def load_Mnist_EP_dataset():
    splits = {'train': 'train.csv', 'test': 'test.csv'}

    # Download train data
    if not os.path.exists("datasets/MindBigData2022_MNIST_EP_train.csv"):
        os.makedirs("datasets", exist_ok=True)
        print("downloading MindBigData2022_MNIST_EP_train ...")
        MindBigData2022_VisMNIST_Cap64_train_train = pd.read_csv("hf://datasets/DavidVivancos/MindBigData2022_MNIST_EP/" + splits["train"])
        MindBigData2022_VisMNIST_Cap64_train_train.to_csv('datasets/MindBigData2022_MNIST_EP_train.csv', index=False)
        print("MindBigData2022_MNIST_EP_train downloaded")
    else:
        print("MindBigData2022_MNIST_EP_train already downloaded")
    
    # Download test data
    if not os.path.exists("datasets/MindBigData2022_MNIST_EP_test.csv"):
        os.makedirs("datasets", exist_ok=True)
        print("downloading MindBigData2022_MNIST_EP_test ...")
        MindBigData2022_VisMNIST_Cap64_train_test = pd.read_csv("hf://datasets/DavidVivancos/MindBigData2022_MNIST_EP/" + splits["test"])
        MindBigData2022_VisMNIST_Cap64_train_test.to_csv('datasets/MindBigData2022_MNIST_EP_test.csv', index=False)
        print("MindBigData2022_MNIST_EP_test downloaded")
    else:
        print("MindBigData2022_MNIST_EP_test already downloaded")

def load_Mnist_MU_dataset():
    splits = {'train': 'train.csv', 'test': 'test.csv'}

    # Download train data
    if not os.path.exists("datasets/MindBigData2022_MNIST_MU_train.csv"):
        os.makedirs("datasets", exist_ok=True)
        print("downloading MindBigData2022_MNIST_MU_train ...")
        MindBigData2022_VisMNIST_Cap64_train_train = pd.read_csv("hf://datasets/DavidVivancos/MindBigData2022_MNIST_MU/" + splits["train"])
        MindBigData2022_VisMNIST_Cap64_train_train.to_csv('datasets/MindBigData2022_MNIST_MU_train.csv', index=False)
        print("MindBigData2022_MNIST_MU_train downloaded")
    else:
        print("MindBigData2022_MNIST_MU_train already downloaded")
    
    # Download test data
    if not os.path.exists("datasets/MindBigData2022_MNIST_MU_test.csv"):
        os.makedirs("datasets", exist_ok=True)
        print("downloading MindBigData2022_MNIST_MU_test ...")
        MindBigData2022_VisMNIST_Cap64_train_test = pd.read_csv("hf://datasets/DavidVivancos/MindBigData2022_MNIST_MU/" + splits["test"])
        MindBigData2022_VisMNIST_Cap64_train_test.to_csv('datasets/MindBigData2022_MNIST_MU_test.csv', index=False)
        print("MindBigData2022_MNIST_MU_test downloaded")
    else:
        print("MindBigData2022_MNIST_MU_test already downloaded")

# WARNING: this dataset is extremely large and will take a LONG time downloading (14GB).
def load_mnist_2B_dataset():
    splits = {'train': 'train.csv', 'test': 'test.csv'}

    # Download train data
    if not os.path.exists("datasets/MindBigData2023_MNIST_2B_train.csv"):
        os.makedirs("datasets", exist_ok=True)
        print("downloading MindBigData2023_MNIST_2B_train ...")
        MindBigData2023_MNIST_2B_train = pd.read_csv("hf://datasets/DavidVivancos/MindBigData2023_MNIST-2B/" + splits["train"])
        MindBigData2023_MNIST_2B_train.to_csv('datasets/MindBigData2023_MNIST_2B_train.csv', index=False)
        print("MindBigData2023_MNIST_2B_train downloaded")
    else:
        print("MindBigData2023_MNIST_2B_train already downloaded")
    
    # Download test data
    if not os.path.exists("datasets/MindBigData2023_MNIST_2B_test.csv"):
        os.makedirs("datasets", exist_ok=True)
        print("downloading MindBigData2023_MNIST_2B_test ...")
        MindBigData2023_MNIST_2B_test = pd.read_csv("hf://datasets/DavidVivancos/MindBigData2023_MNIST-2B/" + splits["test"])
        MindBigData2023_MNIST_2B_test.to_csv('datasets/MindBigData2023_MNIST_2B_test.csv', index=False)
        print("MindBigData2023_MNIST_2B_test downloaded")
    else:
        print("MindBigData2023_MNIST_2B_test already downloaded")




def get_data(dataset):
    df = pd.read_csv(dataset)
    #print(df.values.shape)

    labels = df.iloc[:, 0].values
    signals = df.iloc[:, 1:].values

    return labels, signals

def downsample_blanks(data, labels, target_class=10, downsample_fraction=0.15):
    # Extract indices for the target class
    target_indices = np.where(labels == target_class)[0]

    # Calculate the number of samples to keep
    num_to_keep = int(len(target_indices) * downsample_fraction)

    # Randomly select indices to keep
    selected_indices = np.random.choice(target_indices, num_to_keep, replace=False)

    # Extract the downsampled data and labels
    downsampled_data = data[selected_indices]
    downsampled_labels = labels[selected_indices]

    # Extract data and labels for all other classes
    other_indices = np.where(labels != target_class)[0]
    other_data = data[other_indices]
    other_labels = labels[other_indices]

    # Combine downsampled data with the rest of the data
    data = np.concatenate([downsampled_data, other_data], axis=0)
    labels = np.concatenate([downsampled_labels, other_labels], axis=0)

    return data, labels

def downsample_data(data, labels, number_of_classes, min_samples_per_class):
    # Split data by class
    class_data = {}
    for class_label in range(number_of_classes):
        class_data[class_label] = data[labels == class_label]

    # Determine the minimum number of samples per class
    min_samples_per_class = min_samples_per_class

    # Downsample each class to the minimum number of samples
    downsampled_data = []
    downsampled_labels = []

    for class_label in class_data:
        downsampled_class_data = resample(class_data[class_label], 
                                        n_samples=min_samples_per_class, 
                                        random_state=0)
        downsampled_data.append(downsampled_class_data)
        downsampled_labels.append(np.full(min_samples_per_class, class_label))

    # Combine downsampled data and labels
    downsampled_data = np.concatenate(downsampled_data)
    downsampled_labels = np.concatenate(downsampled_labels)
    return downsampled_data, downsampled_labels

def get_class_distribution(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique_labels, counts))
    print(distribution)
    return distribution

"""
:param eeg_data: dataframe of shape (rows, eeg data) i.e. (1861, 25600)
:returns data: numpy array of shape (rows, number of channels , channel data) i.e (1861, 64, 400)
"""
def reshape_eeg_data(eeg_data, channels, time):
    data = eeg_data.reshape(eeg_data.shape[0], time, channels).transpose(0, 2, 1)
    return data

def plot_data(eeg_data):
    eeg_data = np.asarray(eeg_data)
    
    # Check for NaN or Inf values
    if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
        print("Data contains NaN or Inf values. Please check the preprocessing steps.")
        return
    
    # Scaling the data (if necessary)
    #eeg_data *= 1e-12  # Assuming data might need to be converted to µV

    # Create MNE info structure
    n_channels = eeg_data.shape[0]
    channel_names = [f'EEG {i+1}' for i in range(n_channels)]
    channel_types = ['eeg'] * n_channels
    sfreq = 128  # Sampling frequency in Hz
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)

    # Create Raw object and plot data
    raw = mne.io.RawArray(eeg_data, info)
    raw.plot(n_channels=n_channels, scalings='auto', title='EEG Data', show=True, block=True)

def remove_frequency(eeg_data, lower_band, higher_band):
    mne.set_log_level(50)
    n_channels = eeg_data.shape[0]
    channel_names = [f'EEG {i+1}' for i in range(n_channels)]
    channel_types = ['eeg'] * n_channels
    sfreq = 200  # Sampling frequency in Hz
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)

    raw = mne.io.RawArray(eeg_data, info)
    raw.filter(l_freq=lower_band, h_freq=higher_band)
    raw.notch_filter(freqs=50)
    return raw.get_data()

"""
:param eeg_data: sample of eeg_data of shape (64, 400)
:param type: "band" for bandpass filter; "high" for highpass filter
:returns filtered_data: dictionary of 6 bands, each with array of size (64, 400)
"""
def apply_filter(eeg_data, type="band"):
    fs = 200  # Sampling frequency in Hz
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma_low': (30, 49),
        'gamma_high': (51, 70)
    }

    # Dictionary to store filtered data
    filtered_data = {}

    # Apply bandpass filters to extract frequency bands
    for band_name, (lowcut, highcut) in bands.items():
        if(type == "band"):
            filtered_data[band_name] = _apply_bandpass_filter(eeg_data, lowcut, highcut, fs)
        elif(type == "high"):
            filtered_data[band_name] = _apply_highpass_filter(eeg_data, highcut, fs)
    
    return filtered_data

# Function to apply bandpass filter to EEG data
def _apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Function to apply highpass filter to EEG data
def _apply_highpass_filter(data, highcut, fs, order=5):
    nyquist = 0.5 * fs
    high = highcut / nyquist
    b, a = butter(order, high, btype='high')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def apply_single_filter(eeg_data, low=3, high=49):
    mne.set_log_level(50)
    n_channels = eeg_data.shape[0]
    channel_names = [f'EEG {i+1}' for i in range(n_channels)]
    channel_types = ['eeg'] * n_channels
    sfreq = 128  # Sampling frequency in Hz
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)

    raw = mne.io.RawArray(eeg_data, info)
    raw.filter(l_freq=low, h_freq=high)
    raw.notch_filter(freqs=50)
    return raw.get_data()

def apply_ica(eeg_data, components=5):
    mne.set_log_level(50)
    eeg_data = eeg_data.T

    # Create MNE info structure
    n_channels = eeg_data.shape[0]
    channel_names = [f'EEG {i+1}' for i in range(n_channels)]
    channel_types = ['eeg'] * n_channels
    sfreq = 128  # Sampling frequency in Hz
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)

    # plot data
    raw = mne.io.RawArray(eeg_data, info)
    #raw.filter(l_freq=14.0, h_freq=40.0)

    ica = ICA(n_components=components, random_state=0, max_iter='auto')

    # Fit ICA on the raw EEG data
    ica.fit(raw)

    return ica.get_components().T

def time_frequency(data, cmwX, nKern):
    ''''
    Function to calculate time-frequency representation of multichannel data.

    Citation: This function was gathered from the following resource:
        https://dxganta.medium.com/decoding-thoughts-with-deep-learning-eeg-based-digit-detection-using-cnns-cdf7eee20722

    Parameters:
    data : ndarray
        The EEG data, array of shape (channels, time).
    cmwX : ndarray
        The Fourier coefficients of the complex Morlet wavelets, array of shape (frequencies, nConv).
    nKern : int
        The length of the wavelet kernel.
    channel_labels : list, optional
        The labels of the EEG channels. Must be the same length as the number of channels in the data.
        If not provided, no channel labels will be used.

    Returns:
    tf : ndarray
        The time-frequency representation of the data, array of shape (frequencies, time).
        This is the average power across all channels.
    '''

    # set up convolution parameters
    nData   = data.shape[1]
    nConv   = nData + nKern - 1
    halfwav = (nKern-1)//2

    # initialize time-frequency output matrix
    tf = np.zeros((data.shape[0], cmwX.shape[0], data.shape[1])) # channels X frequency X times

    # loop over channels
    for chani in range(data.shape[0]):

        # compute Fourier coefficients of EEG data
        eegX = fft(data[chani, :] , nConv)

        # perform convolution and extract power (vectorized across frequencies)
        as_ = ifft(cmwX * eegX[None, :], axis=1)
        as_ = as_[:, halfwav: -halfwav]
        tf[chani, :, :] = np.abs(as_) ** 2

    return tf


def get_cmwX(nData, srate, freqrange=[1,40], numfrex=42):
    '''
    Function to calculate the Fourier coefficients of complex Morlet wavelets.

    Citation: This function was gathered from the following resource:
        https://dxganta.medium.com/decoding-thoughts-with-deep-learning-eeg-based-digit-detection-using-cnns-cdf7eee20722

    Parameters:
    nData : int
        The number of data points.
    freqrange : list, optional
        The range of frequencies to consider. Defaults to [1,40].
    numfrex : int, optional
        The number of frequencies between the lowest and highest frequency. Defaults to 42.

    Returns:
    cmwX : ndarray
        The Fourier coefficients of the complex Morlet wavelets, array of shape (frequencies, nConv).
    nKern : int
        The length of the wavelet kernel.
    frex : ndarray
        The array of frequencies.
    '''
    pi = np.pi
    wavtime = np.arange(-2,2-1/srate,1/srate)
    nKern = len(wavtime)
    nConv = nData + nKern - 1
    frex = np.linspace(freqrange[0],freqrange[1],numfrex)
   # create complex morlet wavelets array
    cmwX = np.zeros((numfrex, nConv), dtype=complex)

    # number of cycles
    numcyc = np.linspace(3,15,numfrex);
    for fi in range(numfrex):
        # create time-domain wavelet
        s = numcyc[fi] / (2*pi*frex[fi])
        twoSsquared = (2*s) ** 2
        cmw = np.exp(2*1j*pi*frex[fi]*wavtime) * np.exp( (-wavtime**2) / twoSsquared )


        # compute fourier coefficients of wavelet and normalize
        cmwX[fi, :] = fft(cmw, nConv)
        cmwX[fi, :] = cmwX[fi, :] / max(cmwX[fi, :])

    return cmwX, nKern, frex

def save_cmwt(labels, tf, times, frex):
    print("saving images ...")
    folder_path =  "images/"
    os.makedirs(folder_path, exist_ok=True)
    for i in range(tf.shape[0]):
        if (os.path.isfile(folder_path + str(i) + ".png")):
            print(str(i)+".png already exists ... skipping to next file")
            continue

        code = 0 if labels[i] == 0 else 1
        fig, ax = plt.subplots()
        ax.contourf(times, frex, tf[i,:,:], 40, cmap='jet')
        ax.axis('off')

        # Save the figure as a PNG file with a tight bounding box
        file_path = f"{folder_path}/{i}.png"
        fig.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def plot_sample_cmwt(labels, tf, times, frex):
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i,ax in enumerate(axs.flat):
        x = random.randint(0, tf.shape[0])
        contour = ax.contourf(times, frex, tf[x,:,:], 40, cmap='jet')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequencies (Hz)')
        ax.set_title(f"Time Frequency Plot for {'non-digit' if labels[x] == 0 else 'digit'}")
    fig.savefig("cwt_sample.png")

def get_images(window=(369, 496)):
    print("getting images ...")
    DIR = './images'
    num_files = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    images = np.zeros((num_files, 3, window[0], window[1]))
    for i in range(num_files):
        # Load image and convert back to array
        image = Image.open(f'images/{i}.png')
        image = image.resize((window[1], window[0]), Image.Resampling.LANCZOS)
        image = np.array(image)
        #print(image.shape)
        image = image[:, :, :3] #Only obtain RGD (The 4th column is transparaceny)
        image = image.transpose(2, 0, 1)
        images[i] = image
    return images

def apply_wavelet_transform(channel_data, plot=False):
    fs = 220  # Sampling frequency
    widths = np.arange(1, 7)  # Range of scales (widths) for the wavelet transform

    # Load or select your EEG data
    signal = channel_data  # Assuming eeg_data is of shape (1861, 3, 64, 400)

    # Perform Continuous Wavelet Transform (CWT)
    cwt_matrix = cwt(signal, morlet, widths)
    #plot_data(cwt_matrix)

    # Calculate the corresponding frequencies
    frequencies = fs / widths

    #Plot the results
    if plot:
        plt.figure(figsize=(10, 6))
        plt.imshow(np.abs(cwt_matrix), extent=[0, 1, frequencies[-1], frequencies[0]],
                cmap='jet', aspect='auto', interpolation='bilinear')
        plt.colorbar(label='Magnitude')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title('Continuous Wavelet Transform using Scipy')
        plt.savefig("wavelet_transform.png")
    return(cwt_matrix)

def apply_dwt(channel_data, plot=False):
    wavelet = 'db4'  # Example wavelet
    level = 4  # Level of decomposition
    coeffs = pywt.wavedec(channel_data, wavelet, level=level)
    cA = coeffs[0]
    cD = coeffs[1:]

    if plot:
        plt.figure(figsize=(10, 8))
        plt.subplot(len(cD) + 2, 1, 1)
        plt.plot(channel_data)
        plt.title('Original Signal')

        # Plot approximation coefficients
        plt.subplot(len(cD) + 2, 1, 2)
        plt.plot(cA)
        plt.title('Approximation Coefficients')

        # Plot detail coefficients
        for i, cd in enumerate(cD, 1):
            plt.subplot(len(cD) + 2, 1, i + 2)
            plt.plot(cd)
            plt.title(f'Detail Coefficients Level {i}')

        plt.tight_layout()
        plt.savefig("dwt.png")
        plt.close()

    return(cA, cD)

# Use after applying DWT
def resample_coeffs(coeffs, target_length):
    resampled_coeffs = []
    for c in coeffs:
        x = np.arange(len(c))
        f = interp1d(x, c, kind='linear')
        x_new = np.linspace(0, len(c) - 1, target_length)
        c_resampled = f(x_new)
        resampled_coeffs.append(c_resampled)
    return np.concatenate(resampled_coeffs)

def apply_PSD(eeg_data):
    mne.set_log_level(50)

    # Create MNE info structure
    n_channels = eeg_data.shape[0]
    channel_names = [f'EEG {i+1}' for i in range(n_channels)]
    channel_types = ['eeg'] * n_channels
    sfreq = 128  # Sampling frequency in Hz
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)

    # Create Raw object and plot data
    raw = mne.io.RawArray(eeg_data, info)
    raw = raw.compute_psd()
    return raw.get_data()
