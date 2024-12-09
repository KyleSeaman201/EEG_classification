import sys
import socket
import random

import numpy as np
import pandas as pd

# Pytorch package
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import preprocessing_steve as preprocessing
from utils import train, evaluate, set_seed_nb, unit_test_values, deterministic_init, plot_curves
from Transformer import TransformerClassifier

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_and_plot(device, model, optimizer, scheduler, criterion, filename, EPOCHS,
                   batched_train_data, batched_train_label, batched_valid_data, batched_valid_label, num_classes):
    train_loss_history = []
    train_acc_history= []
    valid_loss_history = []
    valid_acc_history = []

    # batched_train_data = torch.from_numpy(batched_train_data).to(device)
    # batched_train_label = torch.from_numpy(batched_train_label).to(device)
    # batched_valid_data = torch.from_numpy(batched_valid_data).to(device)
    # batched_valid_label = torch.from_numpy(batched_valid_label).to(device)

    # # convert training and validation labels into one-hot vectors
    # batched_train_label = torch.nn.functional.one_hot(batched_train_label, num_classes = num_classes)
    # batched_valid_label = torch.nn.functional.one_hot(batched_valid_label, num_classes = num_classes)

    for epoch_idx in range(EPOCHS):
        print("-----------------------------------")
        print("Epoch %d" % (epoch_idx+1))
        print("-----------------------------------")

        train_loss, avg_train_loss, avg_train_accy = train(model, batched_train_data, batched_train_label, optimizer, criterion, scheduler, device, num_classes)
        scheduler.step(train_loss)

        val_loss, avg_val_loss, avg_valid_accy = evaluate(model, batched_valid_data, batched_valid_label, criterion, device, num_classes)

        train_loss_history.append(avg_train_loss)
        valid_loss_history.append(avg_val_loss)
        train_acc_history.append(avg_train_accy.cpu())
        valid_acc_history.append(avg_valid_accy.cpu())

        print("Training Loss: %.4f. Validation Loss: %.4f. " % (avg_train_loss, avg_val_loss))
        print("Training Accuracy: %.4f. Validation Accuracy: %.4f. " % (avg_train_accy, avg_valid_accy))
        sys.stdout.flush()

    plot_curves(train_loss_history, valid_loss_history, train_acc_history, valid_acc_history, filename)


def do_training(train_labels, train_eeg_data, test_labels, test_eeg_data, VALIDATION_DATA_PERCENT, BATCH_SIZE, EPOCHS):
    #
    # split total training data into training and validation data
    #
    num_training = len(train_eeg_data)
    num_training_minus_valid = int(num_training * (1.0 - VALIDATION_DATA_PERCENT))
    num_valid = num_training - num_training_minus_valid

    print(num_training, num_training_minus_valid, num_valid)
    assert(num_training_minus_valid + num_valid == num_training)

    random_idx = np.arange(num_training)

    #print(randomize, len(randomize))
    random.shuffle(random_idx)

    shuffled_data = train_eeg_data[random_idx]
    shuffled_labels = train_labels[random_idx]

    train_eeg_data_minus_valid = shuffled_data[0:num_training_minus_valid]
    valid_eeg_data = shuffled_data[num_training_minus_valid-1:-1]

    train_labels_minus_valid = shuffled_labels[0:num_training_minus_valid]
    valid_labels = shuffled_labels[num_training_minus_valid-1:-1]

    print(train_eeg_data_minus_valid.shape, valid_eeg_data.shape)
    print(train_labels_minus_valid.shape, valid_labels.shape)
    assert(len(train_eeg_data_minus_valid) == num_training_minus_valid)
    assert(len(valid_eeg_data) == num_valid)
    assert(len(train_labels_minus_valid) == num_training_minus_valid)
    assert(len(valid_labels) == num_valid)
    assert(len(train_eeg_data_minus_valid) + len(valid_eeg_data) == len(train_eeg_data))
    assert(len(train_labels_minus_valid) + len(valid_labels) == len(train_labels))

    #
    # Batch up training data
    #
    num_equal_elements = len(train_eeg_data_minus_valid) // BATCH_SIZE
    #print("train num_equal_elements: ", num_equal_elements)

    batched_train_data = np.split(train_eeg_data_minus_valid[0:num_equal_elements * BATCH_SIZE], num_equal_elements)
    batched_train_label = np.split(train_labels_minus_valid[0:num_equal_elements * BATCH_SIZE], num_equal_elements)

    # if any remaining, we need to add tha remainder manually as a final element in
    # each np array
    remainder = len(train_eeg_data_minus_valid) - num_equal_elements * BATCH_SIZE
    if (remainder > 0):
        #print("remainder that didn't fit: ", remainder)
        batched_train_data.append( np.array(train_eeg_data_minus_valid[num_equal_elements * BATCH_SIZE:].tolist(), dtype=np.float32) )
        batched_train_label.append( np.array(train_labels_minus_valid[num_equal_elements * BATCH_SIZE:].tolist()) )

    print(len(batched_train_data))
    print(type(batched_train_data))
    print(type(batched_train_data[0][0][0][0]))
    print(len(batched_train_label))
    print(type(batched_train_label))
    print(type(batched_train_label[0]))

    #
    # Batch up validation data
    #
    num_equal_elements = len(valid_eeg_data) // BATCH_SIZE
    #print("valid num_equal_elements: ", num_equal_elements)
    if (num_equal_elements > 0):
        batched_valid_data = np.split(valid_eeg_data[0:num_equal_elements * BATCH_SIZE], num_equal_elements)
        batched_valid_label = np.split(valid_labels[0:num_equal_elements * BATCH_SIZE], num_equal_elements)

        # if any remaining, we need to add tha remainder manually as a final element in
        # each np array
        remainder = len(train_eeg_data_minus_valid) - num_equal_elements * BATCH_SIZE
        if (remainder > 0):
            #print("remainder that didn't fit: ", remainder)
            batched_valid_data.append( np.array(valid_eeg_data[num_equal_elements * BATCH_SIZE:].tolist(), dtype=np.float32) )
            batched_valid_label.append( np.array(valid_labels[num_equal_elements * BATCH_SIZE:].tolist()) )
    else:
        # all we have is remainder...
        batched_valid_data = []
        batched_valid_label = []
        batched_valid_data.append( np.array(valid_eeg_data[num_equal_elements * BATCH_SIZE:].tolist(), dtype=np.float32) )
        batched_valid_label.append( np.array(valid_labels[num_equal_elements * BATCH_SIZE:].tolist()) )

    print("len(batched_valid_data): ", len(batched_valid_data))
    print("len(batched_valid_label): ", len(batched_valid_label))

    #
    # Check device availability
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    print("You are using device: %s" % device)


    # code for generating batches for training
    # train_loader = DataLoader(train_eeg_data_minus_valid, batch_size=BATCH_SIZE,
    #                         shuffle=False, collate_fn=generate_batch)
    # valid_loader = DataLoader(valid_eeg_data, batch_size=BATCH_SIZE,
    #                         shuffle=False, collate_fn=generate_batch)
    # test_loader = DataLoader(test_eeg_data, batch_size=BATCH_SIZE,
    #                     shuffle=False, collate_fn=generate_batch)

    # Get the input and the output sizes for model
    # train_eeg_data: (1861, 64, 400) (num_data, num_sensors, num_time_slice_samples)
    # our resolution of the model should be "num_sensors" to match the vocabulary size of an NLM.
    input_size = train_eeg_data_minus_valid.shape[1]

    # output size is the number of unique classes we want to classify against
    # this just the number of unique entries in the labels
    num_classes = output_size = len(np.unique(train_labels))
    print (input_size, output_size)

    # Hyperparameters
    learning_rate = 1e-3

    # Model
    # max_length is the maximum length of an input sequence
    max_seq_length = train_eeg_data_minus_valid.shape[2]

    trans_model = TransformerClassifier(input_size = input_size,
                                        output_size = output_size,
                                        num_heads=2,
                                        num_layers=1,
                                        hidden_dim=256,
                                        dim_feedforward=2048,
                                        device = device,
                                        max_length = max_seq_length).to(device)
    # trans_model = CnnClassifierSteve(
    #     input_size = input_size,
    #     output_size = output_size,
    #     device = device
    #     ).to(device)

    # optimizer and loss calc
    optimizer = torch.optim.Adam(trans_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.CrossEntropyLoss()

    hostname = socket.gethostname()
    print("Current hostname: ", hostname)

    filename='EEG_transformer__' + hostname
    print(filename)
    train_and_plot(device, trans_model, optimizer, scheduler, criterion, filename, EPOCHS,
                   batched_train_data, batched_train_label, batched_valid_data, batched_valid_label, num_classes)


def main():
    RANDOM_SEED = 12345
    seed_torch(RANDOM_SEED)

    BATCH_SIZE = 128
    EPOCHS = 140
    VALIDATION_DATA_PERCENT = 0.10

    ### Start of program ###
    preprocessing.load_small_dataset()

    #
    # load training data
    #
    train_labels, train_images, train_eeg_data = preprocessing.get_data("datasets/MindBigData2022_VisMNIST_Cap64_train.csv")

    train_eeg_data = preprocessing.reshape_eeg_data(train_eeg_data) # returns np array of size (1861, 64, 400)
    print(type(train_eeg_data))

    train_eeg_data = np.float32(train_eeg_data)
    #preprocessing.plot_data(train_eeg_data[0]) # Plot raw data of first sample

    print("train_eeg_data.type: ", type(train_eeg_data))
    print("train_eeg_data.shape: ", train_eeg_data.shape)
    #print("train_eeg_data: ", train_eeg_data)
    print("train_labels: ", train_labels)
    print("train[0] data/label: ", train_eeg_data[0], train_labels[0])

    train_labels = np.array(train_labels.to_list())
    print("unique train_labels: ", np.unique(train_labels))

    # Apply filter to first sample to get different bands
    # TODO: try training with different bands later
    train_filtered_data = preprocessing.apply_filter(train_eeg_data, type="band")
    #preprocessing.plot_data(train_filtered_data['delta'])

    #print("train_filtered_data: ", train_filtered_data)
    #print("train_filtered_data.shape: ", train_filtered_data.shape)
    print("train_filtered_data.type: ", type(train_filtered_data))
    print("train_filtered_data['delta'].type: ", type(train_filtered_data['delta']))
    print("train_filtered_data['delta'].shape: ", train_filtered_data['delta'].shape)
    #print(train_filtered_data['delta'])

    #
    # Bands available: alpha, beta, delta, theta, gamma_low, gamma_high
    # Cat 64 features of selected bands together as separate features
    # 
    new_train_eeg_data = None
    #training_data_bands_set = ['alpha', 'beta', 'delta', 'theta', 'gamma_low', 'gamma_high']
    training_data_bands_set = ['delta', 'theta']
    new_train_eeg_data = train_filtered_data[training_data_bands_set[0]]
    print("Added band: ", training_data_bands_set[0])
    print(new_train_eeg_data.shape)
    for a_band_name in training_data_bands_set[1:]:
        new_train_eeg_data = np.concatenate((new_train_eeg_data, train_filtered_data[a_band_name]), axis=1)
        print("Added band: ", a_band_name)
        print(new_train_eeg_data.shape)

    train_eeg_data = new_train_eeg_data
    train_eeg_data = np.float32(train_eeg_data)

    #
    # load test data
    #
    test_labels, test_images, test_eeg_data = preprocessing.get_data("datasets/MindBigData2022_VisMNIST_Cap64_test.csv")
    test_labels[test_labels == -1] = 10

    test_eeg_data = preprocessing.reshape_eeg_data(test_eeg_data) # returns np array of size (1861, 64, 400)
    #preprocessing.plot_data(train_eeg_data[0]) # Plot raw data of first sample
    test_eeg_data = np.float32(test_eeg_data)

    print("test_eeg_data.type: ", type(test_eeg_data))
    print("test_eeg_data.shape: ", test_eeg_data.shape)
    #print("test_eeg_data: ", test_eeg_data)
    print("test[0] data/label: ", test_eeg_data[0], test_labels[0])

    test_labels = np.array(test_labels.to_list())
    print("unique test_labels: ", np.unique(test_labels))

    # do training with these data sets
    do_training(train_labels, train_eeg_data, test_labels, test_eeg_data, VALIDATION_DATA_PERCENT, BATCH_SIZE, EPOCHS)


if __name__ == "__main__":
    main()