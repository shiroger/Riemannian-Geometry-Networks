'''
#####################################################################################################################
Discription: filter+overlap
#######################################################################################################################
'''
import scipy.io as scio
from sklearn.utils import shuffle
import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import preprocessing
import scipy.io as sio
import utils
import my_mne_utills
from utils import MotorImageryDataset
import glob
import warnings
from scipy.signal import butter, lfilter
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchmetrics import Accuracy
import mne
from scipy.signal import butter, lfilter, filtfilt
import matplotlib.pyplot as plt
# from torchvision.transforms import ToTensor
import resampy
from my_mne_utills import load_bbci_data

warnings.filterwarnings("ignore")
device = torch.device('cuda:0')

isfilter = True
timelap = True
timesplit = False
normalize_EEG = True

save_scm_overlap = True
cal_riemann_mean = True
cal_normalization_coef = True
scm_norm = False
cal_riemann_mean_norm = True

subject_list = range(1, 15)

inp_dim = 44
num_classes = 4

raw_path = 'data'
# scm_path = 'data/scm'

# if timelap:
#     scm_path = 'data/20-Subjects/scm_filter_lap'
#     scm_norm_path = 'data/20-Subjects/scm_filter_lap_norm_subject'
# else:
#     scm_path = 'data/20-Subjects/scm_filter'
#     scm_norm_path = 'data/20-Subjects/scm_filter_norm_subject'
scm_path = 'data/scm_filter'
scm_norm_path = 'data/scm_filter_norm_subject'

fs = 250
C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
C_sensors_with_prefix = ['EEG ' + sensor for sensor in C_sensors]

for subject_id in subject_list:
    inp_stats_file = f"data/input_norm_states_scm_filter_{subject_id}.norm"

    riemann_mean_path = f'data/all_riemann_mean_filter_{subject_id}.mat'
    riemann_mean_norm_path = f'data/all_riemann_mean_filter_norm_{subject_id}.mat'

    lowcut = 0.1  # 低频截止频率（Hz）
    highcut = 124.9  # 高频截止频率（Hz）
    nyq = 0.5 * 250  # Nyquist频率为采样频率的一半
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(N=5, Wn=[low, high], btype='band')

    opt = {
        'time_block': 0.4,
        'fs': 250,
        'rm_transition': 0,
        'overlap': 0
    }
    start_pos = 782
    end_pos = 1437
    if save_scm_overlap:
        eeg_data = []
        labels = []
        low_cut_hz = 0
        file_train = raw_path + '/train/' + f'{subject_id}.edf'
        file_test = raw_path + '/test/' + f'{subject_id}.edf'
        full_train_set = mne.io.read_raw_edf(file_train, preload=True)
        full_test_set = mne.io.read_raw_edf(file_test, preload=True)
        events_train, event_id_train = mne.events_from_annotations(full_train_set)
        events_test, event_id_test = mne.events_from_annotations(full_test_set)

        eeg_train = full_train_set.pick_channels(C_sensors_with_prefix).get_data()
        eeg_test = full_test_set.pick_channels(C_sensors_with_prefix).get_data()
        eeg_train = resampy.resample(eeg_train, 500, fs, axis=1)
        eeg_test = resampy.resample(eeg_test, 500, fs, axis=1)
        eeg_train = eeg_train/np.mean(eeg_train)
        eeg_test = eeg_test/np.mean(eeg_test)
        # eeg_data = dataset['EEG']   # (84,640,64)
        # eeg_data = eeg_data.transpose(2, 1, 0)  # (64,640,84)
        # labels = dataset['labels']  # (1,84)

        # eeg_data = np.stack(eeg_data, axis=-1)   # (22,1875,288)
        # eeg_data = eeg_data[:, start_pos:end_pos, :]   # (22,750,288)
        if isfilter:
            filtered_eeg_train = np.zeros_like(eeg_train)
            filtered_eeg_test = np.zeros_like(eeg_test)
            for j in range(eeg_train.shape[0]):  # Iterate over each channel
                channel_data = eeg_train[j, :]
                filtered_channel_data = filtfilt(b, a, channel_data)
                filtered_eeg_train[j, :] = filtered_channel_data
                channel_data = eeg_test[j, :]
                filtered_channel_data = filtfilt(b, a, channel_data)
                filtered_eeg_test[j, :] = filtered_channel_data
            eeg_train = filtered_eeg_train
            eeg_test = filtered_eeg_test

        events_train[:, 0] = events_train[:, 0] / 2
        events_test[:, 0] = events_test[:, 0] / 2
        for p in range(events_train.shape[0]):
            eeg_data.append(eeg_train[:, events_train[p, 0]:events_train[p, 0] + 4*fs])
            labels.append(events_train[p, 2])
        for p in range(events_test.shape[0]):
            eeg_data.append(eeg_test[:, events_test[p, 0]:events_test[p, 0] + 4*fs])
            labels.append(events_test[p, 2])
        eeg_data = np.stack(eeg_data)
        labels = np.stack(labels)
        eeg_data = eeg_data.transpose(1, 2, 0)

        if normalize_EEG:
            raw_shape = eeg_data.shape  # (44,1000,trails)
            eeg_data = np.reshape(eeg_data, (raw_shape[0], -1))  # (44,1000,trails)->(44,1000*trails)
            eeg_data = eeg_data - np.mean(eeg_data, 0)
            eeg_data = np.reshape(eeg_data, raw_shape)  # (44,1000*trails)->(44,1000,trails)
            for i in range(inp_dim):
                for j in range(eeg_data.shape[1]):
                    mean_x = np.mean(eeg_data[i, j, :])
                    std_x = np.std(eeg_data[i, j, :])
                    eeg_data[i, j, :] = (eeg_data[i, j, :] - mean_x) / std_x

        if timelap:
            # num_lap = 4     # num_lap + 1 mod by 655, num_lap + 1 segments
            # t_len = end_pos - start_pos
            # # eeg_data_reshaped = eeg_data.reshape(inp_dim, num_lap, t_len//num_lap, -1).transpose(0, 2, 1, 3).reshape(inp_dim, t_len//num_lap, -1)
            # # eeg_data = eeg_data_reshaped
            # # labels = np.tile(labels, num_lap)
            # interval = t_len // (num_lap + 1)
            # len_new_samples = t_len * 2 // (num_lap + 1)
            # # split_points = [int(i * interval) for i in range(num_lap + 1)]
            # num_raw_samples = eeg_data.shape[-1]
            # eeg_data_lap = np.empty((inp_dim, len_new_samples, num_raw_samples * num_lap))
            # for t in range(num_lap):
            #     eeg_data_lap[:, :, t * num_raw_samples:(t + 1) * num_raw_samples] = eeg_data[:, t * interval:t * interval + len_new_samples, :]
            # eeg_data = eeg_data_lap
            # labels = np.tile(labels, num_lap)
            num_lap = 6     # num_lap + 1 segments
            t_len = end_pos - start_pos
            # eeg_data_reshaped = eeg_data.reshape(inp_dim, num_lap, t_len//num_lap, -1).transpose(0, 2, 1, 3).reshape(inp_dim, t_len//num_lap, -1)
            # eeg_data = eeg_data_reshaped
            # labels = np.tile(labels, num_lap)
            interval = 2 * fs // 5     # 0.4s
            len_new_samples = 2 * fs    # 2s
            # split_points = [int(i * interval) for i in range(num_lap + 1)]
            num_raw_samples = eeg_data.shape[-1]
            eeg_data_lap = np.empty((inp_dim, len_new_samples, num_raw_samples * num_lap))
            for t in range(num_lap):
                eeg_data_lap[:, :, t * num_raw_samples:(t + 1) * num_raw_samples] = eeg_data[:, t * interval:t * interval + len_new_samples, :]
            eeg_data = eeg_data_lap
            labels = np.tile(labels, num_lap)
        if timesplit:
            num_lap = 2  # num_lap + 1 segments
            # t_len = end_pos - start_pos
            # eeg_data_reshaped = eeg_data.reshape(inp_dim, num_lap, t_len//num_lap, -1).transpose(0, 2, 1, 3).reshape(inp_dim, t_len//num_lap, -1)
            # eeg_data = eeg_data_reshaped
            # labels = np.tile(labels, num_lap)
            interval = 2 * fs  # 64
            len_new_samples = 2 * fs  # 0.4s
            # split_points = [int(i * interval) for i in range(num_lap + 1)]
            num_raw_samples = eeg_data.shape[-1]
            eeg_data_lap = np.empty((inp_dim, len_new_samples, num_raw_samples * num_lap))
            for t in range(num_lap):
                eeg_data_lap[:, :, t * num_raw_samples:(t + 1) * num_raw_samples] = eeg_data[:,
                                                                                    t * interval:t * interval + len_new_samples,
                                                                                    :]
            eeg_data = eeg_data_lap
            labels = np.tile(labels, num_lap)

        scm = np.zeros((eeg_data.shape[-1], inp_dim, inp_dim), dtype=np.float32)
        for i in range(eeg_data.shape[-1]):
            scm[i] = eeg_data[:, :, i] @ eeg_data[:, :, i].T
            scm[i] = scm[i] / eeg_data.shape[1]     # /256
            # scm[i] = scm[i] / np.trace(scm[i])
        # sio.savemat(scm_path + '/' + file_name + '.mat', {'scm': scm, 'labels': labels})

    if cal_riemann_mean:
        # files = glob.glob(scm_path + '/' + f'Dataset_{subject_id}.mat')
        # all_scm = []
        # all_labels = []
        # # for f in files:
        # #     data = sio.loadmat(f)
        #     all_scm.append(data['scm'])
        #     all_labels.append(data['labels'])
        # all_scm = np.concatenate(all_scm, axis=0)
        # all_labels = np.concatenate(all_labels, axis=1)

        riemann_mean = utils.riemann_mean_by_all_classes(scm, labels)  # change from 'J' to 'S'
        sio.savemat(riemann_mean_path, {'riemann_mean': riemann_mean})

    if cal_normalization_coef:
        # files = glob.glob(scm_path + '/' + f'Dataset_{subject_id}.mat')
        # all_scm = []
        # all_labels = []
        # for f in files:
        #     data = sio.loadmat(f)
        #     all_scm.append(data['scm'])
        #     all_labels.append(data['labels'])
        # all_scm = np.concatenate(all_scm, axis=0)
        # all_labels = np.concatenate(all_labels, axis=1)
        all_data = np.reshape(scm, (-1, 1))
        real_scaler = preprocessing.StandardScaler().fit(all_data)
        norm_matrix = np.vstack((real_scaler.mean_, real_scaler.scale_))
        norm_matrix.tofile(inp_stats_file)

    if scm_norm:
        norm_matrix = np.fromfile(inp_stats_file, dtype=np.float64)
        real_scaler_mean = norm_matrix[0]
        real_scaler_scale = norm_matrix[1]
        files = glob.glob(scm_path + '/' + f'Dataset_{subject_id}.mat')
        for f in files:
            file_name, _ = os.path.splitext(os.path.basename(f))
            data = sio.loadmat(f)
            scm = data['scm']
            labels = data['labels']
            norm_scm = (scm - real_scaler_mean) / real_scaler_scale
            norm_scm = norm_scm.astype(np.float32)
            sio.savemat(scm_norm_path + '/' + file_name + '.mat', {'norm_scm': norm_scm, 'labels': labels})

    if cal_riemann_mean_norm:
        riemann_data = sio.loadmat(riemann_mean_path)
        riemann_mean = riemann_data['riemann_mean']

        norm_matrix = np.fromfile(inp_stats_file, dtype=np.float64)
        real_scaler_mean = norm_matrix[0]
        real_scaler_scale = norm_matrix[1]

        riemann_mean = (riemann_mean - real_scaler_mean) / real_scaler_scale
        riemann_mean = riemann_mean.astype(np.float32)
        sio.savemat(riemann_mean_norm_path, {'riemann_mean': riemann_mean})