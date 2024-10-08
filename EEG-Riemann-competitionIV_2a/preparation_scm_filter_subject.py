'''
#####################################################################################################################
Discription: Preprocessing of BCIC 4 2a(calculating SCM and Riemannian mean)
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
warnings.filterwarnings("ignore")
device = torch.device('cuda:0')

isfilter = True
timelap = True

save_scm_overlap = True
cal_riemann_mean = True
cal_normalization_coef = True
scm_norm = True
cal_riemann_mean_norm = True

subject_list = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']

inp_dim = 22
num_classes = 4

raw_path = 'data/bcidatasetIV2a-master'
# scm_path = 'data/scm'
if timelap:
    scm_path = 'data/scm_filter_lap'
    scm_norm_path = 'data/scm_filter_lap_norm_subject'
else:
    scm_path = 'data/scm_filter'
    scm_norm_path = 'data/scm_filter_norm_subject'

for subject_id in subject_list:
    inp_stats_file = f"data/subject/input_norm_states_scm_filter_{subject_id}.norm"

    riemann_mean_path = f'data/subject/all_riemann_mean_filter_{subject_id}.mat'
    riemann_mean_norm_path = f'data/subject/all_riemann_mean_filter_norm_{subject_id}.mat'

    lowcut = 8.0  # 低频截止频率（Hz）
    highcut = 30.0  # 高频截止频率（Hz）
    nyq = 0.5 * 250  # Nyquist频率为采样频率的一半
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(N=5, Wn=[low, high], btype='band')

    opt = {
        'time_block': 0.5,
        'fs': 250,
        'rm_transition': 256,
        'overlap': 0
    }
    start_pos = 782
    end_pos = 1437
    if save_scm_overlap:
        files = glob.glob(raw_path + '/' + f'{subject_id}*.npz')
        subject_eeg_data = []
        subject_labels = []
        subject_scm = []
        for file in files:
            file_name, _ = os.path.splitext(os.path.basename(file))
            dataset = MotorImageryDataset(file)
            eeg_data, labels = dataset.get_trials_from_channel()
            eeg_data = np.stack(eeg_data, axis=-1)   # (22,1875,288)
            eeg_data = eeg_data[:, start_pos:end_pos, :]   # (22,750,288)
            if isfilter:
                filtered_eeg_data = np.zeros_like(eeg_data)
                for i in range(eeg_data.shape[0]):  # Iterate over each channel
                    for j in range(eeg_data.shape[2]):  # Iterate over each epoc
                        channel_data = eeg_data[i, :, j]
                        filtered_channel_data = filtfilt(b, a, channel_data)
                        filtered_eeg_data[i, :, j] = filtered_channel_data
                eeg_data = filtered_eeg_data
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
                interval = 250 // 5     # 50
                len_new_samples = 250    # 1s
                # split_points = [int(i * interval) for i in range(num_lap + 1)]
                num_raw_samples = eeg_data.shape[-1]
                eeg_data_lap = np.empty((inp_dim, len_new_samples, num_raw_samples * num_lap))
                for t in range(num_lap):
                    eeg_data_lap[:, :, t * num_raw_samples:(t + 1) * num_raw_samples] = eeg_data[:, t * interval:t * interval + len_new_samples, :]
                eeg_data = eeg_data_lap
                labels = np.tile(labels, num_lap)

            scm = np.zeros((eeg_data.shape[-1], inp_dim, inp_dim), dtype=np.float32)
            for i in range(eeg_data.shape[-1]):
                scm[i] = eeg_data[:, :, i] @ eeg_data[:, :, i].T
                scm[i] = scm[i] / eeg_data.shape[1]     # /256
                # scm[i] = scm[i] / np.trace(scm[i])
            sio.savemat(scm_path + '/' + file_name + '.mat', {'scm': scm, 'labels': labels})

    if cal_riemann_mean:
        files = glob.glob(scm_path + '/' + f'{subject_id}*.mat')
        all_scm = []
        all_labels = []
        for f in files:
            data = sio.loadmat(f)
            all_scm.append(data['scm'])
            all_labels.append(data['labels'])
        all_scm = np.concatenate(all_scm, axis=0)
        all_labels = np.concatenate(all_labels, axis=1)

        riemann_mean = utils.riemann_mean_by_all_classes(all_scm, all_labels)  # change from 'J' to 'S'
        sio.savemat(riemann_mean_path, {'riemann_mean': riemann_mean})

    if cal_normalization_coef:
        files = glob.glob(scm_path + '/' + f'{subject_id}*.mat')
        all_scm = []
        all_labels = []
        for f in files:
            data = sio.loadmat(f)
            all_scm.append(data['scm'])
            all_labels.append(data['labels'])
        all_scm = np.concatenate(all_scm, axis=0)
        all_labels = np.concatenate(all_labels, axis=1)
        all_data = np.reshape(all_scm, (-1, 1))
        real_scaler = preprocessing.StandardScaler().fit(all_data)
        norm_matrix = np.vstack((real_scaler.mean_, real_scaler.scale_))
        norm_matrix.tofile(inp_stats_file)

    if scm_norm:
        norm_matrix = np.fromfile(inp_stats_file, dtype=np.float64)
        real_scaler_mean = norm_matrix[0]
        real_scaler_scale = norm_matrix[1]
        files = glob.glob(scm_path + '/' + f'{subject_id}*.mat')
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