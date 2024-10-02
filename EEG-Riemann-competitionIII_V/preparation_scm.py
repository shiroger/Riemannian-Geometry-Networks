'''
#####################################################################################################################
Discription: Calculate SCM
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
import glob
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchmetrics import Accuracy

# from torchvision.transforms import ToTensor
warnings.filterwarnings("ignore")
device = torch.device('cuda:0')

save_scm = True
cal_normalization_coef = False
norm_file_save = False
norm_riemann_mean = True

inp_dim = 32
num_classes = 3

raw_path = 'data/raw'
scm_path = 'data/scm'
scm_norm_path = 'data/scm_norm'

riemann_mean_path = 'data/all_riemann_mean.mat'
riemann_mean_norm_path = 'data/all_riemann_mean_norm.mat'

inp_stats_file = "data/input_norm_states_scm.norm"

opt = {
    'time_block': 0.5,
    'fs': 512,
    'rm_transition': 256,
    'overlap': 0
}
if save_scm:
    for folder in os.scandir(raw_path):
        files = glob.glob(folder.path + '/' + 'train*.mat')
        subject_eeg_data = []
        subject_labels = []
        subject_scm = []
        for file in files:
            file_name, _ = os.path.splitext(os.path.basename(file))
            eeg_data, labels = utils.get_data(file, opt)
            subject_eeg_data.append(eeg_data)
            subject_labels.append(labels)

            scm = np.zeros((eeg_data.shape[-1], inp_dim, inp_dim), dtype=np.float32)
            for i in range(eeg_data.shape[-1]):
                scm[i] = eeg_data[:, :, i] @ eeg_data[:, :, i].T
                scm[i] = scm[i] / eeg_data.shape[1]     # /256
            sio.savemat(scm_path + '/' + file_name + '.mat', {'scm': scm, 'labels': labels})

if cal_normalization_coef:
    print('Start calculate normalization coefficient')
    files = glob.glob(scm_path + '/' + '*.mat')
    all_scm = []
    for file in files:
        data = sio.loadmat(file)
        scm = data['scm']
        all_scm.append(scm)

    matrix_array = np.concatenate(all_scm, axis=0)
    combined_matrix = np.stack(matrix_array)    # (总数,32,32)
    all_data = np.reshape(combined_matrix, (-1, 1))
    real_scaler = preprocessing.StandardScaler().fit(all_data)
    norm_matrix = np.vstack((real_scaler.mean_, real_scaler.scale_))
    norm_matrix.tofile(inp_stats_file)

if norm_file_save:
    print('Start normalize and save each file')
    norm_matrix = np.fromfile(inp_stats_file, dtype=np.float64)
    real_scaler_mean = norm_matrix[0]
    real_scaler_scale = norm_matrix[1]
    files = glob.glob(scm_path + '/' + '*.mat')
    for file in files:
        file_name, _ = os.path.splitext(os.path.basename(file))
        data = sio.loadmat(file)
        scm = data['scm']
        labels = data['labels']
        norm_scm = (scm - real_scaler_mean) / real_scaler_scale
        norm_scm = norm_scm.astype(np.float32)
        sio.savemat(scm_norm_path + '/' + file_name + '.mat', {'norm_scm': norm_scm, 'labels': labels})

if norm_riemann_mean:
    print('Start normalize riemann mean')
    riemann_data = sio.loadmat(riemann_mean_path)
    riemann_mean = riemann_data['riemann_mean']

    norm_matrix = np.fromfile(inp_stats_file, dtype=np.float64)
    real_scaler_mean = norm_matrix[0]
    real_scaler_scale = norm_matrix[1]

    riemann_mean = (riemann_mean - real_scaler_mean) / real_scaler_scale
    riemann_mean = riemann_mean.astype(np.float32)
    sio.savemat(riemann_mean_norm_path, {'riemann_mean': riemann_mean})
