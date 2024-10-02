'''
#####################################################################################################################
Discription: Calculate Riemannian mean
#####################################################################################################################
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

cal_riemann_mean = True
cal_normalization_coef = True
norm_file_save = True

inp_dim = 32
num_classes = 2

scm_path = 'data/scm'
riemann_scm_norm_path = 'data/scm_all_riemann_norm'

riemann_mean_path = 'data/all_riemann_mean.mat'
inp_stats_file = "data/input_norm_states_scm_all_riemann.norm"

opt = {
    'time_block': 0.5,
    'fs': 512,
    'rm_transition': 256,
    'overlap': 0
}
if cal_riemann_mean:
    files = glob.glob(scm_path + '/' + '*.mat')
    all_scm = []
    all_labels = []
    for f in files:
        data = sio.loadmat(f)
        all_scm.append(data['scm'])
        all_labels.append(data['labels'])
    all_scm = np.concatenate(all_scm, axis=0)
    all_labels = np.concatenate(all_labels, axis=1)

    riemann_mean = utils.riemann_mean_by_all_classes(all_scm, all_labels)
    sio.savemat(riemann_mean_path, {'riemann_mean': riemann_mean})

if cal_normalization_coef:
    print('Start calculate normalization coefficient')
    riemann_data = sio.loadmat(riemann_mean_path)
    riemann_mean = riemann_data['riemann_mean']

    files = glob.glob(scm_path + '/' + '*.mat')
    all_scm = []
    all_labels = []
    for file in files:
        data = sio.loadmat(file)
        scm = data['scm']
        all_scm.append(scm)
        all_labels.append(data['labels'])

    combined_matrix = np.concatenate(all_scm, axis=0)  # (总数,32,32)
    all_labels = np.concatenate(all_labels, axis=1)    # (1,总数)

    # hand_idx = np.where((all_labels == 2) | (all_labels == 3))
    # scm_hand = combined_matrix[hand_idx[1]]
    # tangent_hand = utils.tangent_space_mapping_2D_f32(scm_hand, riemann_mean[0])
    # unhand_idx = np.where(all_labels == 6)
    # scm_unhand = combined_matrix[unhand_idx[1]]
    # tangent_unhand = utils.tangent_space_mapping_2D_f32(scm_unhand, riemann_mean[1])

    tangent_left = utils.tangent_space_mapping_2D_f32(combined_matrix, riemann_mean[0])
    tangent_right = utils.tangent_space_mapping_2D_f32(combined_matrix, riemann_mean[1])
    tangent_word = utils.tangent_space_mapping_2D_f32(combined_matrix, riemann_mean[2])
    # tangent_hand = np.stack([tangent_left, tangent_right])

    all_data_left = np.reshape(tangent_left, (-1, 1))
    real_scaler = preprocessing.StandardScaler().fit(all_data_left)
    norm_matrix_left = np.vstack((real_scaler.mean_, real_scaler.scale_))

    all_data_right = np.reshape(tangent_right, (-1, 1))
    real_scaler = preprocessing.StandardScaler().fit(all_data_right)
    norm_matrix_right = np.vstack((real_scaler.mean_, real_scaler.scale_))

    all_data_word = np.reshape(tangent_word, (-1, 1))
    real_scaler = preprocessing.StandardScaler().fit(all_data_word)
    norm_matrix_word = np.vstack((real_scaler.mean_, real_scaler.scale_))

    all_data_scm = np.reshape(combined_matrix, (-1, 1))
    real_scaler = preprocessing.StandardScaler().fit(all_data_scm)
    norm_matrix_scm = np.vstack((real_scaler.mean_, real_scaler.scale_))

    norm_matrix = np.hstack((norm_matrix_left, norm_matrix_right, norm_matrix_word, norm_matrix_scm))
    norm_matrix.tofile(inp_stats_file)

if norm_file_save:
    print('Start normalize and save each file')
    riemann_data = sio.loadmat(riemann_mean_path)
    riemann_mean = riemann_data['riemann_mean']
    norm_matrix = np.fromfile(inp_stats_file, dtype=np.float64)
    left_scaler_mean = norm_matrix[0]
    left_scaler_scale = norm_matrix[4]
    right_scaler_mean = norm_matrix[1]
    right_scaler_scale = norm_matrix[5]
    word_scaler_mean = norm_matrix[2]
    word_scaler_scale = norm_matrix[6]
    scm_scaler_mean = norm_matrix[3]
    scm_scaler_scale = norm_matrix[7]

    files = glob.glob(scm_path + '/' + 'train*.mat')
    for file in files:
        file_name, _ = os.path.splitext(os.path.basename(file))
        data = sio.loadmat(file)
        scm = data['scm']
        labels = data['labels']

        tangent_vectors = utils.tangent_space_mapping_both_2D_f32(scm, riemann_mean)

        norm_riemann_scm = np.concatenate((tangent_vectors, np.expand_dims(scm, axis=1)), axis=1)
        norm_riemann_scm[:, 0, :, :] = (norm_riemann_scm[:, 0, :, :] - left_scaler_mean) / left_scaler_scale
        norm_riemann_scm[:, 1, :, :] = (norm_riemann_scm[:, 1, :, :] - right_scaler_mean) / right_scaler_scale
        norm_riemann_scm[:, 2, :, :] = (norm_riemann_scm[:, 2, :, :] - word_scaler_mean) / word_scaler_scale
        norm_riemann_scm[:, 3, :, :] = (norm_riemann_scm[:, 3, :, :] - scm_scaler_mean) / scm_scaler_scale
        norm_riemann_scm = norm_riemann_scm.astype(np.float32)
        sio.savemat(riemann_scm_norm_path + '/' + file_name + '.mat', {'norm_riemann_scm': norm_riemann_scm, 'labels': labels})
