import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import scipy.io as sio
from scipy.signal import find_peaks
from scipy.fft import fft
import scipy
# import pyriemann
from scipy.linalg import logm, expm, sqrtm
# from pyriemann.utils.base import logm, expm, sqrtm
import scipy.constants as C
import os
import re
import torch
from scipy.signal import butter, lfilter, filtfilt
import matplotlib.pyplot as plt


def get_data(file_name, opt=None):
    data = sio.loadmat(file_name, squeeze_me=True)
    X = data['X']
    Y = data['Y']
    nfo = data['nfo'][()]

    num_channel = len(nfo['clab'])
    len_block = int(opt['time_block'] * nfo['fs'])

    # Extract segments corresponding to different labels
    Y_size = len(Y)
    diff_Y = np.diff(Y)
    segment = np.column_stack((np.append(0, np.where(diff_Y != 0)[0] + 1), np.append(np.where(diff_Y != 0)[0], Y_size - 1), Y[np.append(np.where(diff_Y != 0)[0], Y_size - 1)]))

    # Remove data corresponding to some labels if needed
    if opt and 'rm_label' in opt and np.intersect1d(Y, opt['rm_label']).size > 0:
        for rm_label in opt['rm_label']:
            segment = segment[segment[:, 2] != rm_label]

    num_segment = segment.shape[0]

    # Remove data at starting and ending parts of each segment if needed
    if opt and 'rm_transition' in opt and opt['rm_transition'] > 0:
        block = np.column_stack((opt['rm_transition'] * np.ones(num_segment), -opt['rm_transition'] * np.ones(num_segment), np.zeros(num_segment)))
        segment += block.astype(np.int64)
        inx = np.where(segment[:, 1] - segment[:, 0] <= 0)[0]
        segment = np.delete(segment, inx, axis=0)

    len_segment = segment[:, 1] - segment[:, 0] + 1

    # Extract eeg blocks
    len_overlap = opt['overlap'] if (opt and 'overlap' in opt and 0 < opt['overlap'] <= len_block) else 0

    X = X.T

    num_eeg_block = ((len_segment - len_block) // (len_block - len_overlap)) + 1
    num_eeg_block = num_eeg_block.astype(np.int64)
    eeg = np.zeros((num_channel, int(len_block), int(np.sum(num_eeg_block))))
    label = np.zeros(int(np.sum(num_eeg_block)))

    for i in range(num_segment):
        for j in range(num_eeg_block[i]):
            idx = segment[i, 0] + (j * (len_block - len_overlap)) + np.arange(len_block)
            eeg[:, :, np.sum(num_eeg_block[:i]) + j] = X[:, idx]
            label[np.sum(num_eeg_block[:i]) + j] = Y[idx[0]]

    return eeg, label


def get_data_filter(file_name, opt=None):
    lowcut = 8.0  # 低频截止频率（Hz）
    highcut = 30.0  # 高频截止频率（Hz）
    nyq = 0.5 * 512  # Nyquist频率为采样频率的一半
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(N=4, Wn=[low, high], btype='band')

    data = sio.loadmat(file_name, squeeze_me=True)
    X = data['X']
    Y = data['Y']
    nfo = data['nfo'][()]

    filtered_X = np.zeros_like(X)
    for i in range(X.shape[1]):  # Iterate over each channel
        channel_data = X[:, i]
        filtered_channel_data = filtfilt(b, a, channel_data)
        filtered_X[:, i] = filtered_channel_data
    X = filtered_X

    num_channel = len(nfo['clab'])
    len_block = int(opt['time_block'] * nfo['fs'])

    # Extract segments corresponding to different labels
    Y_size = len(Y)
    diff_Y = np.diff(Y)
    segment = np.column_stack((np.append(0, np.where(diff_Y != 0)[0] + 1), np.append(np.where(diff_Y != 0)[0], Y_size - 1), Y[np.append(np.where(diff_Y != 0)[0], Y_size - 1)]))

    # Remove data corresponding to some labels if needed
    if opt and 'rm_label' in opt and np.intersect1d(Y, opt['rm_label']).size > 0:
        for rm_label in opt['rm_label']:
            segment = segment[segment[:, 2] != rm_label]

    num_segment = segment.shape[0]

    # Remove data at starting and ending parts of each segment if needed
    if opt and 'rm_transition' in opt and opt['rm_transition'] > 0:
        block = np.column_stack((opt['rm_transition'] * np.ones(num_segment), -opt['rm_transition'] * np.ones(num_segment), np.zeros(num_segment)))
        segment += block.astype(np.int64)
        inx = np.where(segment[:, 1] - segment[:, 0] <= 0)[0]
        segment = np.delete(segment, inx, axis=0)

    len_segment = segment[:, 1] - segment[:, 0] + 1

    # Extract eeg blocks
    len_overlap = opt['overlap'] if (opt and 'overlap' in opt and 0 < opt['overlap'] <= len_block) else 0

    X = X.T

    num_eeg_block = ((len_segment - len_block) // (len_block - len_overlap)) + 1
    num_eeg_block = num_eeg_block.astype(np.int64)
    eeg = np.zeros((num_channel, int(len_block), int(np.sum(num_eeg_block))))
    label = np.zeros(int(np.sum(num_eeg_block)))

    for i in range(num_segment):
        for j in range(num_eeg_block[i]):
            idx = segment[i, 0] + (j * (len_block - len_overlap)) + np.arange(len_block)
            eeg[:, :, np.sum(num_eeg_block[:i]) + j] = X[:, idx]
            label[np.sum(num_eeg_block[:i]) + j] = Y[idx[0]]

    return eeg, label


def get_data_test(file_name, opt=None):
    file_base_name, _ = os.path.splitext(os.path.basename(file_name))
    data = sio.loadmat(file_name, squeeze_me=True)
    Y = np.loadtxt('data/test_label/' + file_base_name + '.txt').astype('uint8')
    X = data['X']
    nfo = data['nfo'][()]

    num_channel = len(nfo['clab'])
    len_block = int(opt['time_block'] * nfo['fs'])

    # Extract segments corresponding to different labels
    # Y_size = int(X.shape[0])
    Y = np.repeat(Y, len_block)
    Y_size = len(Y)
    diff_Y = np.diff(Y)
    segment = np.column_stack((np.append(0, np.where(diff_Y != 0)[0] + 1), np.append(np.where(diff_Y != 0)[0], Y_size - 1), Y[np.append(np.where(diff_Y != 0)[0], Y_size - 1)]))

    # Remove data corresponding to some labels if needed
    if opt and 'rm_label' in opt and np.intersect1d(Y, opt['rm_label']).size > 0:
        for rm_label in opt['rm_label']:
            segment = segment[segment[:, 2] != rm_label]

    num_segment = segment.shape[0]

    # Remove data at starting and ending parts of each segment if needed
    if opt and 'rm_transition' in opt and opt['rm_transition'] > 0:
        block = np.column_stack((opt['rm_transition'] * np.ones(num_segment), -opt['rm_transition'] * np.ones(num_segment), np.zeros(num_segment)))
        segment += block.astype(np.int64)
        inx = np.where(segment[:, 1] - segment[:, 0] <= 0)[0]
        segment = np.delete(segment, inx, axis=0)

    len_segment = segment[:, 1] - segment[:, 0] + 1

    # Extract eeg blocks
    len_overlap = opt['overlap'] if (opt and 'overlap' in opt and 0 < opt['overlap'] <= len_block) else 0

    X = X.T

    num_eeg_block = ((len_segment - len_block) // (len_block - len_overlap)) + 1
    num_eeg_block = num_eeg_block.astype(np.int64)
    eeg = np.zeros((num_channel, int(len_block), int(np.sum(num_eeg_block))))
    label = np.zeros(int(np.sum(num_eeg_block)))

    for i in range(num_segment):
        for j in range(num_eeg_block[i]):
            idx = segment[i, 0] + (j * (len_block - len_overlap)) + np.arange(len_block)
            eeg[:, :, np.sum(num_eeg_block[:i]) + j] = X[:, idx]
            label[np.sum(num_eeg_block[:i]) + j] = Y[idx[0]]

    return eeg, label


def get_data_test_filter(file_name, opt=None):
    lowcut = 8.0  # 低频截止频率（Hz）
    highcut = 30.0  # 高频截止频率（Hz）
    nyq = 0.5 * 512  # Nyquist频率为采样频率的一半
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(N=4, Wn=[low, high], btype='band')

    file_base_name, _ = os.path.splitext(os.path.basename(file_name))
    data = sio.loadmat(file_name, squeeze_me=True)
    Y = np.loadtxt('data/test_label/' + file_base_name + '.txt').astype('uint8')
    X = data['X']
    nfo = data['nfo'][()]

    filtered_X = np.zeros_like(X)
    for i in range(X.shape[1]):  # Iterate over each channel
        channel_data = X[:, i]
        filtered_channel_data = filtfilt(b, a, channel_data)
        filtered_X[:, i] = filtered_channel_data
    X = filtered_X

    num_channel = len(nfo['clab'])
    len_block = int(opt['time_block'] * nfo['fs'])

    # Extract segments corresponding to different labels
    # Y_size = int(X.shape[0])
    Y = np.repeat(Y, len_block)
    Y_size = len(Y)
    diff_Y = np.diff(Y)
    segment = np.column_stack((np.append(0, np.where(diff_Y != 0)[0] + 1), np.append(np.where(diff_Y != 0)[0], Y_size - 1), Y[np.append(np.where(diff_Y != 0)[0], Y_size - 1)]))

    # Remove data corresponding to some labels if needed
    if opt and 'rm_label' in opt and np.intersect1d(Y, opt['rm_label']).size > 0:
        for rm_label in opt['rm_label']:
            segment = segment[segment[:, 2] != rm_label]

    num_segment = segment.shape[0]

    # Remove data at starting and ending parts of each segment if needed
    if opt and 'rm_transition' in opt and opt['rm_transition'] > 0:
        block = np.column_stack((opt['rm_transition'] * np.ones(num_segment), -opt['rm_transition'] * np.ones(num_segment), np.zeros(num_segment)))
        segment += block.astype(np.int64)
        inx = np.where(segment[:, 1] - segment[:, 0] <= 0)[0]
        segment = np.delete(segment, inx, axis=0)

    len_segment = segment[:, 1] - segment[:, 0] + 1

    # Extract eeg blocks
    len_overlap = opt['overlap'] if (opt and 'overlap' in opt and 0 < opt['overlap'] <= len_block) else 0

    X = X.T

    num_eeg_block = ((len_segment - len_block) // (len_block - len_overlap)) + 1
    num_eeg_block = num_eeg_block.astype(np.int64)
    eeg = np.zeros((num_channel, int(len_block), int(np.sum(num_eeg_block))))
    label = np.zeros(int(np.sum(num_eeg_block)))

    for i in range(num_segment):
        for j in range(num_eeg_block[i]):
            idx = segment[i, 0] + (j * (len_block - len_overlap)) + np.arange(len_block)
            eeg[:, :, np.sum(num_eeg_block[:i]) + j] = X[:, idx]
            label[np.sum(num_eeg_block[:i]) + j] = Y[idx[0]]

    return eeg, label


def select_rows(matrix, idx):
    num_rows = matrix.shape[0]
    selected_rows = []
    for i in range(0, num_rows, idx):
        selected_rows.append(matrix[i])
        if i + 1 < num_rows:
            selected_rows.append(matrix[i + 1])
    result = np.vstack(selected_rows)
    return result


def my_sqrtm(x):
    eigenvalues, eigenvectors = np.linalg.eig(x)
    eigenvalues_positive = eigenvalues
    eigenvalues_positive[eigenvalues < 0] = 1e-6
    sqrt_eigenvalues = np.sqrt(eigenvalues_positive)
    A = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T
    return A


def my_logm(x):
    eigenvalues, eigenvectors = np.linalg.eig(x)
    eigenvalues_positive = eigenvalues
    eigenvalues_positive[eigenvalues < 0] = 1e-6
    log_eigenvalues = np.log(eigenvalues_positive)
    A = eigenvectors @ np.diag(log_eigenvalues) @ eigenvectors.T
    return A


def my_expm(x):
    eigenvalues, eigenvectors = np.linalg.eig(x)
    eigenvalues_positive = eigenvalues
    eigenvalues_positive[eigenvalues < 0] = 1e-6
    exp_eigenvalues = np.exp(eigenvalues_positive)
    A = eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.T
    return A


def matrix_half(matrix):
    A = my_sqrtm(matrix)
    return A


def matrix_minus_half(matrix):
    # A = my_sqrtm(matrix)
    # B = np.linalg.inv(A)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenvalues_positive = eigenvalues
    eigenvalues_positive[eigenvalues < 0] = 1e-6
    sqrt_eigenvalues = 1 / np.sqrt(eigenvalues_positive)
    B = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T
    return B


def my_inv(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenvalues_positive = eigenvalues
    eigenvalues_positive[eigenvalues < 0] = 1e-6
    inv_eigenvalues = 1 / eigenvalues_positive
    B = eigenvectors @ np.diag(inv_eigenvalues) @ eigenvectors.T
    return B


def compute_riemannian_mean(spd_matrices, type_metric, max_iter=15):
    num_spd, dims, _ = spd_matrices.shape   # (总数,32,32)
    M = np.mean(spd_matrices, axis=0)

    for ite_th in range(max_iter):
        if type_metric == 'A':
            A = my_sqrtm(M)  # A = C^(1/2)
            B = np.linalg.inv(A)  # B = C^(-1/2)

            S = np.zeros_like(M)
            for j_th in range(num_spd):
                C = spd_matrices[j_th, :, :]
                S += A @ my_logm(B @ C @ B) @ A
            S /= num_spd

            M = A @ my_expm(B @ S @ B) @ A

            eps = np.linalg.norm(S, 'fro')
            if eps < 1e-6:
                break

        elif type_metric == 'S':
            tmpX = np.zeros_like(M)
            for j_th in range(num_spd):
                tmpX += np.linalg.inv((spd_matrices[j_th, :, :] + M) / 2)
            tmpX /= num_spd
            M = np.linalg.inv(tmpX)

        elif type_metric == 'J':
            A = np.zeros_like(M, dtype=np.float32)
            B = np.sum(spd_matrices, axis=0)
            for i_th in range(num_spd):
                A += np.linalg.inv(spd_matrices[i_th, :, :])
            A_half = matrix_half(A)
            A_minus_half = matrix_minus_half(A)
            M = np.linalg.multi_dot([A_minus_half, np.real(matrix_half(A_half @ B @ A_half)), A_minus_half])
            # M = np.real(np.linalg.multi_dot([(A ** -0.5), (A ** 0.5 @ B @ A ** 0.5) ** 0.5, (A ** -0.5)]))

        elif type_metric == 'L':
            logm_matrices = np.zeros_like(spd_matrices, dtype=np.float32)
            for i_th in range(num_spd):
                logm_matrices[i_th, :, :] = my_logm(spd_matrices[i_th, :, :])
            mean_logDes = np.mean(logm_matrices, axis=0)
            M = np.real(my_expm(mean_logDes))

    return M.astype(np.float32)


def log_P(p, pi):
    A = my_sqrtm(p)  # A = p^(1/2)
    B = np.linalg.inv(A)  # B = p^(-1/2)
    return A @ my_logm(B @ pi @ B) @ A


def log_P_f32(p, pi):
    # eigenvalues, eigenvectors = np.linalg.eig(p)
    # sqrt_eigenvalues = np.sqrt(eigenvalues)
    # A = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T.conjugate()   # A = p^(1/2)
    # B = np.linalg.inv(A)    # B = p^(-1/2)

    # A = my_sqrtm(p)
    # B = np.linalg.inv(A)
    A = matrix_half(p)
    B = matrix_minus_half(p)

    temp_B = B @ pi @ B
    # temp_B = logm(temp_B, disp=False)
    temp_B = my_logm(temp_B)
    # temp_B = temp_B.astype(np.complex64)
    result = np.real(A) @ np.real(temp_B) @ np.real(A)   # real先后影响不大

    return result


def exp_P(p, si):
    A = my_sqrtm(p)  # A = p^(1/2)
    B = np.linalg.inv(A)  # B = p^(-1/2)
    return A @ my_expm(B @ si @ B) @ A


def extract_upper_triangular(matrix):
    upper_triangular = np.triu(matrix)
    rows, cols = upper_triangular.shape
    result_vector = upper_triangular[np.triu_indices(rows)]
    return result_vector.reshape(-1, 1)


def tangent_space_mapping_2D_f32(spd_matrices, M):
    num_spd, dims, _ = spd_matrices.shape
    dim_M, _ = M.shape
    s = np.zeros([num_spd, dim_M, dim_M], dtype=np.float32)

    # eigenvalues, eigenvectors = np.linalg.eig(M)
    # sqrt_eigenvalues = np.sqrt(eigenvalues)
    # inv_M = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T.conjugate()  # p^(1/2)
    # inv_M = np.linalg.inv(inv_M)  # p^(-1/2)

    # inv_M = my_sqrtm(np.real(M))
    # inv_M = np.linalg.inv(inv_M)  # inv_M = M^(-1/2)
    inv_M = matrix_minus_half(M)
    for j_th in range(num_spd):
        si = inv_M @ log_P_f32(M, spd_matrices[j_th, :, :]) @ inv_M
        si = si.astype(np.float32)
        s[j_th, :, :] = si
    return s


def tangent_space_mapping_adaptioin_2D_f32(spd_matrices, M, idx):
    num_spd, dims, _ = spd_matrices.shape
    num_M, dim_M, _ = M.shape
    s = np.zeros([num_spd, dim_M, dim_M], dtype=np.float32)

    # eigenvalues, eigenvectors = np.linalg.eig(M)
    # sqrt_eigenvalues = np.sqrt(eigenvalues)
    # inv_M = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T.conjugate()  # p^(1/2)
    # inv_M = np.linalg.inv(inv_M)  # p^(-1/2)
    inv_M = np.zeros((num_M, dim_M, dim_M), dtype=np.float32)
    for m in range(num_M):
        # inv_M[m] = my_sqrtm(np.real(M[m]))
        # inv_M[m] = np.linalg.inv(inv_M[m])  # inv_M = M^(-1/2)
        inv_M[m] = matrix_minus_half(M[m])
    for j_th in range(num_spd):
        idx_class = idx[j_th]
        si = inv_M[idx_class] @ log_P_f32(M[idx_class], spd_matrices[j_th, :, :]) @ inv_M[idx_class]
        si = si.astype(np.float32)
        s[j_th, :, :] = si
    return s


def tangent_space_mapping_both_2D_f32(spd_matrices, M):
    num_spd, dims, _ = spd_matrices.shape
    num_M, dim_M, _ = M.shape
    s = np.zeros([num_spd, num_M, dim_M, dim_M], dtype=np.float32)

    # eigenvalues, eigenvectors = np.linalg.eig(M)
    # sqrt_eigenvalues = np.sqrt(eigenvalues)
    # inv_M = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T.conjugate()  # p^(1/2)
    # inv_M = np.linalg.inv(inv_M)  # p^(-1/2)
    inv_M = np.zeros((num_M, dim_M, dim_M), dtype=np.float32)
    for m in range(num_M):
        # inv_M[m] = my_sqrtm(np.real(M[m]))
        # inv_M[m] = np.linalg.inv(inv_M[m])  # inv_M = M^(-1/2)
        inv_M[m] = matrix_minus_half(M[m])  # inv_M = M^(-1/2)
    for j_th in range(num_spd):
        for m in range(num_M):
            si = inv_M[m] @ log_P_f32(M[m], spd_matrices[j_th, :, :]) @ inv_M[m]
            si = si.astype(np.float32)
            s[j_th, m, :, :] = si
    return s


def riemann_mean_by_classes(scm, labels):
    hand_idx = np.where(labels == 2)
    scm_hand = scm[hand_idx[1]]
    hand_riemann_mean = compute_riemannian_mean(scm_hand, type_metric='L', max_iter=15)

    unhand_idx = np.where(labels == 3)
    scm_unhand = scm[unhand_idx[1]]
    unhand_riemann_mean = compute_riemannian_mean(scm_unhand, type_metric='L', max_iter=15)

    all_standard_riemann_mean = np.stack((hand_riemann_mean, unhand_riemann_mean), axis=0)

    return all_standard_riemann_mean


def riemann_mean_by_all_classes(scm, labels):
    hand_idx = np.where(labels == 2)
    scm_hand = scm[hand_idx[1]]
    hand_riemann_mean = compute_riemannian_mean(scm_hand, type_metric='L', max_iter=15)

    unhand_idx = np.where(labels == 3)
    scm_unhand = scm[unhand_idx[1]]
    unhand_riemann_mean = compute_riemannian_mean(scm_unhand, type_metric='L', max_iter=15)

    word_idx = np.where(labels == 7)
    scm_word = scm[word_idx[1]]
    word_riemann_mean = compute_riemannian_mean(scm_word, type_metric='L', max_iter=15)

    all_standard_riemann_mean = np.stack((hand_riemann_mean, unhand_riemann_mean, word_riemann_mean), axis=0)

    return all_standard_riemann_mean
