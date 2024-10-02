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


class MotorImageryDataset:
    def __init__(self, dataset='A01T.npz'):
        if not dataset.endswith('.npz'):
            dataset += '.npz'
        if 'T' in dataset:
            self.dataset_train = True
        else:
            self.dataset_train = False

        self.data = np.load(dataset)

        self.Fs = 250   # 250Hz from original paper

        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']

        self.raw = self.data['s'].T
        self.events_type = self.data['etyp'].T
        self.events_position = self.data['epos'].T
        self.events_duration = self.data['edur'].T
        self.artifacts = self.data['artifacts'].T

        if not self.dataset_train:
            self.file_name, _ = os.path.splitext(os.path.basename(dataset))
            self.mat_data = sio.loadmat('data/true_labels/' + self.file_name)
            self.mat_classes = self.mat_data['classlabel'].flatten().tolist()

        # Types of motor imagery
        self.mi_types = {769: 'left', 770: 'right', 771: 'foot', 772: 'tongue', 783: 'unknown'}

        self.mapping = {'left': 1, 'right': 2, 'foot': 3, 'tongue': 4}

    def get_trials_from_channel(self):
        startrial_code = 768
        starttrial_events = self.events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]

        trials = []
        classes = []
        idx_idx = -1
        for index in idxs:
            try:
                idx_idx += 1
                type_e = self.events_type[0, index+1]
                class_e = self.mi_types[type_e]
                if self.dataset_train:
                    classes.append(self.mapping[class_e])
                else:
                    classes.append(self.mat_classes[idx_idx])

                start = self.events_position[0, index]
                stop = start + self.events_duration[0, index]
                trial = self.raw[:22, start:stop]
                trials.append(trial)
            except:
                continue
        if idx_idx != len(idxs)-1:
            raise ValueError

        return trials, classes

# datasetA1 = MotorImageryDataset()
# trials, classes = datasetA1.get_trials_from_channel()
# # trials contains the N valid trials, and clases its related class.


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
            B = matrix_minus_half(M)  # B = C^(-1/2)
            # B = np.linalg.inv(A)

            S = np.zeros_like(M)
            for j_th in range(num_spd):
                C = spd_matrices[j_th, :, :]
                S += A @ my_logm(B @ C @ B) @ A
            S /= num_spd

            M = A @ my_expm(B @ S @ B) @ A
            # M = my_expm(B @ S @ B)

            eps = np.linalg.norm(S, 'fro')
            # print(eps)
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
    hand_idx = np.where(labels == 0)
    scm_hand = scm[hand_idx[1]]
    left_riemann_mean = compute_riemannian_mean(scm_hand, type_metric='L', max_iter=15)

    unhand_idx = np.where(labels == 1)
    scm_unhand = scm[unhand_idx[1]]
    right_riemann_mean = compute_riemannian_mean(scm_unhand, type_metric='L', max_iter=15)

    feet_idx = np.where(labels == 2)
    scm_feet = scm[feet_idx[1]]
    feet_riemann_mean = compute_riemannian_mean(scm_feet, type_metric='L', max_iter=15)

    tongue_idx = np.where(labels == 3)
    scm_tongue = scm[tongue_idx[1]]
    tongue_riemann_mean = compute_riemannian_mean(scm_tongue, type_metric='L', max_iter=15)

    all_standard_riemann_mean = np.stack((left_riemann_mean, right_riemann_mean, feet_riemann_mean, tongue_riemann_mean), axis=0)

    return all_standard_riemann_mean
