'''
#####################################################################################################################
Discription: Train and test RGN
#######################################################################################################################
'''
import glob

import scipy.io as scio
from sklearn.utils import shuffle
import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import scipy.io as sio
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.model_selection import KFold
import seaborn as sns
import random
from sklearn.model_selection import KFold, cross_val_score
from scipy.signal import butter, lfilter, filtfilt
from utils import MotorImageryDataset
import spd_utils.geoopt as geoopt
from spd_utils.modules import *
# from torchvision.transforms import ToTensor
from sklearn.metrics import precision_score, recall_score
from moabb.utils import depreciated_alias
warnings.filterwarnings("ignore")
# device = torch.device('cuda:0')
device = torch.device('cuda:0')
# device_gpu = torch.device('cuda:0')

# train_model = True
# test_model = True
train_model = False
test_model = True

# train_seed = 4
# torch.manual_seed(train_seed)
# torch.cuda.manual_seed(train_seed)
# torch.cuda.manual_seed_all(train_seed)  # 如果使用多个GPU
# random.seed(train_seed)
# np.random.seed(train_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

subject_id = 'A01'
seed = 143

inp_dim = 22
num_classes = 4

# train_file_number_rate = 0.7
# valid_file_number_rate = 0.2
# test_file_number_rate = 0.1
num_fold = 10

raw_path = 'data/bcidatasetIV2a-master'
# scm_path = 'data/scm'
# scm_norm_path = 'data/scm_norm'
# scm_norm_path = 'data/scm_filter_norm_subject'
scm_norm_path = 'data/scm_filter_lap_norm_subject'
inp_stats_file = f"data/subject/input_norm_states_scm_filter_{subject_id}.norm"
riemann_mean_norm_path = f'data/subject/all_riemann_mean_filter_norm_{subject_id}.mat'

# model_name = f'weight/subject/model_classification_all_riemann_middle_filter_{subject_id}.pth'

batch_size = 256
num_epochs = 50
learning_rate = 0.001
L_decay = 0.008
patience = 15

branch_loss_rate = 1e-3


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

class ClassificationNetHand(nn.Module):  # v1
    def __init__(self, inp_dim, num_classes):
        super(ClassificationNetHand, self).__init__()
        self.inp_dim = inp_dim

        # def create_branch():
        #     layers_1 = nn.Sequential(
        #         nn.Conv2d(1, 16, kernel_size=3, stride=(1,2), padding=1),
        #         nn.BatchNorm2d(16),
        #         nn.ReLU(),
        #         nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        #         nn.BatchNorm2d(16),
        #         # nn.ReLU(),
        #     )
        #     layers_1_res = nn.Sequential(
        #         nn.Conv2d(1, 16, kernel_size=1, stride=(1, 2), padding=0),
        #     )
        #
        #     layers_2 = nn.Sequential(
        #         nn.Conv2d(16, 32, kernel_size=3, stride=(1, 2), padding=1),
        #         nn.BatchNorm2d(32),
        #         nn.ReLU(),
        #         nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #         nn.BatchNorm2d(32),
        #     )
        #     layers_2_res = nn.Sequential(
        #         nn.Conv2d(16, 32, kernel_size=1, stride=(1, 2), padding=0),
        #     )
        #
        #     layers_3 = nn.Sequential(
        #         nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        #     )
        #     output_layers = nn.Sequential(
        #         BiMap(1, inp_dim, inp_dim),
        #         # RescaledSymmetricGaussianActivation(),
        #     )
        #
        #     class Branch(nn.Module):
        #         def __init__(self):
        #             super(Branch, self).__init__()
        #             self.layers_1 = layers_1
        #             self.layers_1_res = layers_1_res
        #             self.layers_2 = layers_2
        #             self.layers_2_res = layers_2_res
        #             self.layers_3 = layers_3
        #             # self.output_layers = output_layers
        #
        #         def forward(self, x):
        #             # residual_1 = x
        #             x_1_main = self.layers_1(x)
        #             x_1 = x_1_main + self.layers_1_res(x)
        #             x_1 = nn.ReLU()(x_1)
        #
        #             x_2_main = self.layers_2(x_1)
        #             x_2 = x_2_main + self.layers_2_res(x_1)
        #             x_2 = nn.ReLU()(x_2)
        #
        #             x_3 = self.layers_3(x_2)
        #             branch_output = (x_3 @ x_3.transpose(-1, -2)).to(torch.float32)
        #             # branch_output = branch_output + x
        #             return branch_output
        #
        #     return Branch()
        def create_branch():
            layers_1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1, 10), stride=(1, 2), padding=(0, 4)),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(16),
                # nn.ReLU(),
            )
            layers_1_res = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=1, stride=(1, 2), padding=(0, 0)),
            )

            layers_2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(32),
            )
            layers_2_res = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=1, stride=(1, 2), padding=(0, 0)),
            )

            layers_3 = nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            )
            output_layers = nn.Sequential(
                BiMap(1, inp_dim, inp_dim),
                # RescaledSymmetricGaussianActivation(),
            )

            class Branch(nn.Module):
                def __init__(self):
                    super(Branch, self).__init__()
                    self.layers_1 = layers_1
                    self.layers_1_res = layers_1_res
                    self.layers_2 = layers_2
                    self.layers_2_res = layers_2_res
                    self.layers_3 = layers_3
                    self.output_layers = output_layers

                def forward(self, x):
                    # residual_1 = x
                    x_1_main = self.layers_1(x)
                    x_1 = x_1_main + self.layers_1_res(x)
                    x_1 = nn.ReLU()(x_1)

                    x_2_main = self.layers_2(x_1)
                    x_2 = x_2_main + self.layers_2_res(x_1)
                    x_2 = nn.ReLU()(x_2)

                    x_3 = self.layers_3(x_2)
                    branch_output = (x_3 @ x_3.transpose(-1, -2)).to(torch.double)
                    branch_output = self.output_layers(branch_output).to(torch.float32)
                    return branch_output

            return Branch()

        self.branch_left = create_branch()
        self.branch_right = create_branch()
        self.branch_feet = create_branch()
        self.branch_tongue = create_branch()

        # self.gaussian_activation = RescaledSymmetricGaussianActivation()
        # self.multihead_attention = nn.MultiheadAttention(embed_dim=self.inp_dim*(self.inp_dim+1)//2, num_heads=1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=self.inp_dim * (self.inp_dim + 1) // 2, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder, num_layers=1)

        self.fc_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # scm & riemann mean
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_linear_layers = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            # nn.Linear(8 * 19 * 19, 64),
            nn.ReLU(),
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(num_classes * self.inp_dim * (self.inp_dim + 1) // 2, 128),
            # nn.Linear(8 * 19 * 19, 64),
            nn.ReLU(),
        )
        self.main_layer = nn.Sequential(
            nn.Linear(256, num_classes),
        )



    def vectorize_spd_matrix(self, matrix):
        # Only upper triangular part without diagonal
        indices = torch.triu_indices(matrix.shape[-1], matrix.shape[-1], offset=1)
        upper_triangular = matrix[..., indices[0], indices[1]]

        # Diagonal elements multiplied by sqrt(2)
        diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
        # diagonal = torch.sqrt(torch.tensor(2.0)) * diagonal
        upper_triangular = torch.sqrt(torch.tensor(2.0)) * upper_triangular

        # Concatenate upper triangular and modified diagonal
        vectorized = torch.cat((upper_triangular, diagonal), dim=-1)

        return vectorized

    def forward(self, x):
        batch_size, _, _ = x.size()
        x = x.unsqueeze(1)  # (batch_size, 22, 22) -> (batch_size, 1, 22, 22)

        x_left = self.branch_left(x)
        x_right = self.branch_right(x)
        x_feet = self.branch_feet(x)
        x_tongue = self.branch_tongue(x)

        x_left_vec = self.vectorize_spd_matrix(x_left)
        x_right_vec = self.vectorize_spd_matrix(x_right)
        x_feet_vec = self.vectorize_spd_matrix(x_feet)
        x_tongue_vec = self.vectorize_spd_matrix(x_tongue)
        # x_vec = self.vectorize_spd_matrix(x)

        # x_combined = torch.cat((x, x_left, x_right, x_feet, x_tongue), dim=1)

        x_combined = torch.cat((x_left_vec, x_right_vec, x_feet_vec, x_tongue_vec), dim=1)
        # x_combined = x_combined.to(torch.float32)

        x_combined = torch.transpose(x_combined, 0, 1)
        # x_combined_attention, _ = self.multihead_attention(x_combined, x_combined, x_combined)
        x_combined_attention = self.transformer_encoder(x_combined)
        x_combined_attention = torch.transpose(x_combined_attention, 0, 1)

        # x_combined_attention = torch.cat((x_vec.to(torch.float32), x_combined_attention), dim=1)
        x_combined_attention = x_combined_attention.reshape(batch_size, -1)

        # x_combined_attention, _ = self.multihead_attention(x_combined, x_combined, x_combined)

        # x_combined_with_input = torch.cat((x, x_combined_attention), dim=1)

        linear_fea = self.linear_layer(x_combined_attention)
        x_fea = self.fc_layers(x @ x.transpose(-1, -2))
        x_fea = x_fea.reshape(batch_size, -1)
        x_fea = self.fc_linear_layers(x_fea)
        combined_fea = torch.cat((linear_fea, x_fea), dim=1)
        logits = self.main_layer(combined_fea)
        # conv_output = conv_output.view(batch_size, -1)  # (batch_size, 8 * 16 * 16)
        # logits = self.fc_layers(conv_output)   # (256,128)
        return logits, x_left, x_right, x_feet, x_tongue


def load_onehot_label(label):
    eye_matrix = np.eye(num_classes)

    transformed_list = [eye_matrix[element - 1] for element in np.squeeze(label)]
    return np.array(transformed_list)


# 定义优化器
class Cosine(torch.optim.lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, lr, step_each_epoch, epochs, **kwargs):
        T_max = step_each_epoch * epochs
        super(Cosine, self).__init__(optimizer, T_max=T_max, eta_min=lr)
        self.update_specified = False


class CosineWarmup(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, lr, step_each_epoch, epochs, warmup_epoch=5, **kwargs):
        assert epochs > warmup_epoch, f"total epoch({epochs}) should be larger than warmup_epoch({warmup_epoch}) in CosineWarmup."
        warmup_step = warmup_epoch * step_each_epoch
        start_lr = 0.0
        end_lr = lr
        lr_sch = Cosine(None, lr, step_each_epoch, epochs - warmup_epoch)

        super(CosineWarmup, self).__init__(optimizer, T_0=warmup_step, T_mult=1, eta_min=end_lr)
        self.update_specified = False


class MatrixNormLoss(nn.Module):
    def __init__(self):
        super(MatrixNormLoss, self).__init__()

    def forward(self, x1, x2):
        diff = x1 - x2
        norm = torch.norm(diff, p=2, dim=[-2, -1]).squeeze()  # 计算二范数
        return norm

riemann_data = sio.loadmat(riemann_mean_norm_path)
riemann_mean = riemann_data['riemann_mean']
riemann_mean = torch.tensor(riemann_mean, dtype=torch.float32, device=device)

norm_matrix = np.fromfile(inp_stats_file, dtype=np.float32)
real_scaler_mean = torch.tensor(norm_matrix[0], dtype=torch.float32, device=device)
real_scaler_scale = torch.tensor(norm_matrix[1], dtype=torch.float32, device=device)
# load data
all_scm = []
all_labels = []
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
    filtered_eeg_data = np.zeros_like(eeg_data)
    for i in range(eeg_data.shape[0]):  # Iterate over each channel
        for j in range(eeg_data.shape[2]):  # Iterate over each epoc
            channel_data = eeg_data[i, :, j]
            filtered_channel_data = filtfilt(b, a, channel_data)
            filtered_eeg_data[i, :, j] = filtered_channel_data
    eeg_data = filtered_eeg_data

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
    eeg_data = eeg_data_lap.transpose((2, 0, 1))
    labels = np.tile(labels, num_lap)
    all_scm.append(eeg_data)
    all_labels.append(labels)

all_scm = np.concatenate(all_scm, axis=0)   # (总数,22,250)
all_labels = np.concatenate(all_labels, axis=0)     # (1,总数)

all_labels_onehot = load_onehot_label(all_labels)

all_number = all_labels.shape[-1]

# train_number = int(train_file_number_rate * all_number)
# valid_number = int(valid_file_number_rate * all_number)
# test_number = all_number - train_number - valid_number

all_scm = all_scm.astype(np.float32)
all_labels_onehot = all_labels_onehot.astype(np.float32)


np.random.seed(seed)
np.random.shuffle(all_scm)
np.random.seed(seed)
np.random.shuffle(all_labels)
np.random.seed(seed)
np.random.shuffle(all_labels_onehot)

all_scm = torch.tensor(all_scm)
all_labels = torch.tensor(all_labels)
all_labels_onehot = torch.tensor(all_labels_onehot)

# train_data = all_scm[:train_number]
# train_dist = all_labels[:train_number]
# train_targets = all_labels_onehot[:train_number]
# val_data = all_scm[train_number:train_number + valid_number]
# val_dist = all_labels[train_number:train_number + valid_number]
# val_targets = all_labels_onehot[train_number:train_number + valid_number]
# test_data = all_scm[train_number + valid_number:]
# test_dist = all_labels[train_number + valid_number:]
# test_targets = all_labels_onehot[train_number + valid_number:]

all_acc = []
all_target = []
all_pred = []
kf = KFold(n_splits=num_fold, shuffle=False)
for fold, (train_index, test_index) in enumerate(kf.split(all_scm, all_labels_onehot)):
    train_data, val_data = all_scm[train_index], all_scm[test_index]
    train_targets, val_targets = all_labels_onehot[train_index], all_labels_onehot[test_index]
    model_name = f'weight/subject_cv_5/model_classification_all_riemann_middle_filter_{subject_id}_{fold}.pth'

    train_data = train_data.to(device)
    train_targets = train_targets.to(device)
    val_data = val_data.to(device)
    val_targets = val_targets.to(device)
    # test_data = test_data.to(device)
    # test_targets = test_targets.to(device)


    model = ClassificationNetHand(inp_dim, num_classes)  # 初始化模型
    model = model.to(device)
    # model.load_state_dict(torch.load(f'weight/subject_cv_2/model_classification_all_riemann_middle_filter_{subject_id}_{fold} - 副本.pth'))  # todo
    # optimizer = Adam(model.parameters(), lr=0.00125)  # 初始化优化器

    criterion_main = nn.BCEWithLogitsLoss()
    # criterion_main = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    criterion_main = criterion_main.to(device)
    criterion_branch = MatrixNormLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L_decay)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=0)  # 学习率调度器

    # accuracy = Accuracy()
    # accuracy = accuracy.to(device)


    if train_model:
        best_val_loss = float('inf')  # 初始最佳验证集损失为正无穷大
        best_val_acc = float(0)  # 初始最佳验证集损失为正无穷大
        no_improvement_count = 0  # 记录连续没有改善的 epoch 数
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i in range(0, len(train_data), batch_size):
                inputs = train_data[i:i + batch_size]
                labels = train_targets[i:i + batch_size]

                optimizer.zero_grad()

                outputs, output_left, output_right, output_feet, output_tongue = model(inputs)

                condition_indices = torch.argmax(labels, dim=1)
                loss_branch1 = criterion_branch(output_left, riemann_mean[0])
                loss_branch2 = criterion_branch(output_right, riemann_mean[1])
                loss_branch3 = criterion_branch(output_feet, riemann_mean[2])
                loss_branch4 = criterion_branch(output_tongue, riemann_mean[3])
                loss_branch1 = loss_branch1 * torch.where(condition_indices == 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
                loss_branch2 = loss_branch2 * torch.where(condition_indices == 1, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
                loss_branch3 = loss_branch3 * torch.where(condition_indices == 2, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
                loss_branch4 = loss_branch4 * torch.where(condition_indices == 3, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

                loss_main = criterion_main(outputs, labels)
                loss = torch.mean(branch_loss_rate * (loss_branch1 + loss_branch2 + loss_branch3 + loss_branch4) + loss_main)
                loss.backward()

                optimizer.step()
                scheduler.step()
                running_loss += loss.item()

            # 在每个 epoch 结束时评估模型
            model.eval()
            val_running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            with torch.no_grad():
                for i in range(0, len(val_data), batch_size):
                    inputs = val_data[i:i + batch_size]
                    labels = val_targets[i:i + batch_size]
                    val_outputs, val_output_left, val_output_right, val_output_feet, val_output_tongue = model(inputs)

                    condition_indices = torch.argmax(labels, dim=1)
                    val_loss_branch1 = criterion_branch(val_output_left, riemann_mean[0])
                    val_loss_branch2 = criterion_branch(val_output_right, riemann_mean[1])
                    val_loss_branch3 = criterion_branch(val_output_feet, riemann_mean[2])
                    val_loss_branch4 = criterion_branch(val_output_tongue, riemann_mean[3])
                    val_loss_branch1 = val_loss_branch1 * torch.where(condition_indices == 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
                    val_loss_branch2 = val_loss_branch2 * torch.where(condition_indices == 1, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
                    val_loss_branch3 = val_loss_branch3 * torch.where(condition_indices == 2, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
                    val_loss_branch4 = val_loss_branch4 * torch.where(condition_indices == 3, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

                    val_loss_main = criterion_main(val_outputs, labels)
                    val_loss = torch.mean(branch_loss_rate * (val_loss_branch1 + val_loss_branch2 + val_loss_branch3 + val_loss_branch4) + val_loss_main)
                    val_running_loss += val_loss.item()

                    val_outputs_int = torch.argmax(val_outputs, axis=1)
                    val_labels_int = torch.argmax(labels, axis=1)
                    # val_acc = accuracy(val_outputs_int, val_labels_int)

                    correct_predictions += torch.sum(val_outputs_int == val_labels_int).item()
                    total_samples += labels.size(0)

                val_acc = correct_predictions / total_samples
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_data)}, Val Loss: {val_running_loss/len(val_data)}, Val Accuracy: {val_acc}")

                # 判断是否有改善
                if val_running_loss < best_val_loss:
                    best_val_loss = val_running_loss
                    torch.save(model.state_dict(), model_name)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # 检查是否达到 early stopping 条件
                if no_improvement_count >= patience:
                    print("Early stopping at epoch %d. No improvement in validation loss." % (epoch - patience + 1))
                    break

    if test_model:
        model = ClassificationNetHand(inp_dim, num_classes)
        model = model.to(device)
        model.load_state_dict(torch.load(model_name))
        model.eval()
        print("Loaded model from disk")

        predictions = []
        predictions_class = []
        test_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch_inputs = val_data[i:i + batch_size]
                batch_targets = val_targets[i:i + batch_size]

                outputs, output_left, output_right, output_feet, output_tongue = model(batch_inputs)

                condition_indices = torch.argmax(batch_targets, dim=1)
                loss_branch1 = criterion_branch(output_left, riemann_mean[0])
                loss_branch2 = criterion_branch(output_right, riemann_mean[1])
                loss_branch3 = criterion_branch(output_feet, riemann_mean[2])
                loss_branch4 = criterion_branch(output_tongue, riemann_mean[3])
                loss_branch1 = loss_branch1 * torch.where(condition_indices == 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
                loss_branch2 = loss_branch2 * torch.where(condition_indices == 1, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
                loss_branch3 = loss_branch3 * torch.where(condition_indices == 2, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
                loss_branch4 = loss_branch4 * torch.where(condition_indices == 3, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

                loss_main = criterion_main(outputs, batch_targets)
                loss = torch.mean(branch_loss_rate * (loss_branch1 + loss_branch2 + loss_branch3 + loss_branch4) + loss_main)
                predictions.append(outputs)

                class_indices = torch.argmax(outputs, dim=1)
                predictions_class.append(class_indices)

                test_loss += loss.item()
        # 将预测结果转换为张量
        predictions = torch.cat(predictions, dim=0).cpu()
        predictions_class = torch.cat(predictions_class, dim=0).cpu()
        avg_test_loss = test_loss / len(val_data)

        dist_pred = predictions_class.numpy()
        dist_real = np.argmax(val_targets.cpu().numpy(), axis=1)

        dist_pred = dist_pred.astype(np.int32)
        dist_real = dist_real.astype(np.int32)

        correct_predictions = np.sum(dist_pred == dist_real).item()
        print(f'Test accuracy: {correct_predictions/len(dist_real)}')
        all_target.append(dist_real)
        all_pred.append(dist_pred)
        all_acc.append(correct_predictions / len(dist_real))
if test_model:
    print(f'Mean acc: {np.mean(all_acc)}')
    all_dist_real = np.concatenate(all_target)
    all_dist_pred = np.concatenate(all_pred)
    recall = recall_score(all_dist_real, all_dist_pred, average='macro')
    precision = precision_score(all_dist_real, all_dist_pred, average='macro')
    print(f'Mean Precision: {precision}')
    print(f'Mean Recall: {recall}')
    kappa_value = cohen_kappa_score(all_dist_real, all_dist_pred)
    print("kappa value is %f" % kappa_value)
    # confuse matrix
    C1 = confusion_matrix(all_dist_real, all_dist_pred)  # True_label 真实标签 shape=(n,1);T_predict1 预测标签 shape=(n,1)
    row_sums = C1.sum(axis=1, keepdims=True)
    normalized_conf_matrix = C1 / row_sums.astype(float)
    xtick = ['Left hand', 'Right hand', 'both feet', 'tongue']
    ytick = ['Left hand', 'Right hand', 'both feet', 'tongue']
    h = sns.heatmap(normalized_conf_matrix, fmt='.3f', cmap='Blues', annot=True, cbar=False, xticklabels=xtick,
                    yticklabels=ytick, annot_kws={"size": 12})  # 画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒
    h.tick_params(axis='both', which='major', labelsize=12)
    cb = h.figure.colorbar(h.collections[0])  # 显示colorbar
    cb.ax.tick_params(labelsize=12)
    plt.ylabel('True label', fontsize=12, weight='bold', fontname='Times New Roman')
    plt.xlabel('Predicted label', fontsize=12, weight='bold', fontname='Times New Roman')
    h.set_xticklabels(h.get_xticklabels(), fontname='Times New Roman')
    h.set_yticklabels(h.get_yticklabels(), fontname='Times New Roman')
    # plt.title('Proposed', fontsize=14)
    plt.tight_layout()
    plt.show()

    # kappa_value = cohen_kappa_score(dist_real, dist_pred)
    # print("kappa value is %f" % kappa_value)