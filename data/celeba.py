import os
from torchvision import transforms
import pandas as pd
import os
import copy
import math
import json
import random as rnd
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as  pd
import torchvision.utils as vision_utils
from PIL import Image
import torchvision
from scipy.linalg import qr
import torchvision.transforms as transforms
import pickle
from matplotlib.ticker import NullFormatter


class CELEBA(Dataset):
    def __init__(self, x, y, p, isTrain, target_resolution=(224, 224)):
        self.x = x
        self.y_array = y
        self.p_array = p

        self.isTrain = isTrain
        if isTrain:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(target_resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(target_resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        p = self.p_array[idx]
        img = self.x[idx]
        img = self.transform(Image.open(img).convert("RGB"))

        return img, y, p, idx

    def __getimage__(self, idx):
        img = self.x[idx]
        img = Image.open(img).convert("RGB")
        return img


# This celeba considers the inverse problem, where the target feature is the gender and the spurious feature
# is the hair color. This setup makes the spurious feature easier to learn than the target feature
# y==0 -> female, y==1 -> male, a==0 -> blonde, a==1 -> non-blonde
def get_celeba(spurious_strength, data_dir, seed):
    save_dir = os.path.join(data_dir, f"celeba_{spurious_strength}-{seed}.pkl")
    if os.path.exists(save_dir):
        print("Loading Dataset")
        with open(save_dir, 'rb') as f:
            X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test = pickle.load(f)
        return X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test
    print("Generating Dataset")
    metadata = os.path.join(data_dir, "metadata_celeba.csv")
    df = pd.read_csv(metadata)

    # Train dataset
    train_df = df[df["split"] == ({"tr": 0, "va": 1, "te": 2}["tr"])]
    train_df_x = train_df["filename"].astype(str).map(lambda x: os.path.join(data_dir, x)).tolist()
    # y==0 is non-blonde and y==1 is blonde
    train_df_y = np.array(train_df["a"].tolist())
    train_df_a = 1 - np.array(train_df["y"].tolist())
    y_0_a_0_idx = np.where((train_df_y == 0) & (train_df_a == 0))[0]
    y_0_a_1_idx = np.where((train_df_y == 0) & (train_df_a == 1))[0]
    y_1_a_0_idx = np.where((train_df_y == 1) & (train_df_a == 0))[0]
    y_1_a_1_idx = np.where((train_df_y == 1) & (train_df_a == 1))[0]
    np.random.shuffle(y_0_a_0_idx)
    np.random.shuffle(y_0_a_1_idx)
    np.random.shuffle(y_1_a_0_idx)
    np.random.shuffle(y_1_a_1_idx)

    if spurious_strength > 22880 / (22880+1387):
        majority_count = 22880
        minority_count = int((1 - spurious_strength) / spurious_strength * majority_count)
    else:
        minority_count = 1387
        majority_count = int(spurious_strength / (1 - spurious_strength) * minority_count)

    if minority_count != 0:
        all_idxs = np.concatenate((y_0_a_0_idx[:majority_count], y_1_a_1_idx[:majority_count], y_0_a_1_idx[:minority_count], y_1_a_0_idx[:minority_count]), axis=0)
    else:
        all_idxs = np.concatenate((y_0_a_0_idx[:majority_count], y_1_a_1_idx[:majority_count]), axis=0)

    X_train, Y_train, P_train = [train_df_x[i] for i in all_idxs], np.array(train_df_y[all_idxs]), np.array(train_df_a[all_idxs])

    val_df = df[df["split"] == ({"tr": 0, "va": 1, "te": 2}["va"])]
    val_df_x = val_df["filename"].astype(str).map(lambda x: os.path.join(data_dir, x)).tolist()
    val_df_y = np.array(val_df["a"].tolist())
    val_df_a = 1 - np.array(val_df["y"].tolist())

    y_0_a_0_idx = np.where((val_df_y == 0) & (val_df_a == 0))[0]
    y_0_a_1_idx = np.where((val_df_y == 0) & (val_df_a == 1))[0]
    y_1_a_0_idx = np.where((val_df_y == 1) & (val_df_a == 0))[0]
    y_1_a_1_idx = np.where((val_df_y == 1) & (val_df_a == 1))[0]
    all_count = min(len(y_0_a_0_idx), len(y_0_a_1_idx), len(y_1_a_0_idx), len(y_1_a_1_idx))
    y_0_a_0_idx = np.random.choice(y_0_a_0_idx, size=all_count, replace=False)
    y_1_a_1_idx = np.random.choice(y_1_a_1_idx, size=all_count, replace=False)
    y_0_a_1_idx = np.random.choice(y_0_a_1_idx, size=all_count, replace=False)
    y_1_a_0_idx = np.random.choice(y_1_a_0_idx, size=all_count, replace=False)
    all_idxs = np.concatenate((y_0_a_0_idx, y_1_a_1_idx, y_0_a_1_idx, y_1_a_0_idx), axis=0)
    X_val, Y_val, P_val = [val_df_x[i] for i in all_idxs], np.array(val_df_y[all_idxs]), np.array(val_df_a[all_idxs])

    
    test_df = df[df["split"] == ({"tr": 0, "va": 1, "te": 2}["te"])]
    test_df_x = test_df["filename"].astype(str).map(lambda x: os.path.join(data_dir, x)).tolist()
    test_df_y = np.array(test_df["a"].tolist())
    test_df_a = 1 - np.array(test_df["y"].tolist())

    y_0_a_0_idx = np.where((test_df_y == 0) & (test_df_a == 0))[0]
    y_0_a_1_idx = np.where((test_df_y == 0) & (test_df_a == 1))[0]
    y_1_a_0_idx = np.where((test_df_y == 1) & (test_df_a == 0))[0]
    y_1_a_1_idx = np.where((test_df_y == 1) & (test_df_a == 1))[0]
    all_count = min(len(y_0_a_0_idx), len(y_0_a_1_idx), len(y_1_a_0_idx), len(y_1_a_1_idx))
    y_0_a_0_idx = np.random.choice(y_0_a_0_idx, size=all_count, replace=False)
    y_1_a_1_idx = np.random.choice(y_1_a_1_idx, size=all_count, replace=False)
    y_0_a_1_idx = np.random.choice(y_0_a_1_idx, size=all_count, replace=False)
    y_1_a_0_idx = np.random.choice(y_1_a_0_idx, size=all_count, replace=False)
    all_idxs = np.concatenate((y_0_a_0_idx, y_1_a_1_idx, y_0_a_1_idx, y_1_a_0_idx), axis=0)
    X_test, Y_test, P_test = [test_df_x[i] for i in all_idxs], np.array(test_df_y[all_idxs]), np.array(test_df_a[all_idxs])

    with open(save_dir, 'wb') as f:
        pickle.dump((X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test), f)
    return X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test

