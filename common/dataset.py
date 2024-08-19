import os

import numpy as np
import pandas as pd
import torchvision.datasets as dset
import torchvision.transforms as trn
from PIL import Image
from torch.utils.data import Dataset
from .datasets import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def build_dataset(dataset_name, transform=None, mode='train'):
    #  path of usr
    usr_dir = os.path.expanduser('~')
    data_dir = os.path.join(usr_dir, "data")

    if dataset_name == 'imagenet':
        if transform is None:
            transform = trn.Compose([
                trn.Resize(256),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
            ])

        dataset = dset.ImageFolder(data_dir + "/imagenet/val", transform)
    elif dataset_name == 'mnist':
        if transform is None:
            transform = trn.Compose([
                trn.ToTensor(),
                trn.Normalize((0.1307,), (0.3081,))
            ])
        if mode == "train":
            dataset = dset.MNIST(data_dir, train=True, download=True, transform=transform)
        elif mode == "test":
            dataset = dset.MNIST(data_dir, train=False, download=True, transform=transform)
    else:
        raise NotImplementedError

    return dataset


base_path = "data/"
base_dataset_path = "dataset/" # "C:/Users/lorry/OneDrive - City University of Hong Kong/Documents/github repos/20231229-er_cp/conformal_regression/examples/datasets/" # '../../datasets/'

def build_reg_data(dataset_name, ratio_train=0.6, test_ratio=0.2, seed=42, normalize=True):
    # Load the data
    try:
        X, y = datasets.GetDataset(dataset_name, base_dataset_path)
        print("Loaded dataset '" + dataset_name + "'.")
    except:
        print("Error: cannot load dataset " + dataset_name)
        return

    # Dataset is divided into test and train data based on test_ratio parameter
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

    # Reshape the data
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    n_train = X_train.shape[0]

    # Print input dimensions
    print("Data size: train (%d, %d), test (%d, %d)" % (X_train.shape[0], X_train.shape[1],
                                                        X_test.shape[0], X_test.shape[1]))

    # Set seed for splitting the data into proper train and calibration
    np.random.seed(seed)
    idx = np.random.permutation(n_train)

    # Divide the data into proper training set and calibration set
    n_half = int(np.floor(n_train * ratio_train))
    idx_train, idx_cal = idx[:n_half], idx[n_half:n_train]
    if normalize==False:
        return X_train[idx_train], X_train[idx_cal], X_test, y_train[idx_train], y_train[idx_cal], y_test
    # Zero mean and unit variance scaling of the train and test features
    scalerX = StandardScaler()
    scalerX = scalerX.fit(X_train[idx_train])
    X_propertrain = scalerX.transform(X_train[idx_train])
    X_calib = scalerX.transform(X_train[idx_cal])
    X_test = scalerX.transform(X_test)

    # Scale the labels by dividing each by the mean absolute response
    mean_ytrain = np.mean(np.abs(y_train[idx_train]))
    y_propertrain = np.squeeze(y_train[idx_train])/mean_ytrain
    y_calib = np.squeeze(y_train[idx_cal])/mean_ytrain
    y_test = np.squeeze(y_test)/mean_ytrain

    return X_propertrain, X_calib, X_test, y_propertrain, y_calib, y_test


def build_reg_data_old(data_name="community"):
    if data_name == "community":
        # https://github.com/vbordalo/Communities-Crime/blob/master/Crime_v1.ipynb
        attrib = pd.read_csv(base_path + 'communities_attributes.csv', delim_whitespace=True)
        data = pd.read_csv(base_path + 'communities.data', names=attrib['attributes'])
        data = data.drop(columns=['state', 'county',
                                  'community', 'communityname',
                                  'fold'], axis=1)
        data = data.replace('?', np.nan)

        # Impute mean values for samples with missing values
        data['OtherPerCap'] = data['OtherPerCap'].astype("float")
        mean_value = data['OtherPerCap'].mean()
        data['OtherPerCap'].fillna(value=mean_value, inplace=True)
        data = data.dropna(axis=1)
        X = data.iloc[:, 0:100].values
        y = data.iloc[:, 100].values

    elif data_name == "synthetic":
        X = np.random.uniform(0, 10, 5000)
        noise = np.random.normal(0, 0.1 * X)
        y = X + noise
        y = y*100
        X = X.reshape(-1, 1)
        # X = np.random.rand(500, 5)
        # y_wo_noise = 10 * np.sin(X[:, 0] * X[:, 1] * np.pi) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
        # eplison = np.zeros(500)
        # phi = theta = 0.8
        # delta_t_1 = np.random.randn()
        # for i in range(1, 500):
        #     delta_t = np.random.randn()
        #     eplison[i] = phi * eplison[i - 1] + delta_t_1 + theta * delta_t
        #     delta_t_1 = delta_t
        #
        # y = y_wo_noise + eplison

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    return X, y

