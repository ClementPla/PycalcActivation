from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight


class Dataset:
    def __init__(
        self,
        csv_path,
        csv_pos_path=None,
        position_to_displacement=True,
        test_size=0.2,
        val_size=0.2,
        seed=1234,
        remove_mean=False,
        replace_nan_by_min=True,
    ):
        datapath = Path(csv_path)

        df = pd.read_csv(datapath, header=None)

        col_ageSpeActivated = 4
        col_classesAPL = 9

        filter_bool = ((df[col_ageSpeActivated] == 1) & (df[col_classesAPL] != " ")).values
        df = df[filter_bool]
        sorting_indices = df[col_classesAPL].argsort()

        df = df.iloc[sorting_indices]

        if csv_pos_path is not None:
            df_pos = pd.read_csv(csv_pos_path, header=None)
            df_pos.fillna(0, inplace=True)
            xx = df_pos.iloc[::2]
            yy = df_pos.iloc[1::2]
            xx = xx[filter_bool]
            xx = xx.iloc[sorting_indices].values
            yy = yy[filter_bool]
            yy = yy.iloc[sorting_indices].values
            xx = np.expand_dims(xx, axis=1)
            yy = np.expand_dims(yy, axis=1)
            if position_to_displacement:
                dxx = np.diff(xx, axis=-1, prepend=0)
                dyy = np.diff(yy, axis=-1, prepend=0)

                d = np.sqrt(dxx**2 + dyy**2)
                pos_features = d

            else:
                pos_features = np.concatenate((xx, yy), axis=1)

        classes = df.iloc[:, col_classesAPL]
        unique_classes = classes.unique()

        self.mapping = {k: v for k, v in enumerate(unique_classes)}
        self.inv_mapping = {v: k for k, v in self.mapping.items()}

        classes_int = np.asarray(classes.astype("category").cat.codes)
        sk = StratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=seed)
        skval = StratifiedShuffleSplit(n_splits=2, test_size=val_size, random_state=seed)
        x = df.iloc[:, -120:].values
        x = np.expand_dims(x, axis=1)

        for row in x:
            if remove_mean:
                mean_val = np.nanmean(row)
                row -= mean_val
            if replace_nan_by_min:
                min_val = np.nanmin(row)
            else:
                min_val = 0
            row[np.isnan(row)] = min_val

        if csv_pos_path is not None:
            x = np.concatenate((x, pos_features), axis=1)

        self.n_series = x.shape[0]
        self.features = x.shape[1]
        self.length_serie = x.shape[-1]
        self.n_classes = len(unique_classes)

        train_idx, test_idx = next(sk.split(x, classes_int))

        self.x_train = x[train_idx]
        self.x_test = x[test_idx]

        self.y_train = classes_int[train_idx].astype(int)
        self.y_test = classes_int[test_idx].astype(int)

        train_idx, val_idx = next(skval.split(self.x_train, self.y_train))
        self.x_val = self.x_train[val_idx]
        self.y_val = self.y_train[val_idx]
        self.x_train = self.x_train[train_idx]
        self.y_train = self.y_train[train_idx]

        self.data = x

        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

        self._autocuda = True

    def __len__(self):
        return self.n_series

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

    @property
    def labels(self):
        return list(self.mapping.values())

    @property
    def weights(self):
        class_weights = compute_class_weight("balanced", classes=np.unique(self.y_train), y=self.y_train)

        return torch.from_numpy(class_weights).float()

    def __repr__(self):
        return self.summarize(True).__repr__()

    @property
    def num_classes(self):
        return self.n_classes

    @property
    def summary(self):
        return self.summarize(True)

    def summarize(self, include_weights=False):
        labels = self.labels
        data = {"Total": [], **{label: [] for label in labels}}

        data["Total"].append(self.x_train.shape[0])
        data["Total"].append(self.x_val.shape[0])
        data["Total"].append(self.x_test.shape[0])
        data["Total"].append(sum(data["Total"]))

        weights = self.weights
        for i, label in enumerate(labels):
            data[label].append(np.sum(self.y_train == self.inv_mapping[label]))
            data[label].append(np.sum(self.y_val == self.inv_mapping[label]))
            data[label].append(np.sum(self.y_test == self.inv_mapping[label]))
            data[label].append(sum(data[label]))

        df = pd.DataFrame(data, index=["Train", "Validation", "Test", "Total"])
        print(f"Dataset summary: timepoints {self.length_serie}, features {self.features}")
        if include_weights:
            print("Class weights:")
            for i, label in enumerate(labels):
                print(f"{label}: {weights[i].item():.2f}", end=" ")
        return df

    def get_class_count(self, y):
        return np.bincount(y)

    def test_data(self):
        return self.x_test, self.y_test

    def train_data(self):
        return self.x_train, self.y_train

    def batch_from_data(self, x, y, time_first=False, to_cuda=True):
        """Return a batch from the data
        @param x: the input data as a numpy array
        @param y: the target data as a numpy array
        @param time_first: if True, the time dimension is the second one in the input data. If False, the time dimension is the last one.
        time_first is useful for RNNs, to create an input of shape BxTxF where B is the batch size, T is the time dimension and F is the feature dimension, instead of BxFxT.
        @param to_cuda: if True, the data is moved to the GPU

        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if time_first:
            x = x.permute(0, 2, 1)
        if to_cuda:
            x = x.cuda()
            y = y.cuda()
        return x, y

    def train_batch(self, time_first=False, to_cuda=True):
        """Return the whole training data"""
        return self.batch_from_data(self.x_train, self.y_train, time_first, to_cuda and self._autocuda)

    def test_batch(self, time_first=False, to_cuda=True):
        """Return the whole test data"""
        return self.batch_from_data(self.x_test, self.y_test, time_first, to_cuda and self._autocuda)

    def val_batch(self, time_first=False, to_cuda=True):
        """Return the whole validation data"""
        return self.batch_from_data(self.x_val, self.y_val, time_first, to_cuda and self._autocuda)

    def to(self, device):
        if device == "cuda":
            self._autocuda = True
        else:
            self._autocuda = False
        return self
