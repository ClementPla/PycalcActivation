from operator import is_
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight


def process_dataset(df, filter_bool=None, this_dict=None, is_regression=False, col_classesAPL = 6):

    df = df[filter_bool]
    
    # Sort by class column
    sorting_indices = df[col_classesAPL].argsort()
    df_sorted = df.iloc[sorting_indices].reset_index(drop=True)

    # Extract class labels
    classes = df_sorted.iloc[:, col_classesAPL]
    unique_classes = classes.unique()

    # Extract data (last 120 columns)
    data = df_sorted.iloc[:, -120:].values

    # Map classes
    if is_regression:
        y = np.asarray([this_dict[apl] for apl in classes])
        inv_mapping = {key: value for key, value in this_dict.items() if key in unique_classes}
        mapping = {v: k for k, v in inv_mapping.items()}

    else:
        y = np.asarray(classes.astype("category").cat.codes)
        mapping = {i: cls for i, cls in enumerate(unique_classes)}
        inv_mapping = {v: k for k, v in mapping.items()}

    return {
        "x": data,
        "classes": classes,
        "y": y,
        "mapping": mapping,
        "inv_mapping": inv_mapping,
        "filter": filter_bool,
        "sorting": sorting_indices,
        "n_classes": len(unique_classes),
        "n_series" : data.shape[0],
        "length_serie" : data.shape[-1],
    }

def concatenate_other_data(all_data, datapath):  
    dataset_names = all_data.keys()
    for k in dataset_names:
        this_filter = all_data[k]["filter"]
        this_sorting = all_data[k]["sorting"]
        datas = []
        fnames = []#
        for p in datapath:
            if p.name != "legend.csv":
                fnames.append(p.stem)#
                df = pd.read_csv(p, header=None)
                df = df[this_filter]#
                df = df.iloc[this_sorting]#
                data = df.iloc[:, -120:].values
                data = np.expand_dims(data, axis=1)
                datas.append(data)
                
        x = np.concatenate(datas, axis=1)#
        all_data[k]["x"] = x
        all_data[k]["fnames"] = fnames


    return all_data

def concatenate_postion_data(f, xxx, yyy, position_to_displacement):  
    dataset_names = f.all_data.keys()
    for k in dataset_names:
        this_filter = f.all_data[k]["filter"]
        this_sorting = f.all_data[k]["sorting"]
        xx = xxx[this_filter]
        xx = xx.iloc[this_sorting].values
        yy = yyy[this_filter]
        yy = yy.iloc[this_sorting].values
        if position_to_displacement:
            xx = np.expand_dims(xx, axis=1)
            yy = np.expand_dims(yy, axis=1)

            dxx = np.diff(xx, axis=-1, prepend=0)
            dyy = np.diff(yy, axis=-1, prepend=0)

            d = np.sqrt(dxx**2 + dyy**2)
            pos_features = d
            f.all_data[k]["fnames"] += ["Displacement"] ####
        else:
            # find first non-NA index for each row
            first_non_na = np.array([np.argmax(~np.isnan(row)) for row in xx], dtype=np.int16)
            # Subtract the first non-NaN value from each row
            xx_adjusted = np.array([row - row[first_non_na[i]] for i, row in enumerate(xx)])
            yy_adjusted = np.array([row - row[first_non_na[i]] for i, row in enumerate(yy)])

            xx_adjusted = np.expand_dims(xx_adjusted, axis=1)
            yy_adjusted = np.expand_dims(yy_adjusted, axis=1)

            pos_features = np.concatenate((xx_adjusted, yy_adjusted), axis=1)
            f.all_data[k]["fnames"] += ["X", "Y"] ####
        
        pos_features = np.nan_to_num(pos_features)
        f.all_data[k]["x"] = np.concatenate((f.all_data[k]["x"], pos_features), axis=1)

    return f

class FromLegendFileCSV:
    def __init__(self, csv_path, customFilter, is_regression):
        self.datapath = Path(csv_path)
        datapath = Path(csv_path)

        df = pd.read_csv(datapath, header=None, low_memory=False)
        col_ageSpe = 0
        col_activated = 2
        col_peptide = 5
        col_classesAPL = 6
        col_conc = 7
        col_CTFR = 8
        col_customFilter = 9
        col_user = 11
        # scaling_factor = 0.4
        this_dict = {
                "N4" : 2.28e-13,
                "Q4" : 7.37e-11,
                "T4" : 4.76e-10,
                "Q4H7" : 2.46e-9,
                "M9" : 2.64e-12,
                "L6F" : 1e-8,
                "C9" : 5.29e-8,
                "OT3_N4" : 2.34e-11,
                "OT3_Q4" : 3.92e-12,
                "-6" : 2.28e-13,
                "-8" : 2.28e-13,
                "-10" : 2.28e-13,
                "-12" : 2.28e-13,
            }
        this_dict = {k:np.log10(v) for k,v in this_dict.items()}
        myDay = pd.DataFrame([d for d in df[col_customFilter]])

        if customFilter == None:
            filter_bool_OTI = ((df[col_ageSpe] == 1) & (df[col_activated] == 1) & 
                            (df[col_peptide] == "OVA") & (df[col_conc] == "-6") & 
                            (df[col_classesAPL].isin(["N4", "Q4", "T4", "Q4H7"]))
                            & (df[col_user] == "ST")).values
        else:
            filter_bool_OTI = ((df[col_ageSpe] == 1) & (df[col_activated] == 1) & 
                            (df[col_peptide] == "OVA") & (df[col_conc] == "-6") & 
                            (df[col_classesAPL].isin(["N4", "Q4", "T4", "Q4H7"])) 
                            & (df[col_user] == "ST") & (myDay[0] != customFilter)).values
            

        filter_bool_SL = ((df[col_ageSpe] == 1) & (df[col_activated] == 1) & 
                            (df[col_peptide] == "OVA") & (df[col_conc] == "-6") & 
                            (df[col_classesAPL].isin(["N4", "Q4", "T4", "Q4H7"]))
                            & (df[col_user] == "SL")).values

        filter_bool_P14 =   ((df[col_ageSpe] == 1) & (df[col_activated] == 1) & 
                                (df[col_peptide] == "gp33") & (df[col_conc] == "-6") & 
                                (df[col_user] == "ST")).values

        ot3_mask = pd.Series(["OT3" in apl for apl in df[col_classesAPL]], index=df.index)
        filter_bool_OT3 =   ((df[col_ageSpe] == 1) & (df[col_activated] == 1) & 
                                (df[col_peptide] == "OVA") & ot3_mask &
                                (df[col_CTFR] == "OT3-CTFR") & (df[col_user] == "ST")).values   

        filter_bool_conc = ((df[col_ageSpe] == 1) & (df[col_activated] == 1) & 
                            (df[col_classesAPL] == "N4") & (df[col_conc].isin(["-6", "-8", "-10", "-12"])) & 
                            (df[col_user] == "ST")).values   

        self.all_data = {
            "OTI": process_dataset(df, filter_bool=filter_bool_OTI, this_dict=this_dict, is_regression=is_regression, col_classesAPL = col_classesAPL),
            "SL": process_dataset(df, filter_bool=filter_bool_SL, this_dict=this_dict, is_regression=is_regression, col_classesAPL = col_classesAPL),
            "P14": process_dataset(df, filter_bool=filter_bool_P14, this_dict=this_dict, is_regression=is_regression, col_classesAPL = col_classesAPL),
            "OT3": process_dataset(df, filter_bool=filter_bool_OT3, this_dict=this_dict, is_regression=is_regression, col_classesAPL = col_classesAPL),
            "conc": process_dataset(df, filter_bool=filter_bool_conc, this_dict=this_dict, is_regression=is_regression, col_classesAPL = col_conc),
        }
    @property
    def features_names(self):
        return [f"Normalized Ratio {self.datapath.stem}"]

class FromMultiFileCSV:
    def __init__(self, csv_path, customFilter, is_regression):
        assert isinstance(csv_path, list), "csv_path should be a list of paths"
        datapath = [Path(p) for p in csv_path]
        assert all([p.exists() for p in datapath]), "All paths should exist"
        assert "legend.csv" in [p.name for p in datapath], "legend.csv should be present in the list of paths"

        self.flegend = FromLegendFileCSV([p for p in datapath if p.name == "legend.csv"][0], customFilter, is_regression) ####
        self.all_data = concatenate_other_data(self.flegend.all_data, datapath) ####

    @property
    def features_names(self):
        return {k:self.all_data[k]["fnames"] for k in self.all_data.keys()}

    @property
    def x(self):
        return {k:self.all_data[k]["x"] for k in self.all_data.keys()}

    @property
    def y(self):
        return {k:self.all_data[k]["y"] for k in self.all_data.keys()}

    @property
    def classes(self):
        return {k:self.all_data[k]["classes"] for k in self.all_data.keys()}

    @property
    def sorting(self):
        return {k:self.all_data[k]["sorting"] for k in self.all_data.keys()}

    @property
    def filter(self):
        return {k:self.all_data[k]["filter"] for k in self.all_data.keys()}

    @property
    def mapping(self):
        return {k:self.all_data[k]["mapping"] for k in self.all_data.keys()}

    @property
    def inv_mapping(self):
        return {k:self.all_data[k]["inv_mapping"] for k in self.all_data.keys()}
    
    @property
    def n_series(self):
        return {k:self.all_data[k]["n_series"] for k in self.all_data.keys()}
    
    @property
    def length_serie(self):
        return {k:self.all_data[k]["length_serie"] for k in self.all_data.keys()}
    
    @property
    def n_classes(self):
        return {k:self.all_data[k]["n_classes"] for k in self.all_data.keys()}


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
        customFilter = None,
        is_regression = False,
    ):
        # Load data from file
        if isinstance(csv_path, str) or isinstance(csv_path, Path):
            f = FromLegendFileCSV(csv_path, customFilter, is_regression)
        elif isinstance(csv_path, list):
            f = FromMultiFileCSV(csv_path, customFilter, is_regression)
        
        # transfer attributes of dataset to self
        self.f = f 
        self.features_names = f.features_names["OTI"]
        self.remove_mean = remove_mean 
        self.replace_nan_by_min = replace_nan_by_min
        self.is_regression = is_regression
        self.position_to_displacement = position_to_displacement
        self.csv_pos_path = csv_pos_path
        self.n_series = f.n_series
        self.length_serie = f.length_serie
        self.n_classes = f.n_classes
        self._autocuda = True
                
        # Add position if needed
        if csv_pos_path is not None:
            df_pos = pd.read_csv(csv_pos_path, header=None)
            xx = df_pos.iloc[::2]
            yy = df_pos.iloc[1::2]
            ### Do something about this
            self.f  = concatenate_postion_data(self.f, xx, yy, position_to_displacement)

        
        for k in self.f.all_data.keys():
            # make sure the dataset has 3 dimensions
            if self.f.all_data[k]["x"].ndim == 2:
                self.f.all_data[k]["x"] =  np.expand_dims(self.f.all_data[k]["x"], axis=1)
            elif self.f.all_data[k]["x"].ndim != 3:
                raise ValueError("Data should be 2D or 3D")
            
            # remove mean and replace nan if necessary
            calcium_dims = [i for i, n in enumerate(self.f.all_data[k]["fnames"]) if 'calcium' in n]
            for dim in calcium_dims:
                for row in self.f.all_data[k]["x"][:,dim,:]:
                    if remove_mean:
                        mean_val = np.nanmean(row)
                        row -= mean_val
                    if replace_nan_by_min:
                        min_val = np.nanmin(row)
                    else:
                        min_val = 0
                    row[np.isnan(row)] = min_val
        sk = StratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=seed)
        skval = StratifiedShuffleSplit(n_splits=2, test_size=val_size, random_state=seed)

        x = self.f.all_data["OTI"]["x"]
        y = self.f.all_data["OTI"]["y"]

        train_idx, test_idx = next(sk.split(x, y))

        self.x_train = x[train_idx]
        self.x_test = x[test_idx]

        if not self.is_regression:
            self.y_train = y[train_idx].astype(int)
            self.y_test = y[test_idx].astype(int)
        else:
            self.y_train = y[train_idx]
            self.y_test = y[test_idx]
                
        train_idx, val_idx = next(skval.split(self.x_train, self.y_train))
        self.x_val = self.x_train[val_idx]
        self.y_val = self.y_train[val_idx]
        self.x_train = self.x_train[train_idx]
        self.y_train = self.y_train[train_idx]

        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

        self.train_data = x

    def drop_features(self, index):
        self.x_train = np.delete(self.x_train, index, axis=1)
        self.x_val = np.delete(self.x_val, index, axis=1)
        self.x_test = np.delete(self.x_test, index, axis=1)
        self.train_data = np.delete(self.train_data, index, axis=1)
        for k in self.f.all_data.keys():
            self.f.all_data[k]["x"] = np.delete(self.f.all_data[k]["x"], index, axis=1)
        self.features_names = np.delete(self.features_names, index).tolist()

    def drop_features_by_name(self, *name):
        if not isinstance(name, (tuple, list)):
            name = [name]
        for n in name:
            index = self.features_names.index(n)
            print(index)
            self.drop_features(index)

    def length(self, k):
        return self.f.all_data[k]["n_series"]

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

    def labels(self,k):
        return list(self.f.all_data[k]["mapping"].values())

    @property
    def features(self):
        return self.x_train.shape[1]

    @property
    def weights(self):
        class_weights = compute_class_weight("balanced", classes=np.unique(self.y_train), y=self.y_train) 
        return torch.from_numpy(class_weights).float()

    def __repr__(self):
        return self.summarize("OTI", True).__repr__()

    def num_classes(self, k):
        return self.n_classes[k]

    @property
    def summary(self):
        return self.summarize("OTI", True)

    def summarize(self, k, include_weights=False):
        labels = self.labels(k)
        data = {"Total": [], **{label: [] for label in labels}}

        if k == "OTI":
            data["Total"].append(self.x_train.shape[0])
            data["Total"].append(self.x_val.shape[0])
            data["Total"].append(self.x_test.shape[0])
            data["Total"].append(sum(data["Total"]))

            weights = self.weights
            for i, label in enumerate(labels):
                data[label].append(np.sum(self.y_train == self.f.all_data[k]["inv_mapping"][label]))
                data[label].append(np.sum(self.y_val == self.f.all_data[k]["inv_mapping"][label]))
                data[label].append(np.sum(self.y_test == self.f.all_data[k]["inv_mapping"][label]))
                data[label].append(sum(data[label]))

            df = pd.DataFrame(data, index=["Train", "Validation", "Test", "Total"])
            print(f"Dataset summary: timepoints {self.length_serie[k]}, features {self.features}")
            print("Features:")
            print(f"{' '.join(self.features_names)}")
            if include_weights:
                print("Class weights:")
                for i, label in enumerate(labels):
                    print(f"{label}: {weights[i].item():.2f}", end=" ")

        else:
            data["Total"].append(self.f.all_data[k]["x"].shape[0])

            for i, label in enumerate(labels):
                data[label].append(np.sum(self.f.all_data[k]["y"] == self.f.all_data[k]["inv_mapping"][label]))

            df = pd.DataFrame(data, index=["Total"])
            print(f"Dataset summary: timepoints {self.length_serie[k]}, features {self.features}")
            print("Features:")
            print(f"{' '.join(self.features_names)}")

        return df

    def create_new_feature_by_operations(self, operations):
        if not isinstance(operations, list):
            operations = [operations]

        for i, op in enumerate(operations):
            self.features_names.append(op.__name__)
            for j, x in enumerate([self.x_train, self.x_val, self.x_test, self.f.all_data["OTI"]["x"], self.f.all_data["P14"]["x"], self.f.all_data["OT3"]["x"], self.f.all_data["SL"]["x"], self.f.all_data["conc"]["x"]]):
                new_feature = np.zeros((x.shape[0], len(operations), x.shape[2]), dtype=x.dtype)
                new_feature[:, i, :] = op(x)
                x = np.concatenate((x, new_feature), axis=1)

                match j:
                    case 0:
                        self.x_train = x
                    case 1:
                        self.x_val = x
                    case 2:
                        self.x_test = x
                    case 3:
                        self.f.all_data["OTI"]["x"] = x
                    case 4:
                        self.f.all_data["P14"]["x"] = x
                    case 5:
                         self.f.all_data["OT3"]["x"] = x
                    case 6:
                        self.f.all_data["SL"]["x"] = x
                    case 7:
                        self.f.all_data["conc"]["x"] = x

    def get_class_count(self, y):
        return np.bincount(y)

    def test_data(self):
        return self.x_test, self.y_test

    def val_data(self):
        return self.x_val, self.y_val
    
    def train_data(self):
        return self.x_train, self.y_train

    def batch_from_data(self, x, y, time_first=False, to_cuda=True):
        """Return a batch from the data
        @param x: the input data as a numpy array
        @param y: the target data as a numpy array
        @param time_first: if True, the time dimension is the second one in the input data.
        If False, the time dimension is the last one.
        time_first is useful for RNNs, to create an input of shape BxTxF where B is the batch size,
        T is the time dimension and F is the feature dimension, instead of BxFxT.
        @param to_cuda: if True, the data is moved to the GPU

        """
        x = torch.from_numpy(np.array(x, copy=True)).float()
        y = torch.from_numpy(np.array(y, copy=True))
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

    def list_hparams(self):
        return {
            "num_features": self.features,
            "features_names": self.f.all_data["OTI"]["fnames"] ,
            "datasets": self.f.all_data.keys(),
            "n_series":  [self.f.all_data[k]["n_series"] for k in self.f.all_data.keys()],
            "length_serie": [self.f.all_data[k]["length_serie"] for k in self.f.all_data.keys()],
            "n_classes": [self.f.all_data[k]["n_classes"] for k in self.f.all_data.keys()],
            "position_to_displacement": self.position_to_displacement,
        }
