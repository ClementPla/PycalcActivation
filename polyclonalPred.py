

import pandas as pd
from scipy.stats import spearmanr, pearsonr, gaussian_kde
from torch import norm
from pathlib import Path
from sklearn.metrics import cohen_kappa_score 
from PycalcAct.train_function import *
import matplotlib
matplotlib.use('TkAgg') 
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder

is_regression = True
EC50_path = Path("D:\\sebastien\\PycalcActivation\\EC50.csv")

# Load best model
bestFolder = getPath(is_regression, "")[1].parent
best_model_path = Path(bestFolder).joinpath("best_model", "251230")

# find unique model name and run from the model_info.txt file in best_model_path
with open(best_model_path.joinpath("model_info.txt"), "r") as f:
    lines = f.readlines()
    model_unique_name = None
    sweep_id = None
    for line in lines:
        if "model_unique_name" in line:
            model_unique_name = line.split(":")[1].strip()
        if "sweep_id" in line:
            sweep_id = line.split(":")[1].strip()

## Find best model config in the project config file
modelFolder = getPath(is_regression, sweep_id)[1].parent
runs_df = pd.read_csv(modelFolder.joinpath("project_normalized.csv")) 
config_df = pd.read_csv(modelFolder.joinpath("project_config.csv"))
# find best model from its unique name in the runs_df
best_model_idx = runs_df[runs_df["model_unique_name"] == model_unique_name].index[0]
best_config = config_df.iloc[best_model_idx]
config = best_config.to_dict()


replace_nan_by_min = config['replace_nan_by_min']
remove_mean = config['remove_mean']
whichFFT = config['whichFFT']
numRNN = config['numRNN']
sizeRNN = config['sizeRNN']
bidir = config['bidir']
numFC = config['numFC']
sizeFC = config['sizeFC']
dropout = config['dropout']
weighted = config['weighted']
customLoss =  config['customLoss']
initial_lr=config['initial_lr']
weight_decay=config['weight_decay']
batch_size = config['batch_size']
store_best = config['store_best']
myLoss = config['myLoss']
customFilter = None

EC50 = pd.read_csv(EC50_path, index_col=None , header=None)
EC50 = {
    (row.iloc[0]): row.iloc[1]
    for _, (_, row) in enumerate(EC50.iterrows())
}

dataFolder = Path("D:\\Ca2-Analysis_McGill\\analysis\\retro\\dataset\\round1-OT")
csv_path = [dataFolder.joinpath("legend.csv"), dataFolder.joinpath("calciumRatio_normalized.csv")] 

dataset = Dataset(
    csv_path = csv_path,
    csv_pos_path=None,  # Optional
    position_to_displacement=False,
    remove_mean=remove_mean,
    replace_nan_by_min=replace_nan_by_min,
    customFilter = customFilter, 
    is_regression=is_regression, #True,
    EC50 = EC50,
    # Convert the x, y position to a single displacement value (sqrt((x(t+1)-x(t))^2 + (y(t+1)-y(t))^2)
    )

if numRNN == 1:
    dropout = 0

# Setup model
model = MixedFCTemporalModel(
    n_classes=dataset.n_classes["OTI"] if not is_regression else 1,
    n_rnn_layers=numRNN,
    rnn_hidden_size=sizeRNN,
    n_fc_layers=numFC,
    fc_hidden_size=sizeFC,
    temporal_length=dataset.length_serie["OTI"],
    input_size=dataset.features,
    bidirectional=bidir,
    pooling=None,
    dropout=dropout
)


thisModelName = best_model_path.joinpath("savedModel.pt")
checkpoint = torch.load(thisModelName, weights_only=True)
model.load_state_dict(checkpoint["best"])
model.to("cuda")
del checkpoint

# make prediction on the conc dataset

x, y = dataset.conc_batch(True, to_cuda=True)
y = y.cpu().detach().numpy().squeeze()
y_pred = model(x).squeeze().cpu().detach().numpy()

# write prediction to csv
pred_df = pd.DataFrame({"y": y, "y_pred": y_pred})
pred_df.to_csv(dataFolder.joinpath("retro_predictions.csv"), index=True)