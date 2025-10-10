from math import e
import numpy as np
from sympy import Q
import torch
import matplotlib.pyplot as plt
import pandas as pd
from PycalcAct.dataset import Dataset
from PycalcAct.model import (
    MixedFCTemporalModel)
from PycalcAct.trainer import Trainer
from PycalcAct.calculateCustomAccuracy import calculateCustomAccuracy
from pathlib import Path
from PycalcAct.myCustomCriterion import myCustomCriterion, myFScore
_ = torch.manual_seed(1234)
from socket import gethostname
from argparse import _ArgumentGroup
from copy import deepcopy
from functools import partial
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import is_regressor
import torch
import torch.nn.functional as F
from colorama import Fore, Style
from progress_table import ProgressTable
from torchinfo import summary as torch_summary
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, CohenKappa, ConfusionMatrix

from PycalcAct.utils.wrapper import on_keyboard_interrup
from PycalcAct.myCustomCriterion import myMSELoss, myFScore
from scipy.stats import f_oneway


if gethostname() == 'HM_Lab':
    conditionPath = Path("D:/sebastien/PycalcActivation/trainingOptions_round3.csv")
    dataFolder = Path("D:/Ca2-Analysis_McGill/prediction/agAffinity/datasets/dataset_mcgill")
    saveFolder = Path("D:/sebastien/PycalcActivation/models/round3")
elif gethostname() == 'Hmr_lymph':
    conditionPath = Path('D:/SebastienThis/CalciumPredictions/PycalcActivation/trainingOptions_round2.csv')
    dataFolder = Path("D:/SebastienThis/CalciumPredictions/Ca2-Analysis_McGill/prediction/agAffinity/datasets/trainingData")
    saveFolder = Path("D:/SebastienThis/CalciumPredictions/Ca2-Analysis_McGill/prediction/agAffinity/models")
elif gethostname() == 'HMR-BLOOD':
    conditionPath = Path('D:/Sebastien/PycalcActivation/trainingOptions_round3.csv')
    dataFolder = Path("D:/Sebastien/Ca2-Analysis_McGill/prediction/agAffinity/datasets/dataset_mcgill")
    saveFolder = Path("D:/sebastien/PycalcActivation/models/round3")

myCondition = pd.read_csv(conditionPath, header=None)
myLegend = myCondition.iloc[0,:]
myCondition = myCondition.iloc[1:,:]
myCondition = [myCondition.iloc[i,:].to_numpy() for i in range(0, myCondition.shape[0])] 

cdt = myCondition[1]
# Load training conditions
modelNum = cdt[0]
whichDataset = cdt[1]
whichDisplacement = int(cdt[2]) == 1
whichReplaceNan = int(cdt[3]) == 1
whichRemoveMean = int(cdt[4]) == 1
whichFFT = int(cdt[5]) == 1
numRNN = int(cdt[6])
sizeRNN = int(cdt[7])
bidir = int(cdt[8]) == 1
numFC = int(cdt[9])
sizeFC = int(cdt[10])
dropout = float(cdt[11])
weighted = int(cdt[8]) == 1
customLoss =  int(cdt[13]) == 1
xyDisplacement = int(cdt[14]) == 1
is_regression = int(cdt[15]) == 1
augment_gt = cdt[16]

donePath = saveFolder.joinpath('done.npy')
if donePath.exists():
    done = np.load(donePath)
else:
    done = np.array([])
    
if np.isin(done,modelNum).any():
    print("Already trained")
else:

    print(f"NumberModel = {modelNum}/{len(myCondition)}, Regression = {is_regression},  Dataset = {whichDataset}, Displacement = {whichDisplacement},  Displacement as XY = {xyDisplacement}, Replace Nan by mean = {whichReplaceNan}, Remove Mean = {whichRemoveMean}, FFT = {whichFFT} \n #RNN = {numRNN}, size RNN = {sizeRNN}, bidirectional = {bidir}, #FC = {numFC}, size FC = {sizeFC}, dropout = {dropout}")

    # create model folder
    thisPath = saveFolder.joinpath(modelNum)
    thisPath.mkdir(parents=True, exist_ok=True)
        
    # Setup Dataset
    customFilter = None
    match whichDataset:
        case "ratio":
            csv_path=[dataFolder.joinpath("legend.csv"), dataFolder.joinpath("calciumRatio.csv")]          
        case "ratioNorm":
            csv_path=[dataFolder.joinpath("legend.csv"), dataFolder.joinpath("calciumRatio_normalized.csv")] 
        case "indiv":
            csv_path=[dataFolder.joinpath("legend.csv"), dataFolder.joinpath("calciumFree.csv"), dataFolder.joinpath("calciumBound.csv")] 
        case _:
            csv_path=[dataFolder.joinpath("legend.csv"), dataFolder.joinpath("calciumRatio_normalized.csv")] 
            customFilter = cdt[1]

    if whichDisplacement:
        csv_pos_path = dataFolder.joinpath("position.csv")
        if xyDisplacement:
            position_to_displacement = False
        else:
            position_to_displacement = True
    else:
        csv_pos_path = None
        position_to_displacement = False

    replace_nan_by_min = True if whichReplaceNan else False
    remove_mean = True if whichRemoveMean else False
    

    if augment_gt == "N":
        augment_gt = None
    else:
        augment_gt = "Ca"

    dataset = Dataset(
        csv_path = csv_path,
        csv_pos_path=csv_pos_path,  # Optional
        position_to_displacement=position_to_displacement,
        remove_mean=remove_mean,
        replace_nan_by_min=replace_nan_by_min,
        customFilter = customFilter, 
        is_regression=is_regression, #True,
        augment_gt = augment_gt,
        forEval = False,
        # Convert the x, y position to a single displacement value (sqrt((x(t+1)-x(t))^2 + (y(t+1)-y(t))^2)
        )

    if whichFFT:
        dataset.create_new_feature_by_operations(lambda x: np.abs(np.fft.fft(x[:, 0])))
        if whichDataset == "indiv":
            dataset.create_new_feature_by_operations(lambda x: np.abs(np.fft.fft(x[:, 1])))

    if numRNN == 1:
        dropout = 0

    # Setup model
    model = MixedFCTemporalModel(
        n_classes = dataset.n_classes if not is_regression else 1,
        n_rnn_layers=numRNN,
        rnn_hidden_size=sizeRNN,
        n_fc_layers=numFC,
        fc_hidden_size=sizeFC,
        temporal_length=dataset.length_serie,
        input_size=dataset.features,
        bidirectional=bidir,
        pooling=None,
        dropout=dropout
    )
    # Setup training
    n_epochs = 500
    criterion = None
    D = torch.tensor([  [1,2,3,4], 
                        [3,1,4,2],
                        [4,3,1,2],
                        [3,2,3,1]]).to("cuda")
    
    if customLoss:
        criterion = myCustomCriterion(weight = dataset.weights, device = "cuda", D = D)

    trainer = Trainer(
        dataset,
        model,
        device="cuda",
        batch_size=500 if is_regression else 2000,
        criterion= criterion,
        store_best='myFScore' if is_regression else 'Accuracy',
        use_class_weights= weighted,
        is_regression = is_regression,
        regression_bounds_y = torch.Tensor([-12,-10,-9]).to("cuda") if is_regression else None,
        regression_bounds_y_pred = torch.Tensor([-12,-10,-9]).to("cuda") if is_regression else None,
    )   
    # train model
    if trainer.scheduler is None:
            trainer.scheduler = trainer.default_scheduler()(T_max=n_epochs)
            trainer._initial_scheduler_state_dict = deepcopy(trainer.scheduler.state_dict())
            trainer._initial_scheduler_state_dict = deepcopy(trainer.scheduler.state_dict())

    x, y = trainer.dataset.train_batch(True, to_cuda=True)

    batch_size = len(x) if trainer.batch_size is None else trainer.batch_size

    current_best = 0
    xval, yval = trainer.dataset.val_batch(True, to_cuda=True)

    idx = torch.randperm(x.shape[0], device=x.device)
    x = x[idx]
    y = y[idx]

    trainer.model.train()
    i = 0
    trainer.optim.zero_grad()
    x_batch = x[i : i + batch_size]
    if trainer.is_regression:
        y_batch = y[i : i + batch_size].type(torch.FloatTensor).to(trainer.device)
    else:
        y_batch = y[i : i + batch_size].type(torch.LongTensor).to(trainer.device)
            # Take batch size:

    with torch.autocast("cuda"): #torch.cuda.amp.autocast()
        y_pred = trainer.model(x_batch)
        if trainer.is_regression:
            y_pred = y_pred.squeeze()
            y_batch = y_batch.squeeze()
        loss = trainer.criterion(y_pred, y_batch)
            
    loss.backward()
    trainer.optim.step()

    if trainer.scheduler:
        trainer.scheduler.step()
    x = xval
    y = yval.type(torch.FloatTensor).to(trainer.device)
    trainer.model.eval()
    trainer.metrics.reset()
    trainer.confmat.reset()

    batch_size = len(x) if trainer.batch_size is None else trainer.batch_size

    i = 0 
    xbatch = x[i : i + batch_size]
    if trainer.is_regression:
        ybatch = y[i : i + batch_size].type(torch.FloatTensor).to(trainer.device)
    else:
        ybatch = y[i : i + batch_size].type(torch.LongTensor).to(trainer.device)

    y_pred = trainer.model(xbatch)
    if trainer.is_regression:
        y_pred = y_pred.squeeze()
    loss = trainer.criterion(y_pred, ybatch)
    if trainer.is_regression:
        # From continuous to categorical
        # use regression_bounds to define the thresholds
        y_pred = torch.bucketize(y_pred, trainer.regression_bounds_y_pred).type(torch.FloatTensor).to(trainer.device)
        ybatch = torch.bucketize(ybatch, trainer.regression_bounds_y).to(trainer.device)
    else:
        y_pred = torch.softmax(y_pred, dim=1)
    trainer.metrics.update(y_pred, ybatch)
    trainer.confmat.update(y_pred, ybatch)


    if trainer.is_regression:
        loss, scores = trainer.eval(xval, yval.type(torch.FloatTensor).to(trainer.device))
    else:
        loss, scores = trainer.eval(xval, yval.type(torch.LongTensor).to(trainer.device))
