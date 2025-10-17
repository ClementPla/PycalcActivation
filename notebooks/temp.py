import numpy as np
import matplotlib.pyplot as plt
from PycalcAct.calculateCustomAccuracy import calculateCustomAccuracy
from pathlib import Path
import torch
_ = torch.manual_seed(1234)
from socket import gethostname
from pathlib import Path
import numpy as np
from PycalcAct.myCustomCriterion import myFScore
from PycalcAct.train_function import *


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


config = {
        "whichDataset" : "ratioNorm",
        "whichDisplacement" : False,
        "replace_nan_by_min" :True,
        "remove_mean": False,
        "whichFFT" : False,
        "numRNN" : 1,
        "sizeRNN": 16,
        "bidir" : True,
        "numFC": 1,
        "sizeFC" :  16,
        "dropout" : 0.2,
        "weighted" : True,
        "customLoss" : False,
        "xyDisplacement" : False,
        "initial_lr": 0.001,
        "weight_decay": 1e-4,
        "batch_size": 1024,
        }

is_regression = True
model_unique_name, trainer = setupTrainer(config, is_regression)
n_epoch = 10
trainer.train(n_epoch)
metric_train, metric_val, metric_test = save_model_perf(trainer, model_unique_name)
