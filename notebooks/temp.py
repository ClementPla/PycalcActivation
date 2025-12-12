# C:/Users/labme/.conda/envs/calcium/python.exe
from pathlib import Path
import torch
_ = torch.manual_seed(1234)
from socket import gethostname
from PycalcAct.train_function import *

if gethostname() == 'HM_Lab':
    dataFolder = Path("D:/Ca2-Analysis_McGill/prediction/agAffinity/datasets/dataset_mcgill")
    saveFolder = Path("D:/sebastien/PycalcActivation/models/round3")
elif gethostname() == 'Hmr_lymph':
    dataFolder = Path("D:/SebastienThis/CalciumPredictions/Ca2-Analysis_McGill/prediction/agAffinity/datasets/trainingData")
    saveFolder = Path("D:/SebastienThis/CalciumPredictions/Ca2-Analysis_McGill/prediction/agAffinity/models")
elif gethostname() == 'HMR-BLOOD':
    dataFolder = Path("D:/Sebastien/Ca2-Analysis_McGill/prediction/agAffinity/datasets/dataset_mcgill")
    saveFolder = Path("D:/sebastien/PycalcActivation/models/round3")
    
    config = {
            "whichDataset" : "ratio",
            "whichDisplacement" : "displacement",
            "replace_nan_by_min" :False,
            "remove_mean": False,
            "whichFFT" : True,
            "numRNN" : 3,
            "sizeRNN": 32,
            "bidir" : True,
            "numFC": 3,
            "sizeFC" :  4,
            "dropout" : 0.1,
            "weighted" : False,
            "customLoss" : False,
            "initial_lr": 0.01,
            "weight_decay": 0.0001,
            "batch_size": 2048,
            "store_best": "Accuracy", # Accuracy, CohenKappa, myFScore, mySpearman
            "loss": "None",
            }

sweep_id = ""
is_regression = True

model_unique_name, trainer = setupTrainer(config, is_regression, sweep_id, model_unique_name=None, EC50_path = "D:\sebastien\PycalcActivation\EC50.csv")
n_epoch = 200
trainer.train(n_epoch, val_patience=200)
metric_train, metric_val, metric_test, a, b = save_model_perf(trainer, model_unique_name,  sweep_id, True)
metrics = save_model_generalizability(trainer, model_unique_name, sweep_id, True