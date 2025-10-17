from math import e
import numpy as np
from sympy import Q
import torch
import csv
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
import hashlib
import random
import string

def get_safe_folder_name(config):
    # Convert config to string and hash it
    config_str = str(sorted(config.items()))
    # Generate a random 8-character string
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    # Combine config string with random component
    combined_str = config_str + random_suffix

    hash_id = hashlib.md5(combined_str.encode()).hexdigest()
    return f"run_{hash_id}"


def getPath():
    if gethostname() == 'HM_Lab':
        dataFolder = Path("D:/Ca2-Analysis_McGill/prediction/agAffinity/datasets/dataset_mcgill")
        saveFolder = Path("D:/sebastien/PycalcActivation/models/round3")
    elif gethostname() == 'HMR_LYMPH':
        dataFolder = Path("D:/SebastienThis/CalciumPredictions/Ca2-Analysis_McGill/prediction/agAffinity/datasets/dataset_mcgill")
        saveFolder = Path("D:/SebastienThis/CalciumPredictions/PycalcActivation/models/round3")
    elif gethostname() == 'HMR-BLOOD':
        dataFolder = Path("D:/Sebastien/Ca2-Analysis_McGill/prediction/agAffinity/datasets/dataset_mcgill")
        saveFolder = Path("D:/sebastien/PycalcActivation/models/round3")
    
    return dataFolder, saveFolder

def setupTrainer(config, is_regression):
    whichDataset = config["whichDataset"]
    whichDisplacement = config['whichDisplacement']
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
    xyDisplacement = config['xyDisplacement']   
    initial_lr=config['initial_lr']
    weight_decay=config['weight_decay']
    batch_size = config['batch_size']
    customFilter = None
    augment_gt = False # augment_gt = config['augment_gt']
    dataFolder, saveFolder = getPath()   
    
     # create model folder
    model_unique_name = get_safe_folder_name(config)
    thisPath = saveFolder.joinpath(model_unique_name)
    thisPath.mkdir(parents=True, exist_ok=True)
        
    # Setup Dataset
    
    match whichDataset:
        case "ratio":
            csv_path=[dataFolder.joinpath("legend.csv"), dataFolder.joinpath("calciumRatio.csv")]          
        case "ratioNorm":
            csv_path=[dataFolder.joinpath("legend.csv"), dataFolder.joinpath("calciumRatio_normalized.csv")] 
        case "indiv":
            csv_path=[dataFolder.joinpath("legend.csv"), dataFolder.joinpath("calciumFree.csv"), dataFolder.joinpath("calciumBound.csv")] 

    if whichDisplacement:
        csv_pos_path = dataFolder.joinpath("position.csv")
        if xyDisplacement:
            position_to_displacement = False
        else:
            position_to_displacement = True
    else:
        csv_pos_path = None
        position_to_displacement = False

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
        n_classes=dataset.n_classes if not is_regression else 1,
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
        batch_size=batch_size,
        criterion= criterion,
        store_best='myFScore' if is_regression else 'Accuracy',
        use_class_weights = weighted,
        is_regression = is_regression,
        regression_bounds_y = torch.Tensor([-12,-10,-9]).to("cuda") if is_regression else None,
        regression_bounds_y_pred = torch.Tensor([-12,-10,-9]).to("cuda") if is_regression else None,
        initial_lr=initial_lr,
        weight_decay=weight_decay,
    ) 

    return model_unique_name, trainer


def save_model_perf(trainer, model_unique_name):
    # save model
    _, saveFolder = getPath()   
    thisPath = saveFolder.joinpath(model_unique_name)
    thisModelName = thisPath.joinpath("savedModel.pt")

    torch.save(
        {
            "model": trainer.model.state_dict(),
            "optim": trainer.optim.state_dict(),
            "scheduler": trainer.scheduler.state_dict() if trainer.scheduler else None,
            "best": trainer._best_state_dict,
            "last": trainer._last_state_dict,
        },
        Path(thisModelName),
    )
        
    # save metrics
    trainer.load_best()
    metrics = []
    thisConfmat = []
    callbacks = (
            trainer.dataset.train_batch,
            trainer.dataset.val_batch,
            trainer.dataset.test_batch,
        )
    for _, (name, callable) in enumerate(zip(["Train", "Val", "Test"], callbacks)):
        x, y = callable(True, to_cuda=True)
        if trainer.is_regression:
            _, m = trainer.eval(x, y.type(torch.FloatTensor).to(trainer.device))
            metrics.append(m)
            thisConfmat.append(np.array(trainer.confmat.compute().cpu(), dtype = str))
        else:
            _, m = trainer.eval(x, y.type(torch.LongTensor).to(trainer.device))
            metrics.append(m)
            thisConfmat.append(np.array(trainer.confmat.compute().cpu(), dtype = str))

    np.save(thisPath.joinpath('metrics.npy'), metrics)

    with open(thisPath.joinpath("metrics.csv"), mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Accuracy", "CohenKappa", "myFScore"])
        writer.writeheader()
        writer.writerows(metrics)


    # save figure
    fig, _ = trainer.test("best", show_confmat=False)
    fig.savefig(thisPath.joinpath('confusionMatrix_best.pdf'))

    # save confusion matrix
    thisConfmat = np.array(thisConfmat).reshape(12,4)
    labels = np.array(trainer.dataset.labels, dtype = str)
    labels3=np.tile(labels, 3)
    writeConfmat = np.column_stack((labels3,thisConfmat))
    writeConfmat = np.vstack((np.concatenate(([' '], labels)), writeConfmat))
    np.savetxt(thisPath.joinpath('confusionMatrix.csv'), writeConfmat, delimiter=",", fmt='%s')

 

    # print regression figure
    allMetrics = []
    if trainer.is_regression:
        for _, (name, callable) in enumerate(zip(["Train", "Val", "Test"], callbacks)):
            x, y = callable(True, to_cuda=True)   
            all_classes = np.unique(y.cpu().numpy())

            # best model
            trainer.load_best()     
            y_pred = trainer.model(torch.Tensor(x)).squeeze()
            f = myFScore()
            f.update(preds = torch.tensor(y_pred), target = torch.tensor(y))
            fig = plt.figure()
            for cls in all_classes:
                _ = plt.hist(y_pred[y == cls].cpu().detach().numpy(), 50, alpha=0.5, label=f"Class {cls}")
            _ = fig.suptitle('Best Model - ' + name + " - Fscore = " + str(f.compute().numpy())) 
            plt.savefig(thisPath.joinpath('predictionDistribution_best_' + name + '.pdf'))
            allMetrics.append(f.compute().numpy())

            trainer.load_last()
            y_pred = trainer.model(torch.Tensor(x)).squeeze()
            f = myFScore()
            f.update(preds = torch.tensor(y_pred), target = torch.tensor(y))
            fig = plt.figure()
            for cls in all_classes:
                _ = plt.hist(y_pred[y == cls].cpu().detach().numpy(), 50, alpha=0.5, label=f"Class {cls}")
            _ = fig.suptitle('Last Model - ' + name + " - Fscore = " + str(f.compute().numpy())) 
            plt.savefig(thisPath.joinpath('predictionDistribution_last_' + name + '.pdf'))

        metric_train = allMetrics[0]
        metric_val = allMetrics[1]
        metric_test = allMetrics[2]
    else:
        metric_train = metrics[0]['Accuracy']
        metric_val = metrics[1]['Accuracy']
        metric_test = metrics[2]['Accuracy']

    return metric_train, metric_val, metric_test