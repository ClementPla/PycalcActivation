from math import e
import numpy as np
from sympy import Q
import torch
import csv
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
from PycalcAct.dataset_v2 import Dataset
from PycalcAct.model import (
    MixedFCTemporalModel)
from PycalcAct.trainer_v2 import Trainer
from PycalcAct.calculateCustomAccuracy import calculateCustomAccuracy
from pathlib import Path
from PycalcAct.myCustomCriterion import myCustomCriterion, myFScore
_ = torch.manual_seed(1234)
from socket import gethostname
import hashlib
import random
import string
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
from scipy.interpolate import interp1d

def get_safe_folder_name(config):
    # Convert config to string and hash it
    config_str = str(sorted(config.items()))
    # Generate a random 8-character string
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    # Combine config string with random component
    combined_str = config_str + random_suffix

    hash_id = hashlib.md5(combined_str.encode()).hexdigest()
    return f"run_{hash_id}"


def getPath(is_regression, sweep_id):
    if gethostname() == 'HM_Lab':
        dataFolder = Path("D:/Ca2-Analysis_McGill/prediction/agAffinity/datasets/dataset_mcgill")
        saveFolder = Path("D:/sebastien/PycalcActivation/models/round3")
    elif gethostname() == 'HMR_LYMPH':
        dataFolder = Path("D:/SebastienThis/CalciumPredictions/Ca2-Analysis_McGill/prediction/agAffinity/datasets/dataset_mcgill")
        saveFolder = Path("D:/SebastienThis/CalciumPredictions/PycalcActivation/models/round3")
    elif gethostname() == 'HMR-BLOOD':
        dataFolder = Path("D:/Sebastien/Ca2-Analysis_McGill/prediction/agAffinity/datasets/dataset_mcgill")
        saveFolder = Path("D:/sebastien/PycalcActivation/models/round3")
    if is_regression:
        saveFolder = saveFolder.joinpath("regressor")
    else:
        saveFolder = saveFolder.joinpath("classifier")
    saveFolder = saveFolder.joinpath("sweep_"+ sweep_id)
    
    return dataFolder, saveFolder

def setupTrainer(config, is_regression, sweep_id, model_unique_name = None, EC50_path = "D:\sebastien\PycalcActivation\EC50.csv"):
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
    initial_lr=config['initial_lr']
    weight_decay=config['weight_decay']
    batch_size = config['batch_size']
    store_best = config['store_best']
    myLoss = config['myLoss']
    customFilter = None

    dataFolder, saveFolder = getPath(is_regression, sweep_id)   
    
     # create model folder
    if model_unique_name is None:
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

    if whichDisplacement == "displacement":
        csv_pos_path = dataFolder.joinpath("position.csv")
        position_to_displacement = True
    elif whichDisplacement == "xyPosition":
        csv_pos_path = dataFolder.joinpath("position.csv")
        position_to_displacement = False            
    else:
        csv_pos_path = None
        position_to_displacement = False

    EC50 = pd.read_csv(EC50_path, index_col=None , header=None)
    EC50 = {
        (row.iloc[0]): row.iloc[1]
        for _, (_, row) in enumerate(EC50.iterrows())
    }
    
    dataset = Dataset(
        csv_path = csv_path,
        csv_pos_path=csv_pos_path,  # Optional
        position_to_displacement=position_to_displacement,
        remove_mean=remove_mean,
        replace_nan_by_min=replace_nan_by_min,
        customFilter = customFilter, 
        is_regression=is_regression, #True,
        EC50 = EC50,
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
    # Setup training
    
    criterion = None
    D = torch.tensor([  [1,2,3,4], 
                        [3,1,4,2],
                        [4,3,1,2],
                        [3,2,3,1]]).to("cuda")
    
    if customLoss:
        criterion = myCustomCriterion(weight = dataset.weights, device = "cuda", D = D)

    if store_best == None:
        if is_regression:
            store_best = 'myFScore'
        else:
            store_best = 'Accuracy'

    trainer = Trainer(
        dataset,
        model,
        device="cuda",
        batch_size=batch_size,
        criterion= criterion,
        store_best= store_best,
        use_class_weights = weighted,
        is_regression = is_regression,
        regression_bounds_y = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
        regression_bounds_y_pred = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
        initial_lr=initial_lr,
        weight_decay=weight_decay,
        loss = myLoss,
    ) 

    return model_unique_name, trainer


def save_model_perf(trainer, model_unique_name, sweep_id, save = True):
    # save model
    _, saveFolder = getPath(trainer.is_regression, sweep_id)   
    thisPath = saveFolder.joinpath(model_unique_name)
    thisModelName = thisPath.joinpath("savedModel.pt")

    if save:
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
    if save:
        np.save(thisPath.joinpath('metrics.npy'), metrics)

    with open(thisPath.joinpath("metrics.csv"), mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=trainer.metrics.keys())
        writer.writeheader()
        writer.writerows(metrics)


    # save figure
    fig, _ = trainer.test("best", show_confmat=False)
    if save:
        fig.savefig(thisPath.joinpath('confusionMatrix_best.pdf'))

    # save confusion matrix
    thisConfmat = np.array(thisConfmat).reshape(12,4)
    labels = np.array(trainer.dataset.labels("OTI"), dtype = str)
    labels3=np.tile(labels, 3)
    writeConfmat = np.column_stack((labels3,thisConfmat))
    writeConfmat = np.vstack((np.concatenate(([' '], labels)), writeConfmat))
    if save:
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
            f.update(preds = y_pred.detach().clone().cpu(), target = y.detach().clone().cpu())
            fig = plt.figure()
            for cls in all_classes:
                _ = plt.hist(y_pred[y == cls].cpu().detach().numpy(), 50, alpha=0.5, label=f"Class {cls}", density = True)
            _ = fig.suptitle('Best Model - ' + name + " - Fscore = " + str(f.compute().numpy())) 
            _ = plt.legend()
            if save:
                plt.savefig(thisPath.joinpath('predictionDistribution_best_' + name + '.pdf'))
            allMetrics.append(f.compute().numpy())

            trainer.load_last()
            y_pred = trainer.model(torch.Tensor(x)).squeeze()
            f = myFScore()
            f.update(preds = y_pred.detach().clone().cpu(), target = y.detach().clone().cpu())
            fig = plt.figure()
            for cls in all_classes:
                _ = plt.hist(y_pred[y == cls].cpu().detach().numpy(), 50, alpha=0.5, label=f"Class {cls}", density = True)
            _ = fig.suptitle('Last Model - ' + name + " - Fscore = " + str(f.compute().numpy())) 
            _ = plt.legend()
            if save:
                plt.savefig(thisPath.joinpath('predictionDistribution_last_' + name + '.pdf'))

        metric_train = allMetrics[0]
        metric_val = allMetrics[1]
        metric_test = allMetrics[2]
    else:
        metric_train = metrics[0]['Accuracy']
        metric_val = metrics[1]['Accuracy']
        metric_test = metrics[2]['Accuracy']

    return metric_train, metric_val, metric_test, allMetrics, metrics


def save_model_generalizability(trainer, model_unique_name, sweep_id, save = True, EC50_path = "D:\sebastien\PycalcActivation\EC50.csv"):
    # save model
    _, saveFolder = getPath(trainer.is_regression, sweep_id)   
    thisPath = saveFolder.joinpath(model_unique_name)

    thisPath = saveFolder.joinpath(model_unique_name)
    trainer.load_best()
    metrics = {}
    EC50 = pd.read_csv(EC50_path, index_col=None , header=None)
    EC50 = {
        row.iloc[0] : row.iloc[1]
        for _, (_, row) in enumerate(EC50.iterrows())
    }
    this_dict = EC50
    # cost_matrix = {
    #     "OTI" : np.array([[1.0,0.6,0.3,0],[0.6,1.0,0.6,0.3],[0.3,0.6,1.0,0.6], [0,0.3,0.6,1.0]]),
    #     "SL" : np.array([[1.0,0.6,0.3,0],[0.6,1.0,0.6,0.3],[0.3,0.6,1.0,0.6], [0,0.3,0.6,1.0]]),
    #     "P14" : np.array([[0,0.3,1.0,0.6],[0,0.3,1.0,0.6],[1.0,0.6,0,0.3]]),
    #     "OT3" : np.array([[0.6,1.0,0,0.3],[0.6,1.0,0,0.3]]),
    #     "conc" : np.array([[1.0,0.6,0.3,0],[1.0,0.6,0.3,0],[1.0,0.6,0.3,0],[1.0,0.6,0.3,0]]),
    # }

    accuracy_matrix = {
        "OTI" : np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0], [0,0,0,1]]),
        "SL" : np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0], [0,0,0,1]]),
        "P14" : np.array([[0,0,1,0],[0,0,1,0],[1,0,0,0]]),
        "OT3" : np.array([[0,1,0,0],[0,1,0,0]]),
        "conc" : np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]),
    }

    encoder = LabelEncoder()
    for k in trainer.dataset.f.all_data.keys():
        x = trainer.dataset.f.all_data[k]["x"]
        y = trainer.dataset.f.all_data[k]["y"]
        classes = trainer.dataset.f.all_data[k]["classes"]
        classes_encoded = encoder.fit_transform(classes)
        x,y = trainer.dataset.batch_from_data(x, y, time_first=True, to_cuda=True)
        apl_classes = np.unique(classes_encoded)
        classes_decoder = {code: label for label, code in zip(encoder.classes_, range(len(encoder.classes_)))}

        if trainer.is_regression:
            # make prediction
            y_pred = torch.Tensor()
            batch_size = trainer.batch_size
            for i in range(0, len(x), batch_size):
                x_batch = x[i : i + batch_size]
                with torch.no_grad():
                    y_pred_batch = trainer.model(x_batch)
                    y_pred = torch.cat((y_pred, y_pred_batch.cpu()), dim=0)
            y_pred = y_pred.squeeze().numpy()   

            # distance metrics
            this_metric = np.mean(np.abs(y.cpu().numpy() - y_pred), axis = 0)
            metrics.update({"distance_" + k : this_metric}) # need to min (distance to target)

            # fScore metrics
            f = myFScore()
            f.update(preds = torch.Tensor(y_pred), target = torch.Tensor(classes_encoded))
            metrics.update({"fScore_" + k : f.compute().numpy()}) # need to max (FScore)

            # relative weighted distance
            original_refs = [this_dict[v] for v in ["N4", "Q4", "Q4H7", "T4"]]
            xval, yval = trainer.dataset.test_batch(True)
            y_pred_val = trainer.predict(xval).cpu().squeeze().numpy()
            predicted_refs = [np.mean(y_pred_val[yval.cpu().numpy() == v]) for v in original_refs]
            interp_func = interp1d(original_refs, predicted_refs, kind='linear', fill_value="extrapolate")
            y_interp = interp_func(y.cpu().numpy())
            
            this_interp_distance = np.mean(np.abs(y_interp - y_pred), axis = 0)
            metrics.update({"weighted_distance_" + k: this_interp_distance}) 

            # plot distribution
            fig = plt.figure()
            for cls in apl_classes:
                _ = plt.hist(y_pred[classes_encoded == cls], 100, alpha=0.5, label=f"Class {classes_decoder[cls]}", density=True)
                _ = plt.legend()
                this_x = interp_func(this_dict[classes_decoder[cls]])
                _ = plt.plot([this_x, this_x], [0,1])
            _ = fig.suptitle('Best Model - ' + k + " - Fscore = " + str(f.compute().numpy())) 
            _ = plt.legend()
            if save:
                plt.savefig(thisPath.joinpath('predictionDistribution_' + k + '.pdf'))

        else:
            # model predicion on this dataset
            y_pred = trainer.predict(x)
            predicted_class = y_pred.argmax(dim = 1)

            # generate cost matrix 
            GT = np.array([this_dict[v] for v in trainer.dataset.f.all_data["OTI"]["mapping"].values()])
            pred = np.array([this_dict[v] for v in trainer.dataset.f.all_data[k]["mapping"].values()])
            this_distance_matrix = cdist(pred.reshape(-1,1), GT.reshape(-1,1), metric='euclidean')
            this_accuracy_matrix = accuracy_matrix[k]
       
            # calculate distance metric
            this_distance = np.mean(this_distance_matrix[y.cpu(),predicted_class.cpu()])
            metrics.update({"distance_" + k : this_distance})   # need to min (distance to target)

            # calculate accuracy metric
            this_accuracy = np.mean(this_accuracy_matrix[y.cpu(),predicted_class.cpu()])
            metrics.update({"accuracy_" + k : this_accuracy})  # need to max (distance to target)

            # print and write all "confusion matrices"
            n_pred_classes = len(pred)
            n_GT_classes = len(GT)
            this_confmat = np.zeros((n_pred_classes, n_GT_classes))
            for r in range(0, n_pred_classes):
                for c in range(0,n_GT_classes):
                    this_confmat[r,c] = sum((y == r) & (predicted_class == c))

            # zScore
            this_confmat_norm = (this_confmat - np.mean(this_confmat, axis = 1).reshape(-1,1))/np.std(this_confmat, axis = 1).reshape(-1,1)

            # plot
            fig, ax = plt.subplots()    
            _ = ax.imshow(this_confmat_norm, cmap='magma') 
            cmap_reversed = matplotlib.colormaps.get_cmap('magma_r')
            for i in range(this_confmat.shape[0]):
                for j in range(this_confmat.shape[1]):
                    text = ax.text(j, i, this_confmat[i, j],
                                ha="center", va="center", color = cmap_reversed(this_confmat_norm[i, j]))
            _ = ax.set_xticks(np.arange(this_confmat.shape[1]))
            _ = ax.set_yticks(np.arange(this_confmat.shape[0]))
            _ = ax.set_xticklabels([v for v in trainer.dataset.f.all_data["OTI"]["mapping"].values()]) 
            _ = ax.set_yticklabels([v for v in classes_decoder.values()])
            _ = ax.set_xlabel("Predicted Label")
            _ = ax.set_ylabel("True Label")
            _ = fig.suptitle('Best Model - ' + k + " - Custom metric = " +str(metrics["distance_"+k])) 
            if save:
                plt.savefig(thisPath.joinpath('confusionMatrix_' + k + '.pdf'))

            # write
            labels_pred = np.array(trainer.dataset.labels("OTI"), dtype = str)
            labels_GT = np.array(trainer.dataset.labels(k), dtype = str)
            write_confmat = np.row_stack((labels_pred,this_confmat))
            write_confmat = np.column_stack((np.concatenate(([' '], labels_GT)), write_confmat))
            if save:
                np.savetxt(thisPath.joinpath('confusionMatrix_' + k + '.csv'), write_confmat, delimiter=",", fmt='%s')
            write_confmat = np.row_stack((labels_pred,this_confmat_norm))
            write_confmat = np.column_stack((np.concatenate(([' '], labels_GT)), write_confmat))
            if save:
                np.savetxt(thisPath.joinpath('confusionMatrixZScore_' + k + '.csv'), write_confmat, delimiter=",", fmt='%s')

    return metrics