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
from PycalcAct.myCustomCriterion import myCustomCriterion
_ = torch.manual_seed(1234)
from scipy.stats import f_oneway
from socket import gethostname


if gethostname() == 'HM_Lab':
    conditionPath = Path("D:/sebastien/PycalcActivation/trainingOptions_round3.csv")
    dataFolder = Path("D:/Ca2-Analysis_McGill/prediction/agAffinity/datasets/dataset_mcgill")
    saveFolder = Path("D:/sebastien/PycalcActivation/models/round3")
elif gethostname() == 'Hmr_lymph':
    conditionPath = Path('D:/SebastienThis/CalciumPredictions/PycalcActivation/trainingOptions_round2.csv')
    dataFolder = Path("D:/SebastienThis/CalciumPredictions/Ca2-Analysis_McGill/prediction/agAffinity/datasets/trainingData")
    saveFolder = Path("D:/SebastienThis/CalciumPredictions/Ca2-Analysis_McGill/prediction/agAffinity/models")
elif gethostname() == 'HMR_BLOOD':
    conditionPath = Path('D:/Sebastien/PycalcActivation/trainingOptions_round3.csv')
    dataFolder = Path("D:/Sebastien/Ca2-Analysis_McGill/prediction/agAffinity/datasets/dataset_mcgill")
    saveFolder = Path("D:/sebastien/PycalcActivation/models/round3")



myCondition = pd.read_csv(conditionPath, header=None)
myLegend = myCondition.iloc[0,:]
myCondition = myCondition.iloc[1:,:]
myCondition = [myCondition.iloc[i,:].to_numpy() for i in range(0, myCondition.shape[0])] 

for cdt in myCondition:
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
        done = np.array()
        
    if np.isin(done,modelNum).any():
        print("Already trained")
    else:

        print(f"NumberModel = {modelNum}/{len(myCondition)}, Dataset = {whichDataset}, Displacement = {whichDisplacement},  Displacement as XY = {xyDisplacement}, Replace Nan by mean = {whichReplaceNan}, Remove Mean = {whichRemoveMean}, FFT = {whichFFT} \n #RNN = {numRNN}, size RNN = {sizeRNN}, bidirectional = {bidir}, #FC = {numFC}, size FC = {sizeFC}, dropout = {dropout}")

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
        
        if is_regression:
            regression_bounds_y = [-12,-10,-9]
            regression_bounds_y_pred = [-12,-10,-9]
            weighted = False
        else:
            regression_bounds_y = None
            regression_bounds_y_pred = None

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
            is_regression=False, #True,
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
            n_classes=dataset.n_classes,
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
            batch_size=2000,
            criterion= criterion,
            use_class_weights= weighted,
            regression_bounds_y = regression_bounds_y,
            regression_bounds_y_pred = regression_bounds_y_pred
        )  
        # train model
        trainer.train(n_epochs)
        # trainer.test("best")

        # save model
        thisModelName = thisPath / "savedModel.pt"
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
        fig, metrics = trainer.test("best", show_confmat=False)  # Testing on the best model (in term of validation accuracy)
        np.save(thisPath / 'metrics.npy', metrics)
        
        narr = np.array([metrics['Accuracy'].tolist(), metrics['CohenKappa'].tolist()])
        np.savetxt(thisPath / 'MetricsValue.csv', narr, delimiter=",")

        # save figure
        fig.savefig(thisPath / 'confusionMatrix.pdf')

        # save confusion matrix
        thisConfmat =  trainer.confmat.compute()
        thisConfmat = np.array(thisConfmat.cpu(), dtype = str)   
        labels=np.array(trainer.dataset.labels, dtype = str)
        writeConfmat = np.column_stack((labels,thisConfmat))
        writeConfmat = np.row_stack((np.concatenate(([' '], labels)), writeConfmat))
        np.savetxt(thisPath / 'confusionMatrix.csv', writeConfmat, delimiter=",", fmt='%s')

        # update allCondition file.
        if not is_regression:
            accuracy_N4, accuracy_Q4, accuracy_T4, accuracy_Q4H7, customAcc = calculateCustomAccuracy(thisConfmat)
            myFile = pd.read_csv(conditionPath, header=None)
            myFile.loc[myFile.iloc[:,0] == modelNum, myLegend == "Accuracy"] = f"{metrics['Accuracy']:.4f}"
            myFile.loc[myFile.iloc[:,0] == modelNum,myLegend == "N4_Acc"] = f"{accuracy_N4:.4f}"
            myFile.loc[myFile.iloc[:,0] == modelNum,myLegend == "Q4_Acc"] = f"{accuracy_Q4:.4f}"
            myFile.loc[myFile.iloc[:,0] == modelNum,myLegend == "T4_Acc"] = f"{accuracy_T4:.4f}"
            myFile.loc[myFile.iloc[:,0] == modelNum,myLegend == "Q4H7_Acc"] = f"{accuracy_Q4H7:.4f}"
            myFile.loc[myFile.iloc[:,0] == modelNum,myLegend == "Custom_Acc"] = f"{customAcc:.4f}"
            myFile.to_csv(conditionPath,sep = ",", header = False, index = False)
        else:
            
            myFile = pd.read_csv(conditionPath, header=None)
            myFile.loc[myFile.iloc[:,0] == modelNum, myLegend == "Accuracy"] = f"{metrics['Accuracy']:.4f}"
            myFile.loc[myFile.iloc[:,0] == modelNum,myLegend == "N4_Acc"] = f"{metrics_last['Accuracy']:.4f}"
            myFile.loc[myFile.iloc[:,0] == modelNum,myLegend == "Q4_Acc"] = f"{metrics['myFScore']:.4f}"
            myFile.loc[myFile.iloc[:,0] == modelNum,myLegend == "T4_Acc"] = f"{metrics_last['myFScore']:.4f}"
            myFile.to_csv(conditionPath,sep = ",", header = False, index = False)

            x_test, y_test = trainer.dataset.test_batch(True, to_cuda=True)
            x_train, y_train = trainer.dataset.train_batch(True, to_cuda=True)
            
            all_classes = np.unique(y_train.cpu().numpy())

            trainer.load_best()
            y_pred_train = trainer.model(torch.Tensor(x_train))
            y_pred_test = trainer.model(torch.Tensor(x_test))
            fig, [ax, ax1] = plt.subplots(1,2)
            for cls in all_classes:
                ax.hist(y_pred_train[y_train == cls].cpu().detach().numpy(), 50, alpha=0.5, label=f"Class {cls}")
                ax1.hist(y_pred_test[y_test == cls].cpu().detach().numpy(), 50, alpha=0.5, label=f"Class {cls}") 
            ax.title.set_text('train dataset')
            ax1.title.set_text('test dataset')
            fig.suptitle('Best Model') 
            plt.savefig(thisPath / 'predictionDistribution_best.pdf')
            trainer.load_last()
            y_pred_train = trainer.model(torch.Tensor(x_train))
            y_pred_test = trainer.model(torch.Tensor(x_test))
            fig, [ax, ax1] = plt.subplots(1,2)
            for cls in all_classes:
                ax.hist(y_pred_train[y_train == cls].cpu().detach().numpy(), 50, alpha=0.5, label=f"Class {cls}")
                ax1.hist(y_pred_test[y_test == cls].cpu().detach().numpy(), 50, alpha=0.5, label=f"Class {cls}") 
            ax.title.set_text("train dataset")
            ax1.title.set_text("test dataset")
            fig.suptitle('Last Model') 
            plt.savefig(thisPath / 'predictionDistribution_last.pdf')
            
            
        done = np.append(done, modelNum)
        np.save(saveFolder.joinpath('done.npy'), done)

        # print current metrics
        print(f"Model {modelNum} - Accuracy: {metrics['Accuracy']:.4f} \n\n\n")



##
allAccuracy = np.array([])
for i in range(0,len(myCondition)):
    temp = pd.read_csv(saveFolder.joinpath(str(i+1),"MetricsValue.csv"), header=None)
    allAccuracy = np.append(allAccuracy, temp.iloc[0])
    
bestModel = np.argmax(allAccuracy)
bestAccuracy = np.max(allAccuracy)
# np.savetxt('allAccuracy.csv', allAccuracy, delimiter=",")
print(f"Best Model = {bestModel+1} with Accuracy = {bestAccuracy}")
print(f"NumberModel = {myCondition[bestModel][0]}, Dataset = {myCondition[bestModel][1]}, Displacement = {myCondition[bestModel][2]}, Replace Nan by min = {myCondition[bestModel][3]}, Remove Mean = {myCondition[bestModel][4]}, FFT = {myCondition[bestModel][5]}\n #RNN = {myCondition[bestModel][6]}, size RNN = {myCondition[bestModel][7]}, bidirectional = {myCondition[bestModel][8]}, #FC = {myCondition[bestModel][9]}, size FC = {myCondition[bestModel][10]}, dropout = {myCondition[bestModel][11]} \n\n")

##
allCustomAccuracy = pd.read_csv(conditionPath)
allCustomAccuracy = allCustomAccuracy.iloc[:, np.argwhere(myLegend == "Custom_Acc").item()].to_numpy()
bestModel = np.argmax(allCustomAccuracy)
bestAccuracy = np.max(allCustomAccuracy)

print(f"Best Model = {bestModel+1} with Accuracy = {bestAccuracy}")
print(f"NumberModel = {myCondition[bestModel][0]}, Dataset = {myCondition[bestModel][1]}, Displacement = {myCondition[bestModel][2]}, Replace Nan by min = {myCondition[bestModel][3]}, Remove Mean = {myCondition[bestModel][4]}, FFT = {myCondition[bestModel][5]}\n #RNN = {myCondition[bestModel][6]}, size RNN = {myCondition[bestModel][7]}, bidirectional = {myCondition[bestModel][8]}, #FC = {myCondition[bestModel][9]}, size FC = {myCondition[bestModel][10]}, dropout = {myCondition[bestModel][11]} ")
