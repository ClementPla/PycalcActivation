import numpy as np
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



# conditionPath = 'D:/SebastienThis/CalciumPredictions/PycalcActivation/trainingOptions_round2.csv'
# dataFolder = "D:/SebastienThis/CalciumPredictions/Ca2-Analysis_McGill/prediction/agAffinity/datasets/trainingData/"
# saveFolder = "D:/SebastienThis/CalciumPredictions/Ca2-Analysis_McGill/prediction/agAffinity/models/"

# conditionPath = '//Hmr_lymph/d/SebastienThis/CalciumPredictions/PycalcActivation/trainingOptions_round2.csv'
# dataFolder = "//Hmr_lymph/d/SebastienThis/CalciumPredictions/Ca2-Analysis_McGill/prediction/agAffinity/datasets/trainingData/"
# saveFolder = "//Hmr_lymph/d/SebastienThis/CalciumPredictions/Ca2-Analysis_McGill/prediction/agAffinity/models/"

conditionPath = 'D:/sebastien/PycalcActivation/trainingOptions_round2.csv'
dataFolder = "D:/Ca2-Analysis_McGill/prediction/agAffinity/datasets/trainingData/"
saveFolder = "D:/Ca2-Analysis_McGill/prediction/agAffinity/models/"


myCondition = pd.read_csv(conditionPath, header=None)
myLegend = myCondition.iloc[0,:]
myCondition = myCondition.iloc[1:,:]
myCondition = [myCondition.iloc[i,:].to_numpy() for i in range(0, myCondition.shape[0])] 

done = np.array([])



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
    customLoss =  int(cdt[12]) == 1
    xyDisplacement = int(cdt[13]) == 1


    

    done = np.load(Path(saveFolder) / 'done.npy')
    if np.isin(done,modelNum).any():
        print("Already trained")
    else:

        print(f"NumberModel = {modelNum}/{len(myCondition)}, Dataset = {whichDataset}, Displacement = {whichDisplacement},  Displacement as XY = {xyDisplacement}, Replace Nan by mean = {whichReplaceNan}, Remove Mean = {whichRemoveMean}, FFT = {whichFFT} \n #RNN = {numRNN}, size RNN = {sizeRNN}, bidirectional = {bidir}, #FC = {numFC}, size FC = {sizeFC}, dropout = {dropout}")

        # create model folder
        thisPath = Path(saveFolder + modelNum)
        thisPath.mkdir(parents=True, exist_ok=True)
            
        # Setup Dataset
        customFilter = None
        match whichDataset:
            case "ratio":
                csv_path=[dataFolder + "legend.csv", dataFolder+"calciumRatio.csv"]          
            case "ratioNorm":
                csv_path=[dataFolder + "legend.csv", dataFolder+"calciumRatio_normalized.csv"] 
            case "indiv":
                csv_path=[dataFolder + "legend.csv", dataFolder+"calciumFree.csv", dataFolder+"calciumBound.csv"] 
            case _:
                csv_path=[dataFolder + "legend.csv", dataFolder+"calciumRatio_normalized.csv"] 
                customFilter = cdt[1]

        if whichDisplacement:
            csv_pos_path = dataFolder + "position.csv"
            if xyDisplacement:
                position_to_displacement = False
            else:
                position_to_displacement = True
        else:
            csv_pos_path = None
            position_to_displacement = False

        replace_nan_by_min = True if whichReplaceNan else False
        remove_mean = True if whichRemoveMean else False
        

        dataset = Dataset(
            csv_path = csv_path,
            csv_pos_path=csv_pos_path,  # Optional
            position_to_displacement=position_to_displacement,
            remove_mean=remove_mean,
            replace_nan_by_min=replace_nan_by_min,
            customFilter = customFilter
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
            criterion= criterion
        )  # You can pass your own optimizer, criterion, learning rate, weight decay and learning rate scheduler.

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

        # save confusion matrix

        thisConfmat =  trainer.confmat.compute()
        thisConfmat = np.array(thisConfmat.cpu(), dtype = str)   
        labels=np.array(trainer.dataset.labels, dtype = str)
        writeConfmat = np.column_stack((labels,thisConfmat))
        writeConfmat = np.row_stack((np.concatenate(([' '], labels)), writeConfmat))
        np.savetxt(thisPath / 'confusionMatrix.csv', writeConfmat, delimiter=",", fmt='%s')

        # update allCondition file.
        accuracy_N4, accuracy_Q4, accuracy_T4, accuracy_Q4H7, customAcc = calculateCustomAccuracy(thisConfmat)
        myFile = pd.read_csv(conditionPath, header=None)
        myFile.loc[myFile.iloc[:,0] == modelNum, myLegend == "Accuracy"] = f"{metrics['Accuracy']:.4f}"
        myFile.loc[myFile.iloc[:,0] == modelNum,myLegend == "N4_Acc"] = f"{accuracy_N4:.4f}"
        myFile.loc[myFile.iloc[:,0] == modelNum,myLegend == "Q4_Acc"] = f"{accuracy_Q4:.4f}"
        myFile.loc[myFile.iloc[:,0] == modelNum,myLegend == "T4_Acc"] = f"{accuracy_T4:.4f}"
        myFile.loc[myFile.iloc[:,0] == modelNum,myLegend == "Q4H7_Acc"] = f"{accuracy_Q4H7:.4f}"
        myFile.loc[myFile.iloc[:,0] == modelNum,myLegend == "Custom_Acc"] = f"{customAcc:.4f}"
        myFile.to_csv(conditionPath,sep = ",", header = False, index = False)
        # save figure
        fig.savefig(thisPath / 'confusionMatrix.pdf')

        done = np.append(done, modelNum)
        np.save(Path(saveFolder) / 'done.npy', done)

        # print current metrics
        print(f"Model {modelNum} - Accuracy: {metrics['Accuracy']:.4f} \n\n\n")



##
allAccuracy = np.array([])
for i in range(0,len(myCondition)):
    temp = pd.read_csv(Path(saveFolder)/str(i+1)/"MetricsValue.csv", header=None)
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
