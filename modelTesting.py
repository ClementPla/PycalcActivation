

from cProfile import label
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

conditionPath = 'D:/SebastienThis/CalciumPredictions/PycalcActivation/trainingOptions_round2.csv'
dataFolder = "D:/SebastienThis/CalciumPredictions/Ca2-Analysis_McGill/prediction/agAffinity/datasets/testingData/"
modelsFolder = "D:/SebastienThis/CalciumPredictions/Ca2-Analysis_McGill/prediction/agAffinity/models/"

myCondition = pd.read_csv(conditionPath, header=None)
myLegend = myCondition.iloc[0,:]
myCondition = myCondition.iloc[1:,:]
myCondition = [myCondition.iloc[i,:].to_numpy() for i in range(0, myCondition.shape[0])] 

for cdt in myCondition: #[209:]
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

    done = np.load(Path(modelsFolder) / 'done.npy')
    if np.isin(done,modelNum).any():
        
        # print which model we are testing
        print(f"NumberModel = {modelNum}/{len(myCondition)}, Dataset = {whichDataset}, Displacement = {whichDisplacement},  Displacement as XY = {xyDisplacement}, Replace Nan by mean = {whichReplaceNan}, Remove Mean = {whichRemoveMean}, FFT = {whichFFT} \n #RNN = {numRNN}, size RNN = {sizeRNN}, bidirectional = {bidir}, #FC = {numFC}, size FC = {sizeFC}, dropout = {dropout}")
        
        # create model folder
        thisPath = Path(modelsFolder + modelNum)
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
            customFilter = customFilter,
            forEval = False
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
            n_classes=4,
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
        
        # load best state of model
        thisModelName = thisPath / "savedModel.pt"
        checkpoint = torch.load(Path(thisModelName), weights_only=True)
        model.load_state_dict(checkpoint["best"])
        model.to("cuda")
        del checkpoint

        # Make predictions
        model.eval()
        batch_size = 5000
        x, y = dataset.train_batch(True, to_cuda=True)
        y_pred = torch.zeros(len(x), 4).to("cuda")
            # separate into batch
        for i in range(0, len(x), batch_size):
            x_batch = x[i : i + batch_size]

            with torch.autocast("cuda"): #torch.cuda.amp.autocast()
                y_pred[i : i + batch_size] = model(x_batch)


        # y_pred = model(x)
        y_pred_cat = torch.argmax(y_pred, dim=1)
        
        myHeatmap = np.zeros((len(y.unique()), len(y_pred_cat.unique())))        # calculate Heatmap
        for i in range(len(y.unique())):
            for j in range(len(y_pred_cat.unique())):
                myHeatmap[i,j] = torch.sum((y == i) & (y_pred_cat == j)).cpu().detach().numpy()
        myHeatmap = myHeatmap[:,[0,1,3,2]]
        label_order = np.array([7,8,9,4,5,6,14,15,17,16,10,11,12,13,2,3,0,1]);
        myHeatmap = myHeatmap[label_order,:]

        # Visulalize predictions    
        # calculate Z-Score on heatmap 
        myHeatmap_norm = (myHeatmap - np.mean(myHeatmap, axis=1)[:,np.newaxis]) / np.std(myHeatmap, axis=1)[:,np.newaxis]
        np.savetxt(thisPath / 'testingHeatmap.csv', myHeatmap, delimiter=",", fmt='%s')

        # plot figure
        fig, ax = plt.subplots()
        ax.set_xticks(list(range(len(y_pred_cat.unique()))) ,["N4", "Q4", "T4", "Q4H7"] )
        ax.set_yticks(list(range(len(y.unique()))) ,np.array(dataset.labels)[label_order])
        ax.imshow(myHeatmap_norm, cmap='RdYlBu', interpolation='nearest')
        for i in range(len(y.unique())):
            for j in range(len(y_pred_cat.unique())):
                plt.text(j, i, str(int(myHeatmap[i,j])), ha='center', va='center', color='black', size = 'x-small')
        # plt.show()

        # calculate custom metric
        OT1_EC50 = np.log10(np.array([1.4e-17, 3.9e-12, 8.43e-10, 4.67e-9]))
        testing_EC50 =  np.log10(np.array([1.4e-17, 1.4e-17,1.4e-17,1.4e-17,1.4e-17,1.4e-17,3.9e-12, 3.9e-12, 8.43e-10, 4.67e-9, 5.474e-14, 5.474e-14, 2.508e-16,2.508e-16, 8.995e-13,  8.995e-13, 3.16e-9 , 9.26e-9]))

        thisCorrection = np.abs(OT1_EC50-testing_EC50[:,None])
        thisMetric = myHeatmap*thisCorrection
        thisMetric = thisMetric/np.sum(myHeatmap)
        thisWeigths = dataset.weights.cpu().detach().numpy()
        thisWeigths = thisWeigths[label_order]
        thisMetric = thisMetric*thisWeigths[:,None]
        thisMetric = np.sum(thisMetric, axis = None)

        # save metric to file
        myFile = pd.read_csv(conditionPath, header=None)
        myFile.loc[myFile.iloc[:,0] == modelNum, myLegend == "Testing_Metric"] = f"{thisMetric:.4f}"
        myFile.to_csv(conditionPath,sep = ",", header = False, index = False)

        # save figure
        fig.savefig(thisPath / 'testingMatrix.pdf')
        plt.close(fig)

        del model
        del dataset
        del y_pred_cat
        with torch.no_grad():
            torch.cuda.empty_cache()



        