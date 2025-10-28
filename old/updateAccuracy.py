import pandas as pd
import numpy as np
from pathlib import Path
from PycalcAct.calculateCustomAccuracy import *

conditionPath = 'D:/SebastienThis/CalciumPredictions/PycalcActivation/trainingOptions.csv'
saveFolder = "D:/SebastienThis/CalciumPredictions/Ca2-Analysis_McGill/prediction/agAffinity/models/"

myFile = pd.read_csv(conditionPath, header=None)

for i in range(myFile.shape[0]-1):
    modelName = myFile.iloc[i+1, 0]
    thisPath = Path(saveFolder + modelName)    
    confmat = pd.read_csv(thisPath / 'confusionMatrix.csv')
    w_N4 = 1
    w_Q4 = 3
    w_T4 = 3
    w_Q4H7 = 1
    accuracy_N4, accuracy_Q4, accuracy_T4, accuracy_Q4H7, customAcc = calculateCustomAccuracy(confmat, w_N4, w_Q4, w_T4, w_Q4H7)
    myFile.loc[i+1,13] = f"{accuracy_N4:.4f}"
    myFile.loc[i+1,14] = f"{accuracy_Q4:.4f}"
    myFile.loc[i+1,15] = f"{accuracy_T4:.4f}"
    myFile.loc[i+1,16] = f"{accuracy_Q4H7:.4f}"
    myFile.loc[i+1,17] = f"{customAcc:.4f}"

myFile.to_csv(conditionPath,sep = ",", header = False, index = False)