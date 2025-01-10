import numpy as np
import torch

def calculateCustomAccuracy(confmat, w_N4: int = 1, w_Q4: int = 3, w_T4: int = 4, w_Q4H7: int = 2):
    confmat = np.array(confmat, dtype = float) 
    TP_T4 = confmat[3,3]
    N_T4 = sum(confmat[3,:])
    accuracy_T4 = TP_T4/N_T4
    TP_Q4H7 = confmat[2,2]
    N_Q4H7 = sum(confmat[2,:])
    accuracy_Q4H7 = TP_Q4H7/N_Q4H7
    TP_Q4 = confmat[1,1]
    N_Q4 = sum(confmat[1,:])
    accuracy_Q4 = TP_Q4/N_Q4
    TP_N4 = confmat[0,0]
    N_N4 = sum(confmat[0,:])
    accuracy_N4 = TP_N4/N_N4

    customAcc = (w_N4*accuracy_N4 + w_Q4*accuracy_Q4 + w_T4*accuracy_T4 + w_Q4H7*accuracy_Q4H7)/sum([w_N4, w_Q4, w_T4, w_Q4H7])
    return accuracy_N4, accuracy_Q4, accuracy_T4, accuracy_Q4H7, customAcc