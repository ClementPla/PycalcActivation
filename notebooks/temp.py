from pathlib import Path
import torch
_ = torch.manual_seed(1234)
from socket import gethostname
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
        "whichDisplacement" : "xyPosition",
        "replace_nan_by_min" :False,
        "remove_mean": False,
        "whichFFT" : True,
        "numRNN" : 3,
        "sizeRNN": 64,
        "bidir" : True,
        "numFC": 3,
        "sizeFC" :  8,
        "dropout" : 0.15816701156730234,
        "weighted" : True,
        "customLoss" : False,
        "initial_lr": 0.01,
        "weight_decay": 0.0001,
        "batch_size": 2048,
        "store_best": "Accuracy",
        "loss": "Huber",
        }

weighted:true
whichDataset:"ratio"
whichDisplacement:"displacement"
whichFFT:true
sweep_id = ""
is_regression = True
model_unique_name, trainer = setupTrainer(config, is_regression, sweep_id, None)
n_epoch = 10000
trainer.train(n_epoch, val_patience=100)
metric_train, metric_val, metric_test = save_model_perf(trainer, model_unique_name,  sweep_id, False)
metrics = save_model_generalizability(trainer, model_unique_name, sweep_id, False)

if is_regression:
    saveFolder = saveFolder.joinpath("regressor")
else:
    saveFolder = saveFolder.joinpath("classifier")

from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
   # save model

_, saveFolder = getPath(trainer.is_regression, sweep_id)   
thisPath = saveFolder.joinpath(model_unique_name)

trainer.load_best()
metrics = {}

this_dict = {
                "N4" : -12.9,
                "Q4" : -10.9,
                "T4" : -9.5,
                "Q4H7" : -8.9,
                "M9" : -11.7,
                "L6F" : -8.00,
                "C9" : -8.04,
                "OT3_N4" : -10.6,
                "OT3_Q4" : -11.4,
                "-6" : -12.9,
                "-8" : -12.9,
                "-10" : -12.9,
                "-12" : -12.9,
            }
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
        y_pred = trainer.predict(x).cpu().squeeze().numpy()

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
        plt.savefig(thisPath.joinpath('confusionMatrix_' + k + '.pdf'))

        # write
        labels_pred = np.array(trainer.dataset.labels("OTI"), dtype = str)
        labels_GT = np.array(trainer.dataset.labels(k), dtype = str)
        write_confmat = np.row_stack((labels_pred,this_confmat))
        write_confmat = np.column_stack((np.concatenate(([' '], labels_GT)), write_confmat))
        np.savetxt(thisPath.joinpath('confusionMatrix_' + k + '.csv'), write_confmat, delimiter=",", fmt='%s')
        write_confmat = np.row_stack((labels_pred,this_confmat_norm))
        write_confmat = np.column_stack((np.concatenate(([' '], labels_GT)), write_confmat))
        np.savetxt(thisPath.joinpath('confusionMatrixZScore_' + k + '.csv'), write_confmat, delimiter=",", fmt='%s')

