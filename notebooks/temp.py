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
        "whichDisplacement" : "displacement",
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
        "initial_lr": 0.001,
        "weight_decay": 1e-4,
        "batch_size": 1024,
        }

is_regression = True
model_unique_name, trainer = setupTrainer(config, is_regression)
n_epoch = 50
trainer.train(n_epoch)
metric_train, metric_val, metric_test = save_model_perf(trainer, model_unique_name)

if is_regression:
    saveFolder = saveFolder.joinpath("regressor")
else:
    saveFolder = saveFolder.joinpath("classifier")

from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder

thisPath = saveFolder.joinpath(model_unique_name)
trainer.load_best()
metrics = {}

this_dict = {
                "N4" : 2.28e-13,
                "Q4" : 7.37e-11,
                "T4" : 4.76e-10,
                "Q4H7" : 2.46e-9,
                "M9" : 2.64e-12,
                "L6F" : 1e-8,
                "C9" : 5.29e-8,
                "OT3_N4" : 2.34e-11,
                "OT3_Q4" : 3.92e-12,
                "-6" : 2.28e-13,
                "-8" : 2.28e-13,
                "-10" : 2.28e-13,
                "-12" : 2.28e-13,
            }
# cost_matrix = {
#     "OTI" : np.array([[1.0,0.6,0.3,0],[0.6,1.0,0.6,0.3],[0.3,0.6,1.0,0.6], [0,0.3,0.6,1.0]]),
#     "SL" : np.array([[1.0,0.6,0.3,0],[0.6,1.0,0.6,0.3],[0.3,0.6,1.0,0.6], [0,0.3,0.6,1.0]]),
#     "P14" : np.array([[0,0.3,1.0,0.6],[0,0.3,1.0,0.6],[1.0,0.6,0,0.3]]),
#     "OT3" : np.array([[0.6,1.0,0,0.3],[0.6,1.0,0,0.3]]),
#     "conc" : np.array([[1.0,0.6,0.3,0],[1.0,0.6,0.3,0],[1.0,0.6,0.3,0],[1.0,0.6,0.3,0]]),
# }

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
        metrics.update({k+"_dist": this_metric}) # need to min (distance to target)

        # fScore metrics
        f = myFScore()
        f.update(preds = torch.Tensor(y_pred), target = torch.Tensor(classes_encoded))
        metrics.update({k+"_fScore": 1/f.compute().numpy()}) # need to min ( inverse of FScore)

        # plot distribution
        fig = plt.figure()
        for cls in apl_classes:
            _ = plt.hist(y_pred[classes_encoded == cls], 100, alpha=0.5, label=f"Class {classes_decoder[cls]}", density=True)
        _ = fig.suptitle('Best Model - ' + k + " - Fscore = " + str(f.compute().numpy())) 
        _ = plt.legend()
        plt.savefig(thisPath.joinpath('predictionDistribution_' + k + '.pdf'))


    else:
        # generate cost matrix 
        GT = np.array([np.log10(this_dict[v]) for v in trainer.dataset.f.all_data["OTI"]["mapping"].values()])
        pred = np.array([np.log10(this_dict[v]) for v in trainer.dataset.f.all_data[k]["mapping"].values()])
        this_cost_matrix = cdist(pred.reshape(-1,1), GT.reshape(-1,1), metric='euclidean')

        # model predicion on this dataset
        y_pred = trainer.predict(x)
        predicted_class = y_pred.argmax(dim = 1)
        
        # calculate this metric
        this_metrics = np.mean(this_cost_matrix[y.cpu(),predicted_class.cpu()])
        metrics.update({k + "_dist": this_metrics})   # need to min (distance to target)

        # print and write all "confusion matrices"
        n_pred_classes = len(np.unique(predicted_class.cpu()));
        n_GT_classes = len(apl_classes)
        this_confmat = np.zeros((n_GT_classes, n_pred_classes))
        for r in range(0, n_GT_classes):
            for c in range(0,n_pred_classes):
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
        _ = fig.suptitle('Best Model - ' + k + " - Custom metric = " +str(metrics[k+ "_dist"])) 
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

metric_to_min = sum([metrics[k] for k in ["SL_dist", "P14_dist", "OT3_dist"]]) # need to min
if is_regression:
    metric_to_min += sum([metrics[k] for k in ["SL_fScore", "P14_fScore", "OT3_fScore"]])
