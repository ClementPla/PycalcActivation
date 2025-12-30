import pandas as pd
from scipy.stats import spearmanr, pearsonr, gaussian_kde
from torch import norm
import wandb
import json
import os 
from pathlib import Path
from sklearn.metrics import cohen_kappa_score 
from PycalcAct.train_function import *
import matplotlib
matplotlib.use('TkAgg') 
import shutil
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder


os.environ['WANDB_API_KEY'] = '73246a79f06da26fb325d763bd90ab7fc81bc9e6'
api = wandb.Api()

# Project is specified by <entity/project-name>
is_regression = True
if is_regression:
    runs = api.runs("sebthis-mcgill-university/my-first-sweep-regressor", 
                    filters = {"$or": [
                {"sweep": "rcy5kh8r"},
                {"sweep": "e6dz51o5"}, 
                {"sweep": "z6hep2y2"}]})
else:
    runs = api.runs("sebthis-mcgill-university/my-first-sweep-classifier",
                    filters = {"$or": [
                {"sweep": "1ln4tilj"},
                {"sweep": "67ey3j68"}]})

EC50 = pd.read_csv("EC50.csv", index_col=None , header=None)
EC50 = {
    (row.iloc[0] if pos < 9 else int(row.iloc[0])): row.iloc[1]
    for pos, (_, row) in enumerate(EC50.iterrows())
}
rev_EC50 = {v: k for k, v in list(EC50.items())[0:8]}

# runs_df = pd.DataFrame()
# config_list = []
tableFolder = getPath(is_regression, "")[1].parent
if not os.path.exists(tableFolder.joinpath("project.csv")):
    for i, run in enumerate(runs[261:1582]):
        print(i)     
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files

        summary = json.loads(run.summary._json_dict)
        if 'model_unique_name' not in summary.keys():
            print(f"Run {run.name} has no model_unique_name, skipping...")
            continue
            
        # .name is the human-readable name of th e run.
        test = run.sweep
        model_unique_name = summary['model_unique_name']

        sweep = run.sweep
        sweep_id =  run.sweep.id

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config = {k: v['value'] for k,v in json.loads(run.config).items() if not k.startswith('_')}
        config_list.append(config)
        
        _, saveFolder = getPath(is_regression, sweep_id)    
        thisModelPath = saveFolder.joinpath(model_unique_name)

        if os.path.getsize(thisModelPath) == 0:
            print(f"Model: {model_unique_name} has no saved model.")
        else:
            # check if config has all required keys, if not add default values
            if 'myLoss' not in config.keys():
                config['myLoss'] = 'MSE'

            # reload model from config file
            _, trainer = setupTrainer(config, is_regression, sweep_id, model_unique_name = model_unique_name, EC50_path = "D:\sebastien\PycalcActivation\EC50.csv")
            thisModelName = thisModelPath.joinpath("savedModel.pt")
            checkpoint = torch.load(thisModelName, weights_only=True)
            trainer.model.load_state_dict(checkpoint["best"])
            trainer.model.to("cuda")
            del checkpoint

            # from OTI-test, extract optimal thresholds for confusion matrix and prediction_labels
            x, y = trainer.dataset.train_batch(True, to_cuda=True)
            y = y.cpu().detach().numpy().squeeze()
            y_pred = trainer.model(x).squeeze().cpu().detach().numpy()

            # fig = plt.figure()
            # for cls in np.unique(y):
            #     _ = plt.hist(y_pred[y == cls], 50, alpha=0.5, label=f"Class {cls}", density=True)
            #     _ = plt.legend()
            # fig.show()    

            # find weird repeated value in y_pred and remove from x and y
            vals, counts = np.unique(y_pred, return_counts=True)
            # if np.any(counts > 500):
            #     print("Weird repeated value found in predictions, not removed from analysis.")

            if np.any(counts > 500) and len(counts) > 50:
                weird_value = np.sort(vals[counts > 500])
                print("Weird repeated value found in predictions, removing from analysis.")
            else:
                weird_value = []

            for v in weird_value:
                y = y[y_pred != v]
                x = x[y_pred != v,:,:]
                y_pred = y_pred[y_pred != v]    

            # fig = plt.figure()
            # for cls in np.unique(y):
            #     _ = plt.hist(y_pred[y == cls], 50, alpha=0.5, label=f"Class {cls}", density=True)
            #     _ = plt.legend()
            # fig.show()    

            medians = []
            for unique_class in np.unique(y):
                    class_mask = (y == unique_class)
                    preds_for_class = y_pred[class_mask]
                    median_pred = np.median(preds_for_class)
                    medians.append(median_pred.item())
            optimal_thresholds = []
            sorted_medians = sorted(medians)
            for j in range(len(sorted_medians) - 1):
                threshold = (sorted_medians[j] + sorted_medians[j + 1]) / 2
                optimal_thresholds.append(threshold)
            prediction_labels = [str(rev_EC50[label]) for label in np.unique(y)]

            # remake prediction on all datasets
            callbacks = (
                trainer.dataset.test_batch,
                trainer.dataset.P14_batch,
                trainer.dataset.OT3_batch,
                trainer.dataset.conc_batch,
                trainer.dataset.SL_batch,
                    )
            all_metrics = {}
            all_confmat = []
            for _, (name, callable) in enumerate(zip(["OTI_Test", "P14", "OT3", "conc", "SL"], callbacks)):
                this_metrics = {}
                
                # Load data
                x, y = callable(True, to_cuda=True)
                y = y.cpu().detach().numpy().squeeze()
                if name == "conc":
                    y_ec50 = y.copy()
                    y = trainer.dataset.f.all_data["conc"]["classes"]
                
                encoder = LabelEncoder()
                y_encoded = encoder.fit_transform(y)

                # Make predictions
                y_pred = trainer.model(x).squeeze().cpu().detach().numpy()

                # remove weird values fron y and y_pred
                for v in weird_value:
                    y = y[y_pred != v]
                    x = x[y_pred != v,:,:]
                    y_encoded = y_encoded[y_pred != v]
                    if name == "conc":
                        y_ec50 = y_ec50[y_pred != v]
                    y_pred = y_pred[y_pred != v]

                # find interpolated y given prediction of OT-I
                original_refs = [EC50[v] for v in ["N4", "Q4", "T4", "Q4H7"]]
                predicted_refs = medians
                interp_func = interp1d(original_refs, predicted_refs, kind='linear', fill_value="extrapolate")
                y_interp = interp_func(y_ec50) if name == "conc" else interp_func(y)
            
                # for each unque class of y calculate mean, median, std and distances of predicted values to GT and interp_GT
                for unique_class in np.unique(y):
                    class_mask = (y == unique_class)
                    preds_for_class = y_pred[class_mask]
                    if len(preds_for_class) == 0:
                        continue
                    mean_pred = np.mean(preds_for_class)
                    median_pred = np.median(preds_for_class)
                    std_pred = np.std(preds_for_class)
                    if name == "conc":
                        distance_pred = np.abs(median_pred - int(unique_class))
                        class_name = unique_class
                    else:
                        distance_pred = np.abs(median_pred - unique_class.item())
                        class_name = rev_EC50[unique_class.item()]
                    distance_pred_interp = np.abs(median_pred - y_interp[class_mask][0])
                    this_metrics[class_name + "_mean_pred"] = mean_pred
                    this_metrics[class_name + "_median_pred"] = median_pred
                    this_metrics[class_name + "_std_pred"] = std_pred
                    this_metrics[class_name + "_distance_pred"] = distance_pred
                    this_metrics[class_name + "_distance_pred_interp"] = distance_pred_interp                    

                # calculate F-Score, spearman and pearson regression and overall distance metric for all classes
                f = myFScore()
                f.update(preds = torch.Tensor(y_pred), target = torch.Tensor(y_encoded))
                this_metrics.update({"fScore" : f.compute().item()})

                if name == "conc":
                    rho_spearman = np.nan
                    rho_pearson = np.nan
                    overall_distance = np.mean(np.abs(y_ec50 - y_pred))
                    mean_class_distance = np.mean([this_metrics[unique_class + "_distance_pred"] for unique_class in np.unique(y)])
                else:
                    rho_spearman, _ = spearmanr(y, y_pred)
                    rho_pearson, _ = pearsonr(y, y_pred)
                    overall_distance = np.mean(np.abs(y - y_pred))
                    mean_class_distance = np.mean([this_metrics[rev_EC50[unique_class.item()] + "_distance_pred"] for unique_class in np.unique(y)])

                    
                this_metrics.update({"pearsonCorr" : rho_pearson, "spearmanCorr" : rho_spearman, \
                                     "overall_distance" : overall_distance, "mean_distance" : mean_class_distance})
                
                # compute confusion matrix
                    # calculate conf matrix from optimal thresholds
                y_pred_classes = np.digitize(y_pred, bins=optimal_thresholds)
                conf_matrix = np.zeros((len(np.unique(y)), len(prediction_labels)), dtype=int)
                for true_label, pred_label in zip(y_encoded, y_pred_classes):
                    conf_matrix[true_label, pred_label] += 1
                if name == "conc":
                    GT_labels = np.unique(y).astype(int).tolist()
                else : 
                    GT_labels = [str(rev_EC50[label]) for label in np.unique(y)]

                    # calculate metrics on confusion matrix
                if name == "OTI_Test" or name == "SL":
                    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
                    kappa = cohen_kappa_score(y_encoded, y_pred_classes)
                else:
                    accuracy = np.nan
                    kappa = np.nan
                this_apl_ec50 = np.array([EC50[label] for label in prediction_labels])
                row_ec50 = np.dot(conf_matrix, this_apl_ec50)/np.sum(conf_matrix, axis = 1)

                GT = np.array([EC50[v] for v in prediction_labels])
                pred = np.array([EC50[v] for v in GT_labels])
                this_distance_matrix = cdist(pred.reshape(-1,1), GT.reshape(-1,1), metric='euclidean')
                dist_metric = np.sum(conf_matrix*this_distance_matrix)/np.sum(conf_matrix)
                row_dist_metric = np.sum(conf_matrix*this_distance_matrix, axis=1)/np.sum(conf_matrix, axis=1)

                    # append to this_metrics
                this_metrics['confmat_accuracy'] = accuracy
                this_metrics['confmat_kappa'] = kappa
                this_metrics['confmat_dist_metric'] = dist_metric
                for i, label in enumerate(GT_labels):
                    this_metrics[f'{label}_confmat_row_dist'] = row_dist_metric[i]
                    this_metrics[f'{label}_confmat_row_ec50'] = row_ec50[i]

                    # append row labels and column labels to confmatrix
                conf_matrix_labeled = np.zeros((conf_matrix.shape[0]+1, conf_matrix.shape[1]+1), dtype=object)
                conf_matrix_labeled[0,1:] = prediction_labels
                conf_matrix_labeled[1:,0] = GT_labels
                conf_matrix_labeled[1:,1:] = conf_matrix
                all_confmat.append(conf_matrix_labeled) 

                # # show histogram of prediction
                # fig = plt.figure()
                # for cls in np.unique(y):
                #     _ = plt.hist(y_pred[y == cls], 50, alpha=0.5, label=f"Class {cls}", density=True)
                #     _ = plt.legend()
                # fig.show()

                # pool metrics
                all_metrics[name] = this_metrics



            # save confmat to csv 
            confmat_folder = thisModelPath.joinpath("confusion_matrices_optimalThr")
            confmat_folder.mkdir(exist_ok=True)
            for j, (name, confmat) in enumerate(zip(["OTI_Test", "P14", "OT3", "conc", "SL"], all_confmat)):
                confmat_path = confmat_folder.joinpath(f"{name}_confusion_matrix.csv")
                pd.DataFrame(confmat).to_csv(confmat_path, header=False, index=False)

            # export all_metric to a dataframe
            all_metrics_df = pd.DataFrame.from_dict(all_metrics)
            all_metrics_vector = all_metrics_df.transpose().to_numpy().flatten()
            all_metrics_colnames = []
            for dataset in all_metrics_df.columns:
                for metric in all_metrics_df.index:
                    all_metrics_colnames.append(f"{dataset}_{metric}")
            all_metrics_df = pd.DataFrame(all_metrics_vector.reshape(1, -1), columns=all_metrics_colnames, index=[i])
            all_metrics_df = all_metrics_df.loc[:, ~all_metrics_df.isna().all()]

            # add model name and model unique name to beginning of all_metrics_df
            all_metrics_df.insert(0, 'sweep_id', sweep_id)
            all_metrics_df.insert(0, 'model_unique_name', model_unique_name)
            all_metrics_df.insert(0, 'run_name', run.name)

        runs_df = pd.concat([runs_df, all_metrics_df], ignore_index=True)


    config_list_df = pd.DataFrame(config_list)

    runs_df.to_csv(tableFolder.joinpath("project.csv"))
    config_list_df.to_csv(tableFolder.joinpath("project_config.csv"))
else:
    runs_df = pd.read_csv(tableFolder.joinpath("project.csv"), index_col=0)
    config_list_df = pd.read_csv(tableFolder.joinpath("project_config.csv"), index_col=0)


# Normalization of metrics relating to confusion matrix and synthetic EC50
    # load min and max value from file

limits = pd.read_csv("models/round3/classifier/limits.csv", index_col=0)
limits_row = pd.read_csv("models/round3/classifier/limits_row.csv", index_col=0)
row_dist_ideal = {}
for row in limits_row.index:
    for col in limits_row.columns:
        row_dist_ideal[f"{col}_{row}"] = [float(x) for x in limits_row[col].values[row].strip('[]').split(' ') if x != '']

this_min = [0, 0,0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	\
            limits['this_min'][0],	0,  limits['this_min'][1],	row_dist_ideal["this_min_row_distance_0"][0],	0,	row_dist_ideal["this_min_row_distance_0"][1], \
            0,	row_dist_ideal["this_min_row_distance_0"][3],	0,	row_dist_ideal["this_min_row_distance_0"][2],	0,	\
            0,	0,	0,	0,	0,	limits['this_min'][2],	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	\
            row_dist_ideal["this_min_row_distance_1"][2],	0,	row_dist_ideal["this_min_row_distance_1"][1],	0,	\
            row_dist_ideal["this_min_row_distance_1"][0],	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
            limits['this_min'][3],	row_dist_ideal["this_min_row_distance_2"][1],	0,	0,	0,	0,	0,	0,	row_dist_ideal["this_min_row_distance_2"][0],	0, \
            0,	0,	0,	limits['this_min'][4],	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
            row_dist_ideal["this_min_row_distance_3"][0],	0,	row_dist_ideal["this_min_row_distance_3"][1],	0, \
            row_dist_ideal["this_min_row_distance_3"][2], 0,	row_dist_ideal["this_min_row_distance_3"][3],	0, \
            0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, \
            limits['this_min'][0],	0,	limits['this_min'][5], \
            row_dist_ideal["this_min_row_distance_4"][0],	0,	row_dist_ideal["this_min_row_distance_4"][1],	0, \
            row_dist_ideal["this_min_row_distance_4"][3],	0,	row_dist_ideal["this_min_row_distance_4"][2],	0]

this_max = [1, 1,1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	\
            limits['this_max'][0],	1,  limits['this_max'][1],	row_dist_ideal["this_max_row_distance_0"][0],	1,	row_dist_ideal["this_max_row_distance_0"][1], \
            1,	row_dist_ideal["this_max_row_distance_0"][3],	1,	row_dist_ideal["this_max_row_distance_0"][2],	1,	\
            1,	1,	1,	1,	1,	limits['this_max'][2],	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	\
            row_dist_ideal["this_max_row_distance_1"][2],	1,	row_dist_ideal["this_max_row_distance_1"][1],	1,	\
            row_dist_ideal["this_max_row_distance_1"][0],	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1, \
            limits['this_max'][3],	row_dist_ideal["this_max_row_distance_2"][1],	1,	1,	1,	1,	1,	1,	row_dist_ideal["this_max_row_distance_2"][0],	1, \
            1,	1,	1,	limits['this_max'][4],	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1, \
            row_dist_ideal["this_max_row_distance_3"][0],	1,	row_dist_ideal["this_max_row_distance_3"][1],	1, \
            row_dist_ideal["this_max_row_distance_3"][2], 1,	row_dist_ideal["this_max_row_distance_3"][3],	1, \
            1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1, \
            limits['this_max'][0],	1,	limits['this_max'][5], \
            row_dist_ideal["this_max_row_distance_4"][0],	1,	row_dist_ideal["this_max_row_distance_4"][1],	1, \
            row_dist_ideal["this_max_row_distance_4"][3],	1,	row_dist_ideal["this_max_row_distance_4"][2],	1]

norm_df = runs_df.copy()
for i, col in enumerate(runs_df.iloc[:,3:].columns):
    norm_df[col] = (runs_df[col] - this_min[i]) / (this_max[i] - this_min[i])
norm_df = norm_df.copy()

# composite scores
w_P14 = 1
w_OT3 = 1
w_OTI = 1
w_mean = 1
w_CV = 10
 # average of prediction distance per class  --> minimize
norm_df['composite_score_mean_distance'] = np.sqrt(
    w_OTI * (norm_df['OTI_Test_mean_distance'])**2 + 
    w_P14 * (norm_df['P14_mean_distance'])**2 + 
    w_OT3 * (norm_df['OT3_mean_distance'])**2) 
# average of prediction distance of all cells --> minimize
norm_df['composite_score_overall_distance'] = np.sqrt(
    w_OTI * (norm_df['OTI_Test_overall_distance'])**2 + 
    w_P14 * (norm_df['P14_overall_distance'])**2 + 
    w_OT3 * (norm_df['OT3_overall_distance'])**2) 
# average of prediction interpolated distance of all cells  --> minimize
norm_df['composite_score_mean_distance_interp'] = np.sqrt(
    w_OTI * np.mean([norm_df['OTI_Test_N4_distance_pred_interp'], norm_df['OTI_Test_Q4_distance_pred_interp'], \
                     norm_df['OTI_Test_T4_distance_pred_interp'], norm_df['OTI_Test_Q4H7_distance_pred_interp']], axis = 0)**2 +
    w_P14 * np.mean([norm_df['P14_M9_distance_pred_interp'], norm_df['P14_C9_distance_pred_interp'], norm_df['P14_L6F_distance_pred_interp']], axis = 0)**2 +
    w_OT3 * np.mean([norm_df['OT3_OT3_N4_distance_pred_interp'], norm_df['OT3_Q4_distance_pred_interp']], axis = 0)**2) 
# average of prediction  distance of all cells with SD  --> minimize
norm_df['composite_score_overall_distance_withCV'] = np.sqrt(
    w_mean * w_OTI * (norm_df['OTI_Test_overall_distance'])**2 + 
    w_CV * w_OTI * np.mean([norm_df['OTI_Test_N4_std_pred']/norm_df['OTI_Test_N4_median_pred'], norm_df['OTI_Test_Q4_std_pred']/norm_df['OTI_Test_Q4_median_pred']
                            , norm_df['OTI_Test_T4_std_pred']/norm_df['OTI_Test_T4_median_pred'], norm_df['OTI_Test_Q4H7_std_pred']/norm_df['OTI_Test_Q4H7_median_pred']], axis = 0)**2 +
    w_mean * w_P14 * (norm_df['P14_overall_distance'])**2 + 
    w_CV * w_P14 * np.mean([norm_df['P14_M9_std_pred']/norm_df['P14_M9_median_pred'], norm_df['P14_C9_std_pred']/norm_df['P14_C9_median_pred'], 
                            norm_df['P14_L6F_std_pred']/norm_df['P14_L6F_median_pred']], axis = 0)**2 +
    w_mean * w_OT3 * (norm_df['OT3_overall_distance'])**2 + 
    w_CV * w_OT3 * np.mean([norm_df['OT3_OT3_N4_std_pred']/norm_df['OT3_OT3_N4_median_pred'], 
                            norm_df['OT3_Q4_std_pred']/norm_df['OT3_Q4_median_pred']], axis = 0)**2) 
# average of prediction  distance of all cells with SD  --> minimize
norm_df['composite_score_mean_distance_withCV'] = np.sqrt(
    w_mean * w_OTI * (norm_df['OTI_Test_mean_distance'])**2 + 
    w_CV * w_OTI * np.mean([norm_df['OTI_Test_N4_std_pred']/norm_df['OTI_Test_N4_median_pred'], norm_df['OTI_Test_Q4_std_pred']/norm_df['OTI_Test_Q4_median_pred'], 
                            norm_df['OTI_Test_T4_std_pred']/norm_df['OTI_Test_T4_median_pred'], norm_df['OTI_Test_Q4H7_std_pred']/norm_df['OTI_Test_Q4H7_median_pred']], axis = 0)**2 +
    w_mean * w_P14 * (norm_df['P14_mean_distance'])**2 + 
    w_CV * w_P14 * np.mean([norm_df['P14_M9_std_pred']/norm_df['P14_M9_median_pred'], norm_df['P14_C9_std_pred']/norm_df['P14_C9_median_pred'], 
                            norm_df['P14_L6F_std_pred']/norm_df['P14_L6F_median_pred']], axis = 0)**2 +
    w_mean * w_OT3 * (norm_df['OT3_mean_distance'])**2 + 
    w_CV * w_OT3 * np.mean([norm_df['OT3_OT3_N4_std_pred']/norm_df['OT3_OT3_N4_median_pred'], 
                            norm_df['OT3_Q4_std_pred']/norm_df['OT3_Q4_median_pred']], axis = 0)**2) 
# average of prediction interpolated distance of all cells with SD  --> minimize
norm_df['composite_score_mean_distance_interp_withCV'] = np.sqrt(
    w_CV * w_OTI * np.mean([norm_df['OTI_Test_N4_std_pred']/norm_df['OTI_Test_N4_median_pred'], norm_df['OTI_Test_Q4_std_pred']/norm_df['OTI_Test_Q4_median_pred'], 
                            norm_df['OTI_Test_T4_std_pred']/norm_df['OTI_Test_T4_median_pred'], norm_df['OTI_Test_Q4H7_std_pred']/norm_df['OTI_Test_Q4H7_median_pred']], axis = 0)**2 +
    w_CV * w_P14 * np.mean([norm_df['P14_M9_std_pred']/norm_df['P14_M9_median_pred'], norm_df['P14_C9_std_pred']/norm_df['P14_C9_median_pred'], 
                            norm_df['P14_L6F_std_pred']/norm_df['P14_L6F_median_pred']], axis = 0)**2 +
    w_CV * w_OT3 * np.mean([norm_df['OT3_OT3_N4_std_pred']/norm_df['OT3_OT3_N4_median_pred'], 
                            norm_df['OT3_Q4_std_pred']/norm_df['OT3_Q4_median_pred']], axis = 0)**2 + 
    w_mean * w_OTI * np.mean([norm_df['OTI_Test_N4_distance_pred_interp'], norm_df['OTI_Test_Q4_distance_pred_interp'], \
                     norm_df['OTI_Test_T4_distance_pred_interp'], norm_df['OTI_Test_Q4H7_distance_pred_interp']], axis = 0)**2 +
    w_mean * w_P14 * np.mean([norm_df['P14_M9_distance_pred_interp'], norm_df['P14_C9_distance_pred_interp'], norm_df['P14_L6F_distance_pred_interp']], axis = 0)**2 +
    w_mean * w_OT3 * np.mean([norm_df['OT3_OT3_N4_distance_pred_interp'], norm_df['OT3_Q4_distance_pred_interp']], axis = 0)**2) # average of prediction distance of all cells with SD
# F-Score  --> maximize
norm_df['composite_score_fScore'] = np.sqrt(
    w_OTI * (norm_df['OTI_Test_fScore'])**2 + 
    w_P14 * (norm_df['P14_fScore'])**2 + 
    w_OT3 * (norm_df['OT3_fScore'])**2) 
# spearman correlation  --> maximize
norm_df['composite_score_spearmanCorr'] = np.sqrt(
    w_OTI * (norm_df['OTI_Test_spearmanCorr'])**2 + 
    w_P14 * (norm_df['P14_spearmanCorr'])**2 + 
    w_OT3 * (norm_df['OT3_spearmanCorr'])**2) 
# pearson correlation  --> maximize
norm_df['composite_score_pearsonCorr'] = np.sqrt(
    w_OTI * (norm_df['OTI_Test_pearsonCorr'])**2 + 
    w_P14 * (norm_df['P14_pearsonCorr'])**2 + 
    w_OT3 * (norm_df['OT3_pearsonCorr'])**2) 
# confmat distance metric  (same as classifier) --> minimize
norm_df['composite_score_confmat_distance'] = np.sqrt(
    w_OTI * (norm_df['OTI_Test_confmat_dist_metric'])**2 + 
    w_P14 * (norm_df['P14_confmat_dist_metric'])**2 + 
    w_OT3 * (norm_df['OT3_confmat_dist_metric'])**2)

# plot all composite scores against each other in subplots
# fig, axs = plt.subplots(9,9, figsize=(1, 1))
# composite_scores = ['composite_score_mean_distance', 'composite_score_overall_distance', \
#                     'composite_score_mean_distance_interp', 'composite_score_mean_distance_withCV', \
#                     'composite_score_overall_distance_withCV', 'composite_score_mean_distance_interp_withCV', \
#                     'composite_score_fScore', 'composite_score_spearmanCorr', 'composite_score_pearsonCorr', 'composite_score_confmat_distance']
# for j in range(9):
#     for i in range(j,9):
#         axs[i, j].scatter(norm_df[composite_scores[i]], norm_df[composite_scores[j]])
#         axs[i, j].set_xlabel(composite_scores[i])
#         axs[i, j].set_ylabel(composite_scores[j])
# plt.tight_layout()
# plt.show()

# find best model for each composite score
mean_distance_min_idx = norm_df['composite_score_mean_distance'].idxmin()
overall_distance_min_idx = norm_df['composite_score_overall_distance'].idxmin()
mean_distance_interp_min_idx = norm_df['composite_score_mean_distance_interp'].idxmin()
mean_distance_CV_min_idx = norm_df['composite_score_mean_distance_withCV'].idxmin()
overall_distance_CV_min_idx = norm_df['composite_score_overall_distance_withCV'].idxmin()
mean_distance_interp_CV_min_idx = norm_df['composite_score_mean_distance_interp_withCV'].idxmin()
fScore_max_idx = norm_df['composite_score_fScore'].idxmax()
spearmanCorr_max_idx = norm_df['composite_score_spearmanCorr'].idxmax()
pearsonCorr_max_idx = norm_df['composite_score_pearsonCorr'].idxmax()
confmat_distance_min_idx = norm_df['composite_score_confmat_distance'].idxmin()

# print best model info
best_model_mean_distance =  pd.concat([
    runs_df.loc[mean_distance_min_idx],
    norm_df.loc[mean_distance_min_idx]], axis=1)
best_model_overall_distance =  pd.concat([
    runs_df.loc[overall_distance_min_idx],
    norm_df.loc[overall_distance_min_idx]], axis=1)
best_model_mean_distance_interp =  pd.concat([
    runs_df.loc[mean_distance_interp_min_idx],
    norm_df.loc[mean_distance_interp_min_idx]], axis=1)
best_model_mean_distance_CV =  pd.concat([
    runs_df.loc[mean_distance_CV_min_idx],
    norm_df.loc[mean_distance_CV_min_idx]], axis=1) 
best_model_overall_distance_CV =  pd.concat([
    runs_df.loc[overall_distance_CV_min_idx],
    norm_df.loc[overall_distance_CV_min_idx]], axis=1) 
best_model_mean_distance_interp_CV =  pd.concat([
    runs_df.loc[mean_distance_interp_CV_min_idx],
    norm_df.loc[mean_distance_interp_CV_min_idx]], axis=1) 
best_model_fScore =  pd.concat([
    runs_df.loc[fScore_max_idx],
    norm_df.loc[fScore_max_idx]], axis=1) 
best_model_spearman =  pd.concat([
    runs_df.loc[spearmanCorr_max_idx],
    norm_df.loc[spearmanCorr_max_idx]], axis=1) 
best_model_pearson =  pd.concat([
    runs_df.loc[pearsonCorr_max_idx],
    norm_df.loc[pearsonCorr_max_idx]], axis=1) 
best_model_confmat_distance =  pd.concat([
    runs_df.loc[confmat_distance_min_idx],
    norm_df.loc[confmat_distance_min_idx]], axis=1) 

print(best_model_mean_distance),
print(best_model_overall_distance)
print(best_model_mean_distance_interp)
print(best_model_mean_distance_CV)
print(best_model_overall_distance_CV)
print(best_model_mean_distance_interp_CV)
print(best_model_fScore)
print(best_model_spearman)
print(best_model_pearson)
print(best_model_confmat_distance)

## 

# Save normalized dataframe
norm_df.to_csv(Path(tableFolder).joinpath("project_normalized.csv"))

# copy best model folder to a new location
bestFolder = getPath(is_regression, "")[1].parent
# best model path with todays date
best_model_path = Path(bestFolder).joinpath("best_model", f"{pd.Timestamp.now().strftime('%y%m%d')}")
best_model_src = Path(bestFolder).joinpath("sweep_rcy5kh8r", norm_df['model_unique_name'][overall_distance_min_idx])
if not os.path.exists(best_model_src):
    best_model_src = Path(bestFolder).joinpath("sweep_e6dz51o5", norm_df['model_unique_name'][overall_distance_min_idx])
if not os.path.exists(best_model_src):
    best_model_src = Path(bestFolder).joinpath("sweep_z6hep2y2", norm_df['model_unique_name'][overall_distance_min_idx])
shutil.copytree(best_model_src, best_model_path, dirs_exist_ok=True)

best_config = config_list_df.iloc[overall_distance_min_idx,:]
pd.DataFrame(data = best_config, index=[0]).to_csv(best_model_path.joinpath("wandb_run_info.csv"))
#  write runs_df.loc[synthetic_min_idx] to a text file
# zrite best model info to a text file
with open(best_model_path.joinpath("model_info.txt"), "w") as f:
    f.write("Best model info:\n")
    f.write("Parameter \t Raw Value \t Normalized Value\n")
    for row in runs_df.loc[overall_distance_min_idx].index:
        f.write(f"{row}: \t {runs_df.loc[overall_distance_min_idx, row]} \t {norm_df.loc[overall_distance_min_idx, row]}\n" 
                if runs_df.loc[overall_distance_min_idx, row] != norm_df.loc[overall_distance_min_idx, row] 
                else f"{row}: \t {runs_df.loc[overall_distance_min_idx, row]}\n")
    f.write("\n\nBest model config:\n")
    for row in best_config.index:
        f.write(f"{row}: {best_config.loc[row]}\n")


## Load best model
best_config_dict = best_config.to_dict()
_, best_trainer = setupTrainer(best_config_dict, is_regression, runs_df.loc[overall_distance_min_idx, 'sweep_id'], 
                          model_unique_name = runs_df.loc[overall_distance_min_idx, 'model_unique_name'], 
                          EC50_path = "D:\sebastien\PycalcActivation\EC50.csv")
thisModelName = best_model_path.joinpath("savedModel.pt")
checkpoint = torch.load(thisModelName, weights_only=True)
best_trainer.model.load_state_dict(checkpoint["best"])
best_trainer.model.to("cuda")
del checkpoint

confmat_folder = best_model_path.joinpath("confusion_matrices_optimalThr")
zscore_folder = best_model_path.joinpath("confusion_matrices_zscore")
zscore_folder.mkdir(exist_ok=True)
pdf_folder = best_model_path.joinpath("predicted_pdfs")
pdf_folder.mkdir(exist_ok=True)

callbacks = (
    best_trainer.dataset.test_batch,
    best_trainer.dataset.P14_batch,
    best_trainer.dataset.OT3_batch,
    best_trainer.dataset.conc_batch,
    best_trainer.dataset.SL_batch,
        )

for _, (name, callable) in enumerate(zip(["OTI_Test", "P14", "OT3", "conc", "SL"], callbacks)):  
    # Load data
    print(name)
    if name == "OTI_Test":
        medians = []
        for unique_class in np.unique(y):
                class_mask = (y == unique_class)
                preds_for_class = y_pred[class_mask]
                median_pred = np.median(preds_for_class)
                medians.append(median_pred.item())
        optimal_thresholds = []
        sorted_medians = sorted(medians)
        for j in range(len(sorted_medians) - 1):
            threshold = (sorted_medians[j] + sorted_medians[j + 1]) / 2
            optimal_thresholds.append(threshold)
        # write optimal thresholds to csv
        optimal_thresholds_df = pd.DataFrame(optimal_thresholds, columns=["optimal_thresholds"])
        optimal_thresholds_df.to_csv(best_model_path.joinpath(f"optimal_thresholds.csv"), index=False)

    x, y = callable(True, to_cuda=True)
    y = y.cpu().detach().numpy().squeeze()
    if name == "conc":
        y_ec50 = y.copy()
        y = best_trainer.dataset.f.all_data["conc"]["classes"]
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Make predictions
    y_pred = best_trainer.model(x).squeeze().cpu().detach().numpy()

    # compute pdf of each class
    fig = plt.figure()
    x_eval = {}
    pdf_values = {}
    x_eval = np.linspace(-16,-7, 1000)
    for cls in np.unique(y):
        kde = gaussian_kde(y_pred[y == cls])
        cls_name = str(rev_EC50[cls]) if name != "conc" else str(int(cls))
        pdf_values[cls_name] = kde(x_eval)
        _ = plt.plot(x_eval, pdf_values[cls_name], label=f"Class {cls_name}")
        _ = plt.legend()
    fig.suptitle(f"PDF of predicted values for {name} dataset")
    fig.show()

    # save pdf values to csv
    pdf_df = pd.DataFrame(pdf_values, index=x_eval)
    pdf_df.to_csv(pdf_folder.joinpath(f"{name}_predicted_pdf.csv"))

# compute z-score for each confusion matrix row
for confmat_file in confmat_folder.iterdir():
    zscore = pd.read_csv(confmat_file, header=None).to_numpy()
    for i in range(1, zscore.shape[0]):
        row = zscore[i,1:].astype(float)
        row_mean = np.mean(row)
        row_std = np.std(row)
        if row_std == 0:
            zscore[i,1:] = 0
        else:
            zscore[i,1:] = (row - row_mean) / row_std
    pd.DataFrame(zscore).to_csv(zscore_folder.joinpath(confmat_file.name), header=False, index=False)



            # medians = []
            # for unique_class in np.unique(y):
            #         class_mask = (y == unique_class)
            #         preds_for_class = y_pred[class_mask]
            #         median_pred = np.median(preds_for_class)
            #         medians.append(median_pred.item())
            # optimal_thresholds = []
            # sorted_medians = sorted(medians)
            # for j in range(len(sorted_medians) - 1):
            #     threshold = (sorted_medians[j] + sorted_medians[j + 1]) / 2
            #     optimal_thresholds.append(threshold)
            # prediction_labels = [str(rev_EC50[label]) for label in np.unique(y)]

            # # remake prediction on all datasets
            # callbacks = (
            #     trainer.dataset.test_batch,
            #     trainer.dataset.P14_batch,
            #     trainer.dataset.OT3_batch,
            #     trainer.dataset.conc_batch,
            #     trainer.dataset.SL_batch,
            #         )
            # all_metrics = {}
            # all_confmat = []
            # for _, (name, callable) in enumerate(zip(["OTI_Test", "P14", "OT3", "conc", "SL"], callbacks)):
            #     this_metrics = {}
                
            #     # Load data
            #     x, y = callable(True, to_cuda=True)
            #     y = y.cpu().detach().numpy().squeeze()
            #     if name == "conc":
            #         y_ec50 = y.copy()
            #         y = trainer.dataset.f.all_data["conc"]["classes"]
                
            #     encoder = LabelEncoder()
            #     y_encoded = encoder.fit_transform(y)

            #     # Make predictions
            #     y_pred = trainer.model(x).squeeze().cpu().detach().numpy()

            #     # remove weird values fron y and y_pred
            #     for v in weird_value:
            #         y = y[y_pred != v]
            #         x = x[y_pred != v,:,:]
            #         y_encoded = y_encoded[y_pred != v]
            #         if name == "conc":
            #             y_ec50 = y_ec50[y_pred != v]
            #         y_pred = y_pred[y_pred != v]

            #     # find interpolated y given prediction of OT-I
            #     original_refs = [EC50[v] for v in ["N4", "Q4", "T4", "Q4H7"]]
            #     predicted_refs = medians
            #     interp_func = interp1d(original_refs, predicted_refs, kind='linear', fill_value="extrapolate")
            #     y_interp = interp_func(y_ec50) if name == "conc" else interp_func(y)
            
            #     # for each unque class of y calculate mean, median, std and distances of predicted values to GT and interp_GT
            #     for unique_class in np.unique(y):
            #         class_mask = (y == unique_class)
            #         preds_for_class = y_pred[class_mask]
            #         if len(preds_for_class) == 0:
            #             continue
            #         mean_pred = np.mean(preds_for_class)
            #         median_pred = np.median(preds_for_class)
            #         std_pred = np.std(preds_for_class)
            #         if name == "conc":
            #             distance_pred = np.abs(median_pred - int(unique_class))
            #             class_name = unique_class
            #         else:
            #             distance_pred = np.abs(median_pred - unique_class.item())
            #             class_name = rev_EC50[unique_class.item()]
            #         distance_pred_interp = np.abs(median_pred - y_interp[class_mask][0])
            #         this_metrics[class_name + "_mean_pred"] = mean_pred
            #         this_metrics[class_name + "_median_pred"] = median_pred
            #         this_metrics[class_name + "_std_pred"] = std_pred
            #         this_metrics[class_name + "_distance_pred"] = distance_pred
            #         this_metrics[class_name + "_distance_pred_interp"] = distance_pred_interp                    

            #     # calculate F-Score, spearman and pearson regression and overall distance metric for all classes
            #     f = myFScore()
            #     f.update(preds = torch.Tensor(y_pred), target = torch.Tensor(y_encoded))
            #     this_metrics.update({"fScore" : f.compute().item()})

            #     if name == "conc":
            #         rho_spearman = np.nan
            #         rho_pearson = np.nan
            #         overall_distance = np.mean(np.abs(y_ec50 - y_pred))
            #         mean_class_distance = np.mean([this_metrics[unique_class + "_distance_pred"] for unique_class in np.unique(y)])
            #     else:
            #         rho_spearman, _ = spearmanr(y, y_pred)
            #         rho_pearson, _ = pearsonr(y, y_pred)
            #         overall_distance = np.mean(np.abs(y - y_pred))
            #         mean_class_distance = np.mean([this_metrics[rev_EC50[unique_class.item()] + "_distance_pred"] for unique_class in np.unique(y)])

                    
            #     this_metrics.update({"pearsonCorr" : rho_pearson, "spearmanCorr" : rho_spearman, \
            #                          "overall_distance" : overall_distance, "mean_distance" : mean_class_distance})
                
            #     # compute confusion matrix
            #         # calculate conf matrix from optimal thresholds
            #     y_pred_classes = np.digitize(y_pred, bins=optimal_thresholds)
            #     conf_matrix = np.zeros((len(np.unique(y)), len(prediction_labels)), dtype=int)
            #     for true_label, pred_label in zip(y_encoded, y_pred_classes):
            #         conf_matrix[true_label, pred_label] += 1
            #     if name == "conc":
            #         GT_labels = np.unique(y).astype(int).tolist()
            #     else : 
            #         GT_labels = [str(rev_EC50[label]) for label in np.unique(y)]

            #         # calculate metrics on confusion matrix
            #     if name == "OTI_Test" or name == "SL":
            #         accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
            #         kappa = cohen_kappa_score(y_encoded, y_pred_classes)
            #     else:
            #         accuracy = np.nan
            #         kappa = np.nan
            #     this_apl_ec50 = np.array([EC50[label] for label in prediction_labels])
            #     row_ec50 = np.dot(conf_matrix, this_apl_ec50)/np.sum(conf_matrix, axis = 1)

            #     GT = np.array([EC50[v] for v in prediction_labels])
            #     pred = np.array([EC50[v] for v in GT_labels])
            #     this_distance_matrix = cdist(pred.reshape(-1,1), GT.reshape(-1,1), metric='euclidean')
            #     dist_metric = np.sum(conf_matrix*this_distance_matrix)/np.sum(conf_matrix)
            #     row_dist_metric = np.sum(conf_matrix*this_distance_matrix, axis=1)/np.sum(conf_matrix, axis=1)

            #         # append to this_metrics
            #     this_metrics['confmat_accuracy'] = accuracy
            #     this_metrics['confmat_kappa'] = kappa
            #     this_metrics['confmat_dist_metric'] = dist_metric
            #     for i, label in enumerate(GT_labels):
            #         this_metrics[f'{label}_confmat_row_dist'] = row_dist_metric[i]
            #         this_metrics[f'{label}_confmat_row_ec50'] = row_ec50[i]

            #         # append row labels and column labels to confmatrix
            #     conf_matrix_labeled = np.zeros((conf_matrix.shape[0]+1, conf_matrix.shape[1]+1), dtype=object)
            #     conf_matrix_labeled[0,1:] = prediction_labels
            #     conf_matrix_labeled[1:,0] = GT_labels
            #     conf_matrix_labeled[1:,1:] = conf_matrix
            #     all_confmat.append(conf_matrix_labeled) 

            #     # # show histogram of prediction
            #     # fig = plt.figure()
            #     # for cls in np.unique(y):
            #     #     _ = plt.hist(y_pred[y == cls], 50, alpha=0.5, label=f"Class {cls}", density=True)
            #     #     _ = plt.legend()
            #     # fig.show()

            #     # pool metrics
            #     all_metrics[name] = this_metrics