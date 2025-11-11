import pandas as pd
import wandb
import json
import os 
from pathlib import Path
from PycalcAct.train_function import *
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import shutil
import numpy as np
from scipy.spatial.distance import cdist

os.environ['WANDB_API_KEY'] = '73246a79f06da26fb325d763bd90ab7fc81bc9e6'
api = wandb.Api()

# Project is specified by <entity/project-name>
is_regression = False
if is_regression:
    runs = api.runs("sebthis-mcgill-university/my-first-sweep-regressor")
else:
    runs = api.runs("sebthis-mcgill-university/my-first-sweep-classifier",
                    filters = {"$or": [
                {"sweep": "1ln4tilj"},
                {"sweep": "67ey3j68"}]})

summary_list, config_list, name_list, acc_OTI,\
    distance_P14, distance_OT3, distance_OTI, distance_conc, distance_SL,\
    synthetic_EC50_P14, synthetic_EC50_OT3, synthetic_EC50_OTI,synthetic_EC50_conc,synthetic_EC50_SL\
    = [], [], [], [], [], [], [], [], [] , [] ,[],[], [],[]
tableFolder = getPath(is_regression, "")[1].parent
if not os.path.exists(tableFolder.joinpath("project.csv")):
    for i, run in enumerate(runs):
        print(i)     
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files

        summary = json.loads(run.summary._json_dict)
        if 'model_unique_name' not in summary.keys():
            print(f"Run {run.name} has no model_unique_name, skipping...")
            continue

        summary_list.append(summary)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config = {k: v['value'] for k,v in json.loads(run.config).items() if not k.startswith('_')}
        config_list.append(config)
            
        # .name is the human-readable name of th e run.
        name_list.append(run.name)

        model_unique_name = summary['model_unique_name']
        sweep_id =  run.sweep.id    
        _, saveFolder = getPath(is_regression, sweep_id)    
        thisModelPath = os.path.join(saveFolder, model_unique_name)

        if os.path.getsize(thisModelPath) == 0:
            print(f"Model: {model_unique_name} has no saved model.")
        else:
            if not is_regression:
                # load already calculated metrics
                if 'accuracy_test' in summary.keys():
                    acc_OTI.append(summary['accuracy_test'])
                else:
                    acc_OTI.append(summary['metric_test'])
                # compute synthetic EC50 from confusion matrix
                    # load confusion matrix
                OTI_conf = pd.read_csv(os.path.join(thisModelPath, "confusionMatrix.csv"), index_col=0, header = 0).to_numpy(dtype = float)
                OTI_conf = OTI_conf[-4:, :]
                OTI_conf_row_label = pd.read_csv(os.path.join(thisModelPath, "confusionMatrix.csv"), index_col=0, header = 0).index.to_list()
                OTI_conf_row_label = OTI_conf_row_label[-4:]
                P14_conf = pd.read_csv(os.path.join(thisModelPath, "confusionMatrix_P14.csv"), index_col=0, header = 0).to_numpy(dtype = float)
                P14_conf_row_label = pd.read_csv(os.path.join(thisModelPath, "confusionMatrix_P14.csv"), index_col=0, header = 0).index.to_list()
                OT3_conf = pd.read_csv(os.path.join(thisModelPath, "confusionMatrix_OT3.csv"), index_col=0, header = 0).to_numpy(dtype = float)
                OT3_conf_row_label = pd.read_csv(os.path.join(thisModelPath, "confusionMatrix_OT3.csv"), index_col=0, header = 0).index.to_list()
                conc_conf = pd.read_csv(os.path.join(thisModelPath, "confusionMatrix_conc.csv"), index_col=0, header = 0).to_numpy(dtype = float)
                conc_conf_row_label = pd.read_csv(os.path.join(thisModelPath, "confusionMatrix_conc.csv"), index_col=0, header = 0).index.to_list() 
                prediction_labels = pd.read_csv(os.path.join(thisModelPath, "confusionMatrix_OT3.csv"), index_col=0, header = 0).columns.to_list()  
                SL_conf = pd.read_csv(os.path.join(thisModelPath, "confusionMatrix_SL.csv"), index_col=0, header = 0).to_numpy(dtype = float)
                SL_conf_row_label = pd.read_csv(os.path.join(thisModelPath, "confusionMatrix_SL.csv"), index_col=0, header = 0).index.to_list()


                iterable = [(OTI_conf, OTI_conf_row_label, "OTI"),(P14_conf, P14_conf_row_label, "P14"),\
                            (OT3_conf,OT3_conf_row_label, "OT3"), (conc_conf, conc_conf_row_label, "conc"),\
                                (SL_conf, SL_conf_row_label, "SL")]
                
                EC50 = {
                    "N4" : -12.9,
                    "Q4" : -10.9,
                    "T4" : -9.5,
                    "Q4H7" : -8.9,
                    "M9" : -11.7,
                    "L6F" : -8.00,
                    "C9" : -8.04,
                    "OT3_N4" : -10.6,
                    "OT3_Q4" : -11.4,
                    -6 : -12.9,
                    -8 : -12.9,
                    -10 : -12.9,
                    -12 : -12.9,
                }


                row_ec50_dist = {}
                row_dist = {}
                row_ec50 = {}   
                ec50_dist = {}
                for conf, labels, name in iterable:
                        # do matrix multiplication of this row by EC50 values of the colomns labels
                    row_ec50.update({name:np.dot(conf, np.array([EC50[label] for label in prediction_labels]))/np.sum(conf, axis = 1)})
                    row_ec50_dist.update({name : np.abs([EC50[label] for label in labels] - row_ec50[name])})
                    ec50_dist.update({name:np.average(row_ec50_dist[name], weights = np.sum(conf, axis = 1))})
                    GT = np.array([EC50[v] for v in prediction_labels])
                    pred = np.array([EC50[v] for v in labels])
                    this_distance_matrix = cdist(pred.reshape(-1,1), GT.reshape(-1,1), metric='euclidean')
                    # transpose this distnace matrix to have shape (num labels, num prediction labels)
                        # do matrix multiplication of this row by EC50 values of the colomns labels
                    row_dist.update({name:np.sum(conf*this_distance_matrix)/np.sum(conf)})
                distance_OTI.append(row_dist['OTI'])
                distance_P14.append(row_dist['P14'])
                distance_OT3.append(row_dist['OT3'])
                distance_conc.append(row_dist['conc'])
                distance_SL.append(row_dist['SL'])
                synthetic_EC50_OTI.append(ec50_dist['OTI'])
                synthetic_EC50_P14.append(ec50_dist['P14'])
                synthetic_EC50_OT3.append(ec50_dist['OT3'])
                synthetic_EC50_conc.append(ec50_dist['conc'])
                synthetic_EC50_SL.append(ec50_dist['SL'])


    runs_df = pd.DataFrame({
        "name": name_list, 
        "model_unique_name": [summary['model_unique_name'] for summary in summary_list],
        "acc_OTI": acc_OTI,
        "distance_OTI": distance_OTI,
        "distance_P14": distance_P14,   
        "distance_OT3": distance_OT3,
        "distance_conc": distance_conc,
        "distance_SL": distance_SL,
        "synthetic_EC50_OTI": synthetic_EC50_OTI,
        "synthetic_EC50_P14": synthetic_EC50_P14,
        "synthetic_EC50_OT3": synthetic_EC50_OT3,
        "synthetic_EC50_conc": synthetic_EC50_conc,
        "synthetic_EC50_SL": synthetic_EC50_SL,
        })

    runs_df.to_csv(tableFolder.joinpath("project.csv"))
else:
    
    runs_df = pd.read_csv(tableFolder.joinpath("project.csv"), index_col=0)

if not is_regression:

    # plot all parameters
    # fig, ax = plt.subplots(1,5, figsize=(15,5))
    # plot_df = runs_df.copy()
    # ax[0].plot(plot_df['acc_OTI'], plot_df['distance_OTI'], 'o')
    # ax[0].set_xlabel("Normalized Accucracy OTI")
    # ax[0].set_ylabel("Normalized Distance OTI")
    # ax[1].plot(plot_df['distance_P14'], plot_df['distance_OT3'], 'o')
    # ax[1].set_xlabel("Normalized Distance P14")
    # ax[1].set_ylabel("Normalized Distance OT3")
    # ax[2].plot(plot_df['synthetic_EC50_OTI'], plot_df['synthetic_EC50_P14'], 'o')
    # ax[2].set_xlabel("Synthetic EC50 OTI")
    # ax[2].set_ylabel("Synthetic EC50 P14")
    # ax[3].plot(plot_df['synthetic_EC50_OTI'], plot_df['synthetic_EC50_OT3'], 'o')
    # ax[3].set_xlabel("Synthetic EC50 OTI") 
    # ax[3].set_ylabel("Synthetic EC50 OT3")
    # ax[4].plot(plot_df['distance_conc'], plot_df['synthetic_EC50_conc'], 'o')
    # ax[4].set_xlabel("Normalized Distance conc") 
    # ax[4].set_ylabel("Synthetic EC50 conc")
    # plt.suptitle("Model performances normalized")
    # plt.show()

    norm_df = runs_df.copy() 
    this_min = [0.5,    0.6,    1.7 ,   1.1,    0,      0,      0.35,   1.5,    0.2,    0,      0]
    this_max = [0.7,    1.1,    2,      1.5,    1.2,    1.5,    0.9,    2.80,   0.7,    1.2,    1.2] 
    this_percentile_1 = runs_df.iloc[:,2:].quantile(0.00, axis=0)
    for i, col in enumerate(runs_df.iloc[:,2:].columns):
        norm_df[col] = (runs_df[col] - this_min[i]) / (this_max[i] - this_min[i])
    for i, col in enumerate(runs_df.iloc[:,2:].columns):
        norm_df.loc[norm_df[col] > 1, :] = np.nan
        norm_df.loc[norm_df[col] < 0, :] = np.nan


    w_P14 = 1
    w_OT3 = 1
    w_OTI = 1
    norm_df['composite_score_accuracy'] = np.sqrt(w_OTI * (1-norm_df['acc_OTI'])**2 + w_P14 * (norm_df['distance_P14'])**2 + w_OT3 * (norm_df['distance_OT3'])**2)
    norm_df['composite_score_distance'] = np.sqrt(w_OTI * (norm_df['distance_OTI'])**2 + w_P14 * (norm_df['distance_P14'])**2 + w_OT3 * (norm_df['distance_OT3'])**2)
    norm_df['composite_score_syntheticEC50'] = np.sqrt(w_OTI * (norm_df['synthetic_EC50_OTI'])**2 + w_P14 * (norm_df['synthetic_EC50_P14'])**2 + w_OT3 * (norm_df['synthetic_EC50_OT3'])**2)

    accuracy_min_idx = norm_df['composite_score_accuracy'].idxmin()
    distance_min_idx = norm_df['composite_score_distance'].idxmin()
    synthetic_min_idx = norm_df['composite_score_syntheticEC50'].idxmin()
    
    best_model_accuracy =  pd.concat([
        runs_df.loc[accuracy_min_idx],
        norm_df.loc[accuracy_min_idx]], axis=1)
    best_model_distance =  pd.concat([
        runs_df.loc[distance_min_idx],
        norm_df.loc[distance_min_idx]], axis=1)
    best_model_syntheticEC50=  pd.concat([
        runs_df.loc[synthetic_min_idx],
        norm_df.loc[synthetic_min_idx]], axis=1)

    print(best_model_accuracy)
    print(best_model_distance)
    print(best_model_syntheticEC50)

    
# Save normalized dataframe
    norm_df.to_csv(Path(tableFolder).joinpath("project_normalized.csv"))

# copy best model folder to a new location
bestFolder = getPath(is_regression, "")[1].parent
# best model path with todays date
best_model_path = Path(bestFolder).joinpath("best_model", f"{pd.Timestamp.now().strftime('%y%m%d')}")
best_model_src = Path(bestFolder).joinpath("sweep_1ln4tilj", runs_df['model_unique_name'][distance_min_idx])
if not os.path.exists(best_model_src):
    best_model_src = Path(bestFolder).joinpath("sweep_67ey3j68", runs_df['model_unique_name'][distance_min_idx])
shutil.copytree(best_model_src, best_model_path, dirs_exist_ok=True)

#  write runs_df.loc[distance_min_idx] to a text file
with open(best_model_path.joinpath("model_info.txt"), "w") as f:
    f.write("Best model based on composite score (distance metrics):\n")
    f.write(str(best_model_distance))

EC50 = {
    "N4" : -12.9,
    "Q4" : -10.9,
    "T4" : -9.5,
    "Q4H7" : -8.9,
    "M9" : -11.7,
    "L6F" : -8.00,
    "C9" : -8.04,
    "OT3_N4" : -10.6,
    "OT3_Q4" : -11.4,
    -6 : -12.9,
    -8 : -12.9,
    -10 : -12.9,
    -12 : -12.9,
}

OTI_conf = pd.read_csv(os.path.join(best_model_path, "confusionMatrix.csv"), index_col=0, header = 0).to_numpy(dtype = float)
OTI_conf = OTI_conf[-4:, :]
OTI_conf_row_label = pd.read_csv(os.path.join(best_model_path, "confusionMatrix.csv"), index_col=0, header = 0).index.to_list()
OTI_conf_row_label = OTI_conf_row_label[-4:]
P14_conf = pd.read_csv(os.path.join(best_model_path, "confusionMatrix_P14.csv"), index_col=0, header = 0).to_numpy(dtype = float)
P14_conf_row_label = pd.read_csv(os.path.join(best_model_path, "confusionMatrix_P14.csv"), index_col=0, header = 0).index.to_list()
OT3_conf = pd.read_csv(os.path.join(best_model_path, "confusionMatrix_OT3.csv"), index_col=0, header = 0).to_numpy(dtype = float)
OT3_conf_row_label = pd.read_csv(os.path.join(best_model_path, "confusionMatrix_OT3.csv"), index_col=0, header = 0).index.to_list()
conc_conf = pd.read_csv(os.path.join(best_model_path, "confusionMatrix_conc.csv"), index_col=0, header = 0).to_numpy(dtype = float)
conc_conf_row_label = pd.read_csv(os.path.join(best_model_path, "confusionMatrix_conc.csv"), index_col=0, header = 0).index.to_list() 
prediction_labels = pd.read_csv(os.path.join(best_model_path, "confusionMatrix_OT3.csv"), index_col=0, header = 0).columns.to_list()  


iterable = [(OTI_conf, OTI_conf_row_label, "OTI"),(P14_conf, P14_conf_row_label, "P14"),\
            (OT3_conf,OT3_conf_row_label, "OT3"), (conc_conf, conc_conf_row_label, "conc")]

row_ec50_dist = {}
for conf, labels, name in iterable:
    GT = np.array([EC50[v] for v in prediction_labels])
    pred = np.array([EC50[v] for v in labels])
    this_distance_matrix = cdist(pred.reshape(-1,1), GT.reshape(-1,1), metric='euclidean').T
    # transpose this distnace matrix to have shape (num labels, num prediction labels)
        # do matrix multiplication of this row by EC50 values of the colomns labels
    row_ec50_dist.update({name:np.diag(np.dot(conf, this_distance_matrix))/np.sum(conf, axis = 1)})

# write to file
with open(best_model_path.joinpath("model_info.txt"), "a") as f:
    f.write("\n\nDetailed EC50 distances per class:\n")
    for _, labels, name in iterable :
        f.write(f"\n{name}:\n")
        for label, dist in zip(labels, row_ec50_dist[name]):
            f.write(f"Class {label}: EC50 distance = {dist}\n") 

thisConf_OTI = [[2228,561,401,160],[592,717,727,302],[252,258,652,286],[37,37,75,450]]
thisConf_P14 = [[461,181,117,72],[97,95,165,406],[359,317,595,809]]
thisConf_OT3 = [[789,763,862,421],[985,858,630,259]]
thisConf_conc = [[11543,2605,1963,803],[679,150,107,69],[439,124,211,163],[266,186,197,173]]

iterable = [(thisConf_OTI, prediction_labels, "OTI"),(thisConf_P14, P14_conf_row_label, "P14"), \
            (thisConf_OT3,OT3_conf_row_label, "OT3"), (thisConf_conc, ["N4", "N4", "N4", "N4"], "conc")]
row_ec50, row_ec50_dist, ec50_dist, row_dist = {}, {}, {}, {}   
for conf, labels, name in iterable:
        # do matrix multiplication of this row by EC50 values of the colomns labels
    row_ec50.update({name:np.dot(conf, np.array([EC50[label] for label in prediction_labels]))/np.sum(conf, axis = 1)})
    row_ec50_dist.update({name : np.abs([EC50[label] for label in labels] - row_ec50[name])})
    ec50_dist.update({name:np.average(row_ec50_dist[name], weights = np.sum(conf, axis = 1))})
    GT = np.array([EC50[v] for v in prediction_labels])
    pred = np.array([EC50[v] for v in labels])
    this_distance_matrix = cdist(pred.reshape(-1,1), GT.reshape(-1,1), metric='euclidean')
    # transpose this distnace matrix to have shape (num labels, num prediction labels)
        # do matrix multiplication of this row by EC50 values of the colomns labels
    row_dist.update({name:np.sum(conf*this_distance_matrix)/np.sum(conf)})

