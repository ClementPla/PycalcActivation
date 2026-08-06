import pandas as pd
import wandb
import json
import os 
from pathlib import Path
from scipy import stats
from PycalcAct.train_function import *
import matplotlib
matplotlib.use('TkAgg') 
import shutil
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

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
                {"sweep": "67ey3j68"},
                {"sweep": "t9x8pdxp"}]})

EC50 = pd.read_csv("EC50.csv", index_col=None , header=None)
EC50 = {
    (row.iloc[0] if pos < 9 else int(row.iloc[0])): row.iloc[1]
    for pos, (_, row) in enumerate(EC50.iterrows())
}

summary_list, config_list, name_list, acc_OTI,\
    distance_P14, distance_OT3, distance_OTI, distance_conc, distance_SL,\
    synthetic_EC50_P14, synthetic_EC50_OT3, synthetic_EC50_OTI,synthetic_EC50_conc,synthetic_EC50_SL, \
    SD_P14, SD_OT3, SD_OTI, SD_conc, SD_SL, \
        pearson_corr_OTI, pearson_corr_P14, pearson_corr_OT3, pearson_corr_conc, pearson_corr_SL\
    = [], [], [], [], [], [], [], [], [] , [] ,[],[], [],[], [], [], [], [], [], [], [], [], [], []
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

            row_ec50_dist = {}
            row_SD = {}
            mean_SD = {}
            dist = {}
            row_ec50 = {}   
            ec50_dist = {}
            pearson_corr = {}
            for conf, labels, name in iterable:
                # Synthetic EC50 on each row
                this_apl_ec50 = np.array([EC50[label] for label in prediction_labels])
                row_ec50.update({name:np.dot(conf, this_apl_ec50)/np.sum(conf, axis = 1)})
                row_ec50_dist.update({name : np.abs([EC50[label] for label in labels] - row_ec50[name])})
                ec50_dist.update({name:np.average(row_ec50_dist[name], weights = np.sum(conf, axis = 1))})

                # distance metric
                GT = np.array([EC50[v] for v in prediction_labels])
                pred = np.array([EC50[v] for v in labels])
                this_distance_matrix = cdist(pred.reshape(-1,1), GT.reshape(-1,1), metric='euclidean')
                # transpose this distnace matrix to have shape (num labels, num prediction labels)
                    # do matrix multiplication of this row by EC50 values of the colomns labels
                dist.update({name:np.sum(conf*this_distance_matrix)/np.sum(conf)})

                # Metric to measure spread of the predictions
                row_SD.update({name:np.max(conf/np.sum(conf, axis = 1).reshape(-1,1), axis = 1)})
                mean_SD.update({name:np.average(row_SD[name])})

                # pearson correlation between predicted and actual EC50 values
                true_idx = np.repeat([EC50[label] for label in labels], len(prediction_labels))
                pred_idx = np.tile([EC50[label] for label in prediction_labels], len(labels))
                counts = conf.flatten().astype(int)
                y_true = np.repeat(true_idx, counts)
                y_pred = np.repeat(pred_idx, counts)

                if name != "conc":  # conc has some classes with zero counts leading to constant arrays
                    pearson, _= stats.pearsonr(y_true, y_pred)
                    pearson_corr.update({name:pearson})
                else:   
                    pearson_corr.update({name:np.nan})

            distance_OTI.append(dist['OTI'])
            distance_P14.append(dist['P14'])
            distance_OT3.append(dist['OT3'])
            distance_conc.append(dist['conc'])
            distance_SL.append(dist['SL'])
            synthetic_EC50_OTI.append(ec50_dist['OTI'])
            synthetic_EC50_P14.append(ec50_dist['P14'])
            synthetic_EC50_OT3.append(ec50_dist['OT3'])
            synthetic_EC50_conc.append(ec50_dist['conc'])
            synthetic_EC50_SL.append(ec50_dist['SL'])
            SD_P14.append(mean_SD['P14'])
            SD_OT3.append(mean_SD['OT3'])   
            SD_OTI.append(mean_SD['OTI'])
            SD_conc.append(mean_SD['conc'])
            SD_SL.append(mean_SD['SL'])
            pearson_corr_OTI.append(pearson_corr['OTI'])
            pearson_corr_P14.append(pearson_corr['P14'])
            pearson_corr_OT3.append(pearson_corr['OT3'])
            pearson_corr_conc.append(pearson_corr['conc'])
            pearson_corr_SL.append(pearson_corr['SL'])



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
        "SD_OTI": SD_OTI,
        "SD_P14": SD_P14,
        "SD_OT3": SD_OT3,
        "SD_conc": SD_conc,
        "SD_SL": SD_SL,
        "pearson_corr_OTI": pearson_corr_OTI,
        "pearson_corr_P14": pearson_corr_P14,
        "pearson_corr_OT3": pearson_corr_OT3,
        "pearson_corr_conc": pearson_corr_conc,
        "pearson_corr_SL": pearson_corr_SL,    
        })
    
    config_list_df = pd.DataFrame(config_list)

    runs_df.to_csv(tableFolder.joinpath("project.csv"))
    config_list_df.to_csv(tableFolder.joinpath("project_config.csv"))
else:
    runs_df = pd.read_csv(tableFolder.joinpath("project.csv"), index_col=0)
    config_list_df = pd.read_csv(tableFolder.joinpath("project_config.csv"), index_col=0)

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



# find max and mix value for all metrics to normalize between 0 and 1

OTI_best = np.identity(4) * np.array([3383., 2328.,  561., 1563.]).reshape(-1,1)
OTI_worst = np.array([[0,0,1,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]) * np.array([3383., 2328.,  561., 1563.]).reshape(-1,1)
P14_best = np.array([[0,0,1,0],[0,0,1,0],[0,1,0,0]])* np.array([2080.,  763.,  831.]).reshape(-1,1)
P14_worst = np.array([[1,0,0,0],[1,0,0,0],[0,0,1,0]])* np.array([2080.,  763.,  831.]).reshape(-1,1)
OT3_best = np.array([[0,1,0,0],[0,1,0,0]]) * np.array([3975., 3940.]).reshape(-1,1)
OT3_worst = np.array([[1,0,0,0],[1,0,0,0]]) * np.array([3975., 3940.]).reshape(-1,1)
conc_best = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]) * np.array([  937.,   822., 16914.,  1005.]).reshape(-1,1)
conc_worst = np.array([[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]])* np.array([  937.,   822., 16914.,  1005.]).reshape(-1,1)
SL_best = np.identity(4) * np.array([9640., 7804., 3427., 4405.]).reshape(-1,1)
SL_worst = np.array([[0,0,1,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]])  * np.array([9640., 7804., 3427., 4405.]).reshape(-1,1)
OTI_conf_row_label = ['N4', 'Q4', 'Q4H7', 'T4']
P14_conf_row_label = ['C9', 'L6F', 'M9']
OT3_conf_row_label = ['OT3_N4', 'OT3_Q4']
conc_conf_row_label = [-10, -12, -6, -8]
SL_conf_row_label = ['N4', 'Q4', 'Q4H7', 'T4']
prediction_labels = ['N4', 'Q4', 'Q4H7', 'T4']

iterable_ideal = [(OTI_best, OTI_conf_row_label, "OTI_best"),(P14_best, P14_conf_row_label, "P14_best"),\
            (OT3_best,OT3_conf_row_label, "OT3_best"), (conc_best, conc_conf_row_label, "conc_best"),\
                (SL_best, SL_conf_row_label, "SL_best"), (OTI_worst, OTI_conf_row_label, "OTI_worst"),(P14_worst, P14_conf_row_label, "P14_worst"),\
            (OT3_worst,OT3_conf_row_label, "OT3_worst"), (conc_worst, conc_conf_row_label, "conc_worst"),\
                (SL_worst, SL_conf_row_label, "SL_worst")]

row_ec50_dist_ideal = {}
row_ec50_ideal = {}   
ec50_dist_ideal = {}

row_dist_ideal= {}
dist_ideal = {}
for conf, labels, name in iterable_ideal:
        # do matrix multiplication of this row by EC50 values of the colomns labels

    row_ec50_ideal.update({name:np.dot(conf, np.array([EC50[label] for label in prediction_labels]))/np.sum(conf, axis = 1)})
    row_ec50_dist_ideal.update({name : np.abs([EC50[label] for label in labels] - row_ec50_ideal[name])})
    ec50_dist_ideal.update({name:np.average(row_ec50_dist_ideal[name], weights = np.sum(conf, axis = 1))})

    GT = np.array([EC50[v] for v in prediction_labels])
    pred = np.array([EC50[v] for v in labels])
    this_distance_matrix = cdist(pred.reshape(-1,1), GT.reshape(-1,1), metric='euclidean')
    # print(this_distance_matrix)
    # transpose this distnace matrix to have shape (num labels, num prediction labels)
        # do matrix multiplication of this row by EC50 values of the colomns labels
    dist_ideal.update({name:np.sum(conf*this_distance_matrix)/np.sum(conf)})
    row_dist_ideal.update({name:np.sum(conf*this_distance_matrix, axis=1)/np.sum(conf, axis=1)})



norm_df = runs_df.copy() 
this_min = [0.5, dist_ideal['OTI_best'], dist_ideal['P14_best'], dist_ideal['OT3_best'], dist_ideal['conc_best'], dist_ideal['SL_best'], \
            ec50_dist_ideal['OTI_best'], ec50_dist_ideal['P14_best'], ec50_dist_ideal['OT3_best'], ec50_dist_ideal['conc_best'], ec50_dist_ideal['SL_best'], 
            0.25,0.25,0.25,0.25,0.25, -1, -1, -1, -1, -1]
this_max = [0.7, dist_ideal['OTI_worst'], dist_ideal['P14_worst'], dist_ideal['OT3_worst'], dist_ideal['conc_worst'], dist_ideal['SL_worst'], \
            ec50_dist_ideal['OTI_worst'], ec50_dist_ideal['P14_worst'], ec50_dist_ideal['OT3_worst'], ec50_dist_ideal['conc_worst'], ec50_dist_ideal['SL_worst'], \
                1,1,1,1,1, 1, 1, 1, 1, 1]
# this_min_row_synthetic_ec50 = [v for k,v in row_ec50_dist_best.items() if 'best' in k]
# this_max_row_synthetic_ec50 = [v for k,v in row_ec50_dist_best.items() if 'worst' in k]
this_min_row_distance = [v for k,v in row_dist_ideal.items() if 'best' in k]
this_max_row_distance = [v for k,v in row_dist_ideal.items() if 'worst' in k]
# this_min = [0.5,    0.6,    1.7 ,   1.1,    0,      0,      0.35,   1.5,    0.2,    0,      0]
# this_max = [0.7,    1.1,    2,      1.5,    1.2,    1.5,    0.9,    2.80,   0.7,    1.2,    1.2] 
# this_percentile_1 = runs_df.iloc[:,2:].quantile(0.00, axis=0)

# write this min and co to file
limit_df = pd.DataFrame({
    "this_min": this_min,
    "this_max": this_max})
limit_row_df = pd.DataFrame({
    "this_min_row_distance": this_min_row_distance,
    "this_max_row_distance": this_max_row_distance,
    })
limit_df.to_csv(Path(tableFolder).joinpath("limits.csv"))
limit_row_df.to_csv(Path(tableFolder).joinpath("limits_row.csv"))

for i, col in enumerate(runs_df.iloc[:,2:].columns):
    norm_df[col] = (runs_df[col] - this_min[i]) / (this_max[i] - this_min[i])
# for i, col in enumerate(runs_df.iloc[:,2:].columns):
#     norm_df.loc[norm_df[col] > 1, :] = np.nan
#     norm_df.loc[norm_df[col] < 0, :] = np.nan


w_P14 = 10
w_OT3 = 1
w_OTI = 1
norm_df['composite_score_accuracy'] = np.sqrt(w_OTI * (1-norm_df['acc_OTI'])**2 + w_P14 * (norm_df['distance_P14'])**2 + w_OT3 * (norm_df['distance_OT3'])**2)
norm_df['composite_score_distance'] = np.sqrt(w_OTI * (norm_df['distance_OTI'])**2 + w_P14 * (norm_df['distance_P14'])**2 + w_OT3 * (norm_df['distance_OT3'])**2)
norm_df['composite_score_syntheticEC50'] = np.sqrt(w_OTI * (norm_df['synthetic_EC50_OTI'])**2 + w_P14 * (norm_df['synthetic_EC50_P14'])**2 + w_OT3 * (norm_df['synthetic_EC50_OT3'])**2)
norm_df['composite_score_distance_withSD'] = np.sqrt(w_OTI * (norm_df['distance_OTI'])**2 + w_P14 * (norm_df['distance_P14'])**2 + w_OT3 * (norm_df['distance_OT3'])**2 + \
                                                        w_OTI * (norm_df['SD_OTI'])**2 + w_P14 * (norm_df['SD_P14'])**2 + w_OT3 * (norm_df['SD_OT3'])**2)
norm_df['composite_score_syntheticEC50_withSD'] = np.sqrt(w_OTI * (norm_df['synthetic_EC50_OTI'])**2 + w_P14 * (norm_df['synthetic_EC50_P14'])**2 + w_OT3 * (norm_df['synthetic_EC50_OT3'])**2 + \
                                                        (w_OTI * (norm_df['SD_OTI'])**2 + w_P14 * (norm_df['SD_P14'])**2 + w_OT3 * (norm_df['SD_OT3'])**2))
norm_df['composite_score_pearson'] = np.sqrt(w_OTI * (1 - norm_df['pearson_corr_OTI'])**2 + w_P14 * (1 - norm_df['pearson_corr_P14'])**2 + w_OT3 * (1 - norm_df['pearson_corr_OT3'])**2)
norm_df['composite_score_pearson_SD'] = np.sqrt(w_OTI * (1 - norm_df['pearson_corr_OTI'])**2 + w_P14 * (1 - norm_df['pearson_corr_P14'])**2 + w_OT3 * (1 - norm_df['pearson_corr_OT3'])**2 + \
                                                        w_OTI * (norm_df['SD_OTI'])**2 + w_P14 * (norm_df['SD_P14'])**2 + w_OT3 * (norm_df['SD_OT3'])**2)

accuracy_min_idx = norm_df['composite_score_accuracy'].idxmin()
distance_min_idx = norm_df['composite_score_distance'].idxmin()
synthetic_min_idx = norm_df['composite_score_syntheticEC50'].idxmin()
distance_SD_min_idx = norm_df['composite_score_distance_withSD'].idxmin()
synthetic_SD_min_idx = norm_df['composite_score_syntheticEC50_withSD'].idxmin()
pearson_min_idx = norm_df['composite_score_pearson'].idxmin()
pearson_SD_min_idx = norm_df['composite_score_pearson_SD'].idxmin()   

best_model_accuracy =  pd.concat([
    runs_df.loc[accuracy_min_idx],
    norm_df.loc[accuracy_min_idx]], axis=1)
best_model_distance =  pd.concat([
    runs_df.loc[distance_min_idx],
    norm_df.loc[distance_min_idx]], axis=1)
best_model_syntheticEC50=  pd.concat([
    runs_df.loc[synthetic_min_idx],
    norm_df.loc[synthetic_min_idx]], axis=1)
best_model_distance_SD =  pd.concat([
    runs_df.loc[distance_SD_min_idx],
    norm_df.loc[distance_SD_min_idx]], axis=1)
best_model_syntheticEC50_SD =  pd.concat([
    runs_df.loc[synthetic_SD_min_idx],
    norm_df.loc[synthetic_SD_min_idx]], axis=1)
best_model_pearson =  pd.concat([
    runs_df.loc[pearson_min_idx],
    norm_df.loc[pearson_min_idx]], axis=1)
best_model_pearson_SD   =  pd.concat([
    runs_df.loc[pearson_SD_min_idx],
    norm_df.loc[pearson_SD_min_idx]], axis=1)


print(best_model_accuracy)
print(best_model_distance)
print(best_model_syntheticEC50)
print(best_model_distance_SD)   
print(best_model_syntheticEC50_SD)
print(best_model_pearson)
print(best_model_pearson_SD)

# Save normalized dataframe
norm_df.to_csv(Path(tableFolder).joinpath("project_normalized.csv"))

# copy best model folder to a new location
bestFolder = getPath(is_regression, "")[1].parent
# best model path with todays date
best_model_path = Path(bestFolder).joinpath("best_model", "251231") #f"{pd.Timestamp.now().strftime('%y%m%d')}"
best_model_src = Path(bestFolder).joinpath("sweep_1ln4tilj", runs_df['model_unique_name'][distance_min_idx])
if not os.path.exists(best_model_src):
    best_model_src = Path(bestFolder).joinpath("sweep_67ey3j68", runs_df['model_unique_name'][distance_min_idx])
if not os.path.exists(best_model_src):
    best_model_src = Path(bestFolder).joinpath("sweep_t9x8pdxp", runs_df['model_unique_name'][distance_min_idx])
shutil.copytree(best_model_src, best_model_path, dirs_exist_ok=True)

####################################################### Change this #########################################################
best_run = api.run("/sebthis-mcgill-university/my-first-sweep-classifier/runs/m41h1zkc")
#############################################################################################################################


best_config = best_run.config
pd.DataFrame(data = best_config, index=[0]).to_csv(best_model_path.joinpath("wandb_run_info.csv"))
#  write runs_df.loc[synthetic_min_idx] to a text file
with open(best_model_path.joinpath("model_info.txt"), "w") as f:
    f.write("Best model based on composite score (synthetic EC50 metrics):\n")
    f.write(str(best_model_distance))

# compute detailed EC50 distances per class and write to file
OTI_conf_best = pd.read_csv(os.path.join(best_model_path, "confusionMatrix.csv"), index_col=0, header = 0).to_numpy(dtype = float)
OTI_conf = OTI_conf_best[-4:, :]
OTI_conf_row_label = pd.read_csv(os.path.join(best_model_path, "confusionMatrix.csv"), index_col=0, header = 0).index.to_list()
OTI_conf_row_label = OTI_conf_row_label[-4:]
P14_conf_best = pd.read_csv(os.path.join(best_model_path, "confusionMatrix_P14.csv"), index_col=0, header = 0).to_numpy(dtype = float)
P14_conf_row_label = pd.read_csv(os.path.join(best_model_path, "confusionMatrix_P14.csv"), index_col=0, header = 0).index.to_list()
OT3_conf_best = pd.read_csv(os.path.join(best_model_path, "confusionMatrix_OT3.csv"), index_col=0, header = 0).to_numpy(dtype = float)
OT3_conf_row_label = pd.read_csv(os.path.join(best_model_path, "confusionMatrix_OT3.csv"), index_col=0, header = 0).index.to_list()
conc_conf_best = pd.read_csv(os.path.join(best_model_path, "confusionMatrix_conc.csv"), index_col=0, header = 0).to_numpy(dtype = float)
conc_conf_row_label = pd.read_csv(os.path.join(best_model_path, "confusionMatrix_conc.csv"), index_col=0, header = 0).index.to_list() 
prediction_labels = pd.read_csv(os.path.join(best_model_path, "confusionMatrix_OT3.csv"), index_col=0, header = 0).columns.to_list()  


iterable_best = [(OTI_conf_best[-4:, :], OTI_conf_row_label, "OTI",0),(P14_conf_best, P14_conf_row_label, "P14",1),\
            (OT3_conf_best,OT3_conf_row_label, "OT3",2), (conc_conf_best, conc_conf_row_label, "conc",3)]

def confusion_to_lists(conf_matrix, x_labels=None, y_labels=None):
    y_true = []
    y_pred = []

    n_classes_x = conf_matrix.shape[1]
    n_classes_y = conf_matrix.shape[0]

    if x_labels is None:
        x_labels = list(range(n_classes_x))
    if y_labels is None:
        y_labels = list(range(n_classes_y))

    for true_label in range(n_classes_y):
        for pred_label in range(n_classes_x):
            count = conf_matrix[true_label, pred_label]
            y_true.extend([y_labels[true_label]] * count)
            y_pred.extend([x_labels[pred_label]] * count)

    return np.array(y_true), np.array(y_pred)

# row_ec50_dist = {}
# row_ec50_dist_norm = {}
row_ec50_best = {}
# ec50_dist = {}
# ec50_dist_norm = {}
row_dist_best = {}
row_dist_best_norm = {}
allPred = np.empty((0, 1))
allGT = np.empty((0, 1))
sperman_corr_df = {}
for conf, labels, name, norm_col_row in iterable_best:
    this_apl_ec50 = np.array([EC50[label] for label in prediction_labels])
    row_ec50_best.update({name:np.dot(conf, this_apl_ec50)/np.sum(conf, axis = 1)})
    # row_ec50_dist.update({name : np.abs([EC50[label] for label in labels] - row_ec50[name])})
    # row_ec50_dist_norm.update({name : np.abs((row_ec50_dist[name] - this_min_row_synthetic_ec50[norm_col_row]) / \
    #                                           (this_max_row_synthetic_ec50[norm_col_row] - this_min_row_synthetic_ec50[norm_col_row]))})
    # ec50_dist.update({name:np.average(row_ec50_dist[name], weights = np.sum(conf, axis = 1))})
    # ec50_dist_norm.update({name:np.abs((ec50_dist[name] - this_min[norm_col]) / (this_max[norm_col] - this_min[norm_col]))})
    GT = np.array([EC50[v] for v in prediction_labels])
    pred = np.array([EC50[v] for v in labels])
    this_distance_matrix = cdist(pred.reshape(-1,1), GT.reshape(-1,1), metric='euclidean')
    # transpose this distnace matrix to have shape (num labels, num prediction labels)
        # do matrix multiplication of this row by EC50 values of the colomns labels
    row_dist_best.update({name:np.sum(conf*this_distance_matrix, axis=1)/np.sum(conf, axis=1 )})
    row_dist_best_norm.update({name:np.abs((row_dist_best[name] - this_min_row_distance[norm_col_row]) / \
                                      (this_max_row_distance[norm_col_row] - this_min_row_distance[norm_col_row]))})

    y_true, y_pred = confusion_to_lists(conf[1:, 1:].astype(int), x_labels=prediction_labels, y_labels=labels)
    y_true = np.array([EC50[v] for v in y_true])
    y_pred = np.array([EC50[v] for v in y_pred])
    spearman_corr, _ = spearmanr(y_true, y_pred)
    allPred = np.concatenate((allPred, y_pred.reshape(-1, 1)), axis=0)
    allGT = np.concatenate((allGT, y_true.reshape(-1, 1)), axis=0)
    spearman_corr_df[name] = spearman_corr
spearman_corr_df["all"],_ = spearmanr(allGT, allPred)
allPred_df = pd.concat([pd.DataFrame(allPred, columns=['Pred']), pd.DataFrame(allGT, columns=['GT'])], axis=1)
allPred_df.to_csv(best_model_path.joinpath("all_predictions.csv"), index=False)
spearman_corr_df = pd.DataFrame.from_dict(spearman_corr_df, orient='index', columns=['spearman_corr'])
spearman_corr_df.to_csv(best_model_path.joinpath("all_spearmancorr.csv"))

# write to file
with open(best_model_path.joinpath("model_info.txt"), "a") as f:
    f.write("\n\nDetailed synthetic EC50 distances per class:\n")
    for _, labels, name, _ in iterable_best :
        f.write(f"\n{name}:\n")
        for label, dist, dist_norm, mean_ec50 in zip(labels, row_dist_best[name], row_dist_best_norm[name], row_ec50_best[name]):
            f.write(f"Class {label}: synthetic EC50 dist = {dist} \n\
                            normalized = {dist_norm}\n \
                            average EC50 = {mean_ec50}\n")


## Extracted parameter model performance
thisConf_OTI_extracted = [[2228,561,401,160],[592,717,727,302],[252,258,652,286],[37,37,75,450]]
thisConf_P14_extracted = [[461,181,117,72],[97,95,165,406],[359,317,595,809]]
thisConf_OT3_extracted = [[789,763,862,421],[985,858,630,259]]
thisConf_conc_extracted = [[11543,2605,1963,803],[679,150,107,69],[439,124,211,163],[266,186,197,173]]

iterable_extracted = [(thisConf_OTI_extracted, prediction_labels, "OTI", 1,0),(thisConf_P14_extracted, P14_conf_row_label, "P14",2,1), \
            (thisConf_OT3_extracted,OT3_conf_row_label, "OT3",3,2), (thisConf_conc_extracted, ["N4", "N4", "N4", "N4"], "conc",4,3)]
dist_extracted, dist_extracted_norm, row_dist_extracted, row_dist_extracted_norm ,row_ec50_extracted  = {}, {}, {}, {}, {}
for conf, labels, name, norm_col, norm_col_row in iterable_extracted:
    #     # do matrix multiplication of this row by EC50 values of the colomns labels
    row_ec50_extracted.update({name:np.dot(conf, np.array([EC50[label] for label in prediction_labels]))/np.sum(conf, axis = 1)})
    # row_ec50_dist.update({name : np.abs([EC50[label] for label in labels] - row_ec50[name])})
    # ec50_dist.update({name:np.average(row_ec50_dist[name], weights = np.sum(conf, axis = 1))})
    GT = np.array([EC50[v] for v in prediction_labels])
    pred = np.array([EC50[v] for v in labels])
    this_distance_matrix = cdist(pred.reshape(-1,1), GT.reshape(-1,1), metric='euclidean')
    # transpose this distnace matrix to have shape (num labels, num prediction labels)
        # do matrix multiplication of this row by EC50 values of the colomns labels
    dist_extracted.update({name:np.sum(conf*this_distance_matrix)/np.sum(conf)})
    row_dist_extracted.update({name:np.sum(conf*this_distance_matrix, axis=1)/np.sum(conf, axis=1)})
    dist_extracted_norm.update({name:np.abs((dist_extracted[name] - this_min[norm_col]) / \
                                      (this_max[norm_col] - this_min[norm_col]))})
    row_dist_extracted_norm.update({name:np.abs((row_dist_extracted[name] - this_min_row_distance[norm_col_row]) / \
                                      (this_max_row_distance[norm_col_row] - this_min_row_distance[norm_col_row]))})

# normalize based on this_max and this_min


with open(best_model_path.joinpath("model_info.txt"), "a") as f:
    f.write("\n\nExtracted parameter model performance:\n")
    for _,_,name,_,_ in iterable_extracted :
        f.write(f"\n{name}:\n")
        f.write(f"Overall distance metric = {dist_extracted[name]}\n") 
        f.write(f"Normalized distance metric = {dist_extracted_norm[name]}\n")

with open(best_model_path.joinpath("model_info.txt"), "a") as f:
    f.write("\n\nDetailed EC50 distances per class for extracted parameter model performance:\n")
    for _, labels, name,_,_ in iterable_extracted :
        f.write(f"\n{name}:\n")
        for label, dist, dist_norm, mean_ec50 in zip(labels, row_dist_extracted[name], row_dist_extracted_norm[name], row_ec50_extracted[name]):
            f.write(f"Class {label}: EC50 distance = {dist} \n \
                         normalized = {dist_norm} \n \
                         average EC50 = {mean_ec50}\n") 
            

## retrain model with a fraction of SL dataset
model_unique_name = "best_model_retrained_SL_fraction"
# load SL dataset and split into train and test
config = best_run.config
sweep_id = ""
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
store_best = None
loss = None
customFilter = None

dataFolder = Path("D:/Ca2-Analysis_McGill/prediction/agAffinity/datasets/dataset_mcgill_SL")
saveFolder = Path("D:/sebastien/PycalcActivation/models/round3/classifier/sweep_")
    
     # create model folder
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
# Setup Dataset
dataset = Dataset(
    csv_path = csv_path,
    csv_pos_path=csv_pos_path,
    EC50 = EC50,  # Optional
    position_to_displacement=position_to_displacement,
    remove_mean=remove_mean,
    replace_nan_by_min=replace_nan_by_min,
    customFilter = customFilter, 
    is_regression=is_regression, 
    test_size = 0.6,
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
    regression_bounds_y = torch.Tensor([-12,-10,-9]).to("cuda") if is_regression else None,
    regression_bounds_y_pred = torch.Tensor([-12,-10,-9]).to("cuda") if is_regression else None,
    initial_lr=initial_lr,
    weight_decay=weight_decay,
    loss = loss,
) 


# load best model weights
model_load_path = best_model_path.joinpath("savedModel.pt")
checkpoint = torch.load(model_load_path, weights_only=True)
trainer.model.load_state_dict(checkpoint["best"])
trainer.model.to("cuda")
del checkpoint
n_epoch = 1000
val_patience = 250
trainer.train(n_epoch, val_patience = val_patience)
metric_train, metric_val, metric_test, _, _ = save_model_perf(trainer, model_unique_name, sweep_id, save = True)
metrics = save_model_generalizability(trainer, model_unique_name, sweep_id, save = True)

_, save_folder = getPath(is_regression, sweep_id)
save_folder = save_folder.joinpath("best_model_retrained_SL_fraction")
OTI_conf_retrained = pd.read_csv(os.path.join(save_folder, "confusionMatrix.csv"), index_col=0, header = 0).to_numpy(dtype = float)
OTI_conf = OTI_conf_retrained[-4:, :]
OTI_conf_row_label = pd.read_csv(os.path.join(save_folder, "confusionMatrix.csv"), index_col=0, header = 0).index.to_list()
OTI_conf_row_label = OTI_conf_row_label[-4:]
prediction_labels = pd.read_csv(os.path.join(save_folder, "confusionMatrix_OT3.csv"), index_col=0, header = 0).columns.to_list()  

iterable_retrained = [(thisConf_OTI_extracted, prediction_labels, "OTI", 1,0)]
dist_retrained, dist_retrained_norm, row_dist_retrained, row_dist_retrained_norm ,row_ec50_retrained  = {}, {}, {}, {}, {}
for conf, labels, name, norm_col, norm_col_row in iterable_retrained:
    #     # do matrix multiplication of this row by EC50 values of the colomns labels
    row_ec50_retrained.update({name:np.dot(conf, np.array([EC50[label] for label in prediction_labels]))/np.sum(conf, axis = 1)})
    # row_ec50_dist.update({name : np.abs([EC50[label] for label in labels] - row_ec50[name])})
    # ec50_dist.update({name:np.average(row_ec50_dist[name], weights = np.sum(conf, axis = 1))})
    GT = np.array([EC50[v] for v in prediction_labels])
    pred = np.array([EC50[v] for v in labels])
    this_distance_matrix = cdist(pred.reshape(-1,1), GT.reshape(-1,1), metric='euclidean')
    # transpose this distnace matrix to have shape (num labels, num prediction labels)
        # do matrix multiplication of this row by EC50 values of the colomns labels
    dist_retrained.update({name:np.sum(conf*this_distance_matrix)/np.sum(conf)})
    row_dist_retrained.update({name:np.sum(conf*this_distance_matrix, axis=1)/np.sum(conf, axis=1)})
    dist_retrained_norm.update({name:np.abs((dist_retrained[name] - this_min[norm_col]) / \
                                      (this_max[norm_col] - this_min[norm_col]))})
    row_dist_retrained_norm.update({name:np.abs((row_dist_retrained[name] - this_min_row_distance[norm_col_row]) / \
                                      (this_max_row_distance[norm_col_row] - this_min_row_distance[norm_col_row]))})

# normalize based on this_max and this_min


with open(save_folder.joinpath("model_info.txt"), "a") as f:
    f.write("\n\nExtracted parameter model performance:\n")
    for _,_,name,_,_ in iterable_retrained :
        f.write(f"\n{name}:\n")
        f.write(f"Overall distance metric = {dist_retrained[name]}\n") 
        f.write(f"Normalized distance metric = {dist_retrained_norm[name]}\n")

with open(save_folder.joinpath("model_info.txt"), "a") as f:
    f.write("\n\nDetailed EC50 distances per class for extracted parameter model performance:\n")
    for _, labels, name,_,_ in iterable_retrained :
        f.write(f"\n{name}:\n")
        for label, dist, dist_norm, mean_ec50 in zip(labels, row_dist_retrained[name], row_dist_retrained_norm[name], row_ec50_retrained[name]):
            f.write(f"Class {label}: EC50 distance = {dist} \n \
                         normalized = {dist_norm} \n \
                         average EC50 = {mean_ec50}\n") 





## Investigate how migration and FFT data influences the different metrics
unique_migration = config_list_df['whichDisplacement'].unique()
mig_norm_df = {}
DF = pd.DataFrame(columns=["Condition", "Group", "Value"])
for mig in unique_migration:
    mig_norm_df[str(mig)] = norm_df[config_list_df['whichDisplacement'] == mig if not pd.isna(mig) else config_list_df['whichDisplacement'].isna()]
    mig_norm_df_melt = mig_norm_df[str(mig)].iloc[:,3:7].melt()
    temp = pd.DataFrame({
        "Value": mig_norm_df_melt.value.values,
        "Condition": mig_norm_df_melt.variable.values,
        "Group": [str(mig)]*len(mig_norm_df_melt)
    })
    DF = pd.concat([DF, temp], ignore_index=True)
    
# # Summary statistics per Condition x Group
# sum_df = DF.groupby(["Condition", "Group"], as_index=False).agg(
#     Mean=("Value", "mean"),
#     N=("Value", "size"),
#     SD=("Value", "std"),
# )
# sum_df["SEM"] = sum_df["SD"] / np.sqrt(sum_df["N"])


# # Ordering
# conditions = sorted(sum_df["Condition"].unique().tolist())
# groups = sorted(sum_df["Group"].unique().tolist())
# cond_pos = np.arange(len(conditions))
# dodge = 0.20               # half-width for group separation
# jitter = 0.12              # horizontal jitter magnitude
# colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(groups)))
# color_map = dict(zip(groups, colors))

# fig, ax = plt.subplots(figsize=(8, 5))

# # Scatter individual points with manual dodge & jitter
# for i, cond in enumerate(conditions):
#     sub = DF[DF["Condition"] == cond]
#     for j, grp in enumerate(groups):
#         subg = sub[sub["Group"] == grp]["Value"].values
#         if subg.size == 0:
#             continue
#         x_center = cond_pos[i] + (j - (len(groups)-1)) * (dodge)
#         x = x_center + np.random.uniform(-jitter, jitter, size=subg.size)
#         ax.scatter(x, subg, s=45, alpha=0.85, color=color_map[grp], edgecolor="black")

# # Overlay mean ± SEM
# for i, cond in enumerate(conditions):
#     for j, grp in enumerate(groups):
#         row = sum_df[(sum_df["Condition"] == cond) & (sum_df["Group"] == grp)]
#         if row.empty:
#             continue
#         m, sem = float(row["Mean"]), float(row["SEM"])
#         x_center = cond_pos[i] + (j - (len(groups)-1)) * (dodge)
#         ax.errorbar(x_center, m, yerr=sem, fmt="_", elinewidth=1.6, capsize=6,
#                     color='black', zorder=10)
#         ax.scatter(x_center, m, s=70, color='black', edgecolor="black", zorder=10)

# # Cosmetics / labels
# ax.set_xticks(cond_pos)
# ax.set_xticklabels(conditions, rotation=0)
# ax.set_title("Grouped dot plot with mean ± SEM (Prism style)")
# ax.set_ylabel("Value")
# ax.set_xlabel("")
# ax.grid(True, axis="y", alpha=0.3)

# # Legend
# handles = [plt.Line2D([0],[0], marker='o', color='w', label=g,
#                       markerfacecolor=color_map[g], markeredgecolor='black',
#                       markersize=9) for g in groups]
# ax.legend(handles=handles, title="Group", frameon=False, loc="upper right")

# plt.tight_layout()
# plt.show()

DF.to_csv(best_model_path.joinpath("migration_effects_on_metrics_data.csv"))
