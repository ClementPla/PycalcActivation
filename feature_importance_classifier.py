from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor 
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from PycalcAct.dataset_v2 import Dataset
from PycalcAct.model import (
    MixedFCTemporalModel)
from PycalcAct.trainer_v2 import Trainer
from PycalcAct.train_function import *


## read project_config and projec_normalized file
df_values = pd.read_csv('models/round3/classifier/project_normalized.csv')
df_values = df_values.rename(columns={'Unnamed: 0': 'model_id'})
df_config = pd.read_csv('models/round3/classifier/project_config.csv')
df_config = df_config.rename(columns={'Unnamed: 0': 'model_id'})
df = pd.merge(pd.DataFrame(df_values.iloc[:,[0,25]]), pd.DataFrame(df_config), on = 'model_id')

# replace nan with 'no disp' in whichDisplacement
df['whichDisplacement'] = df['whichDisplacement'].fillna("['no disp']")
categorical_cols = ['bidir', 'replace_nan_by_min', 'weighted', 'whichFFT']
multicat_cols = ['whichDataset', 'whichDisplacement']

df_encoded = df.copy().drop(columns=['model_id', 'store_best', 'customLoss', 'remove_mean', 'myLoss'])
for col in categorical_cols:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

for col in multicat_cols:
    df_encoded = pd.merge(df_encoded, pd.get_dummies(df_encoded[col],drop_first=False, dtype=int), left_index=True, right_index=True, how='outer').drop(columns=[col])


X = df_encoded.drop(columns=['composite_score_distance'])
y = df_encoded['composite_score_distance']

model_gradient = GradientBoostingRegressor()
model_forest = RandomForestRegressor()
model_gradient.fit(X, y)
model_forest.fit(X, y)

importance_gradient = pd.Series(model_gradient.feature_importances_, index=X.columns).sort_values(ascending=False)
importance_forest = pd.Series(model_forest.feature_importances_, index=X.columns).sort_values(ascending=False)

corr_matrix = df_encoded.corr(method='pearson')
corr_to_score = corr_matrix.iloc[0,:]

final_importance = pd.merge(pd.DataFrame(importance_gradient).reset_index().rename(columns={'index':'feature',0:'importance_gradient'}), \
                            pd.DataFrame(importance_forest).reset_index().rename(columns={'index':'feature',0:'importance_forest'}), on = 'feature').merge(\
                            pd.DataFrame(corr_to_score).reset_index().rename(columns={'index':'feature','composite_score_distance':'corr_to_score'}), on='feature')

# split corr to score into absolute and sign
final_importance['corr_to_score_sign'] = final_importance['corr_to_score'].apply(lambda x: 'positive' if x>=0 else 'negative')
final_importance['corr_to_score_abs'] = final_importance['corr_to_score'].abs()


final_importance.to_csv('models/round3/classifier/best_model/251211/feature_importance.csv', index=False)



## test importance of mean, order, and 'smoothness'

is_regression = True
# import best model
config = {
    'whichDataset': "ratioNorm",
    'whichDisplacement': 'None',
    'replace_nan_by_min': False,
    'whichFFT': False,
    'numRNN': 2,
    'sizeRNN': 32,
    'bidir': True,
    'numFC': 2,
    'sizeFC': 32,
    'dropout': 0.0691649489571053,
    'weighted': True,
    'initial_lr': 0.01,
    'weight_decay': 0.001,
    'batch_size': 128,
    'customLoss': False,
    'remove_mean': False,
    'loss' : nan,
}
EC50_path = Path('EC50.csv')
dataFolder, saveFolder = getPath(False, "")   
model_unique_name = "best_model_test_mean_order_smoothness"
thisPath = saveFolder.joinpath(model_unique_name)
thisPath.mkdir(parents=True, exist_ok=True)
        
    # Setup Dataset

match config['whichDataset']:
    case "ratio":
        csv_path=[dataFolder.joinpath("legend.csv"), dataFolder.joinpath("calciumRatio.csv")]          
    case "ratioNorm":
        csv_path=[dataFolder.joinpath("legend.csv"), dataFolder.joinpath("calciumRatio_normalized.csv")] 
    case "indiv":
        csv_path=[dataFolder.joinpath("legend.csv"), dataFolder.joinpath("calciumFree.csv"), dataFolder.joinpath("calciumBound.csv")] 

if config['whichDisplacement'] == "displacement":
    csv_pos_path = dataFolder.joinpath("position.csv")
    position_to_displacement = True
elif config['whichDisplacement'] == "xyPosition":
    csv_pos_path = dataFolder.joinpath("position.csv")
    position_to_displacement = False            
else:
    csv_pos_path = None
    position_to_displacement = False

EC50 = pd.read_csv(EC50_path, index_col=None , header=None)
EC50 = {
    (row.iloc[0]): row.iloc[1]
    for _, (_, row) in enumerate(EC50.iterrows())
}


df = pd.DataFrame()

for i in range(25):
    print(i)
    dataset_regular = Dataset(
        csv_path = csv_path,csv_pos_path=csv_pos_path,  # Optional
        position_to_displacement=position_to_displacement,
        remove_mean=config['remove_mean'],
        replace_nan_by_min=config['replace_nan_by_min'],
        is_regression=is_regression, #True,
        EC50 = EC50,
        # Convert the x, y position to a single displacement value (sqrt((x(t+1)-x(t))^2 + (y(t+1)-y(t))^2)
        )
    dataset_smooth = Dataset(
        csv_path = csv_path,csv_pos_path=csv_pos_path,  # Optional
        position_to_displacement=position_to_displacement,
        remove_mean=config['remove_mean'],
        replace_nan_by_min=config['replace_nan_by_min'],
        is_regression=is_regression, #True,
        EC50 = EC50,
        smooth_time=True,
        # Convert the x, y position to a single displacement value (sqrt((x(t+1)-x(t))^2 + (y(t+1)-y(t))^2)
        )
    dataset_scramble_mean = Dataset(
        csv_path = csv_path,csv_pos_path=csv_pos_path,  # Optional
        position_to_displacement=position_to_displacement,
        remove_mean=config['remove_mean'],
        replace_nan_by_min=config['replace_nan_by_min'],
        is_regression=is_regression, #True,
        EC50 = EC50,
        scramble_mean=True,
        # Convert the x, y position to a single displacement value (sqrt((x(t+1)-x(t))^2 + (y(t+1)-y(t))^2)
        )
    dataset_scramble_time = Dataset(
        csv_path = csv_path,csv_pos_path=csv_pos_path,  # Optional
        position_to_displacement=position_to_displacement,
        remove_mean=config['remove_mean'],
        replace_nan_by_min=config['replace_nan_by_min'],
        is_regression=is_regression, #True,
        EC50 = EC50,
        scramble_time=True,
        # Convert the x, y position to a single displacement value (sqrt((x(t+1)-x(t))^2 + (y(t+1)-y(t))^2)
        )

    # load best model
    model = MixedFCTemporalModel(
        n_classes=4,
        n_rnn_layers=config['numRNN'],
        rnn_hidden_size=config['sizeRNN'],
        n_fc_layers=config['numFC'],
        fc_hidden_size=config['sizeFC'],
        temporal_length=dataset_regular.length_serie["OTI"],
        input_size=dataset_regular.features,
        bidirectional=config['bidir'],
        pooling=None,
        dropout=config['dropout']
    )
    checkpoint = torch.load("models/round3/classifier/best_model/251211/savedModel.pt", weights_only=True)
    model.load_state_dict(checkpoint["best"])
    model.to("cuda")
    del checkpoint

    # make trainers
    trainer_regular = Trainer(
        dataset_regular,
        model,
        device="cuda",
        batch_size=config['batch_size'],
        criterion= None,
        store_best= 'Accuracy',
        use_class_weights = config['weighted'],
        is_regression = is_regression,
        regression_bounds_y = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
        regression_bounds_y_pred = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
        initial_lr=config['initial_lr'],
        weight_decay=config['weight_decay'],
    ) 
    trainer_smooth = Trainer(
        dataset_smooth,
        model,
        device="cuda",
        batch_size=config['batch_size'],
        criterion= None,
        store_best= 'Accuracy',
        use_class_weights = config['weighted'],
        is_regression = is_regression,
        regression_bounds_y = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
        regression_bounds_y_pred = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
        initial_lr=config['initial_lr'],
        weight_decay=config['weight_decay'],
    ) 
    trainer_scramble_mean = Trainer(
        dataset_scramble_mean,
        model,
        device="cuda",
        batch_size=config['batch_size'],
        criterion= None,
        store_best= 'Accuracy',
        use_class_weights = config['weighted'],
        is_regression = is_regression,
        regression_bounds_y = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
        regression_bounds_y_pred = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
        initial_lr=config['initial_lr'],
        weight_decay=config['weight_decay'],
    ) 
    trainer_scramble_time = Trainer(
        dataset_scramble_time,
        model,
        device="cuda",
        batch_size=config['batch_size'],
        criterion= None,
        store_best= 'Accuracy',
        use_class_weights = config['weighted'],
        is_regression = is_regression,
        regression_bounds_y = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
        regression_bounds_y_pred = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
        initial_lr=config['initial_lr'],
        weight_decay=config['weight_decay'],
    ) 

    # make predictions
    tableFolder = Path("models/round3/classifier")
    limits = pd.read_csv(tableFolder.joinpath("limits.csv"))
    this_min = limits['this_min'].values
    this_max = limits['this_max'].values
    all_distance = {}
    all_distance_norm = {}
    EC50_dist = {}
    for name_trainer, trainer in zip(["regular", "smooth", "scramble_mean", "scramble_time"], \
                                    [trainer_regular, trainer_smooth, trainer_scramble_mean, trainer_scramble_time]):
        callbacks = (
            trainer.dataset.test_batch,
            trainer.dataset.P14_batch,
            trainer.dataset.OT3_batch,
        )
        dist = {}
        norm_dist = {}
        for name, callable, norm_col in zip(["OTI", "P14", "OT3"], \
                                            callbacks, \
                                            [1,2,3]):
            x, y = callable(True, to_cuda=True)   
            all_classes = np.unique(y.cpu().numpy())

            # best model
            trainer.load_best()     
            _, metrics = trainer.eval(x, y.type(torch.LongTensor).to(trainer.device))
            conf = np.array(trainer.confmat.compute().cpu(), dtype = int)
            # remove empty rows and columns from confusion matrix
            conf = conf[all_classes,:]
        
            conf_row_label = trainer.dataset.labels(name)
            prediction_labels = trainer.dataset.labels("OTI")
            pd.DataFrame(conf, index=conf_row_label, columns=prediction_labels).to_csv(thisPath.joinpath(f'confusion_matrix_{name_trainer}_{name}.csv'))

            GT = np.array([EC50[v] for v in prediction_labels])
            pred = np.array([EC50[v] for v in conf_row_label])
            this_distance_matrix = cdist(pred.reshape(-1,1), GT.reshape(-1,1), metric='euclidean')
            # transpose this distnace matrix to have shape (num labels, num prediction labels)
                # do matrix multiplication of this row by EC50 values of the colomns labels
            dist.update({name:np.sum(conf*this_distance_matrix)/np.sum(conf)})
            norm_dist.update({name:(dist[name]-this_min[norm_col])/(this_max[norm_col]-this_min[norm_col])})
            
        norm_dist.update({'overall': np.sqrt((norm_dist['OTI'])**2 + (norm_dist['P14'])**2 + (norm_dist['OT3'])**2)})        
        all_distance.update({name_trainer: dist})
        all_distance_norm.update({name_trainer: norm_dist})
        this_df = pd.DataFrame.from_dict(all_distance_norm).stack()

    df = pd.concat([df, this_df.rename(f'distance_iter_{i}')], axis=1)
df.to_csv(thisPath.joinpath('metrics.csv'))

print(df.mean(axis=1))