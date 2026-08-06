from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor 
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from PycalcAct.dataset_v2 import Dataset
from PycalcAct.model import MixedFCTemporalModel
from PycalcAct.trainer_v2 import Trainer
from PycalcAct.train_function import *
from scipy.ndimage import gaussian_filter1d


## test importance of mean, order, and 'smoothness'
is_regression = True
# import best model
config = {
    'whichDataset': "ratioNorm",
    'whichDisplacement': 'None',
    'replace_nan_by_min': False,
    'whichFFT': False,
    'numRNN': 1,
    'sizeRNN': 16,
    'bidir': True,
    'numFC': 3,
    'sizeFC': 8,
    'dropout': 0.1503081582126325,
    'weighted': True,
    'initial_lr': 0.01,
    'weight_decay': 0.001,
    'batch_size': 256,
    'customLoss': False,
    'remove_mean': False,
    'loss' : None,
    'store_best': 'myFScore',
    'myLoss': 'MSE'
}
EC50_path = Path('EC50.csv')
dataFolder, saveFolder = getPath(is_regression, "")   
model_unique_name = "best_model_test_mean_order_smoothness_retrained"
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


################################  Test different features #############################################
# df = pd.DataFrame()
# for i in range(1):
#     print(i)
#     dataset_regular = Dataset(
#         csv_path = csv_path,csv_pos_path=csv_pos_path,  # Optional
#         position_to_displacement=position_to_displacement,
#         remove_mean=config['remove_mean'],
#         replace_nan_by_min=config['replace_nan_by_min'],
#         is_regression=is_regression, #True,
#         EC50 = EC50,
#         # Convert the x, y position to a single displacement value (sqrt((x(t+1)-x(t))^2 + (y(t+1)-y(t))^2)
#         )
#     dataset_smooth = Dataset(
#         csv_path = csv_path,csv_pos_path=csv_pos_path,  # Optional
#         position_to_displacement=position_to_displacement,
#         remove_mean=config['remove_mean'],
#         replace_nan_by_min=config['replace_nan_by_min'],
#         is_regression=is_regression, #True,
#         EC50 = EC50,
#         smooth_time=True,
#         # Convert the x, y position to a single displacement value (sqrt((x(t+1)-x(t))^2 + (y(t+1)-y(t))^2)
#         )
#     dataset_scramble_mean = Dataset(
#         csv_path = csv_path,csv_pos_path=csv_pos_path,  # Optional
#         position_to_displacement=position_to_displacement,
#         remove_mean=config['remove_mean'],
#         replace_nan_by_min=config['replace_nan_by_min'],
#         is_regression=is_regression, #True,
#         EC50 = EC50,
#         scramble_mean=True,
#         # Convert the x, y position to a single displacement value (sqrt((x(t+1)-x(t))^2 + (y(t+1)-y(t))^2)
#         )
#     dataset_scramble_time = Dataset(
#         csv_path = csv_path,csv_pos_path=csv_pos_path,  # Optional
#         position_to_displacement=position_to_displacement,
#         remove_mean=config['remove_mean'],
#         replace_nan_by_min=config['replace_nan_by_min'],
#         is_regression=is_regression, #True,
#         EC50 = EC50,
#         scramble_time=True,
#         # Convert the x, y position to a single displacement value (sqrt((x(t+1)-x(t))^2 + (y(t+1)-y(t))^2)
#         )

#     # load best model
#     model = MixedFCTemporalModel(
#         n_classes=4 if not is_regression else 1,
#         n_rnn_layers=config['numRNN'],
#         rnn_hidden_size=config['sizeRNN'],
#         n_fc_layers=config['numFC'],
#         fc_hidden_size=config['sizeFC'],
#         temporal_length=dataset_regular.length_serie["OTI"],
#         input_size=dataset_regular.features,
#         bidirectional=config['bidir'],
#         pooling=None,
#         dropout=config['dropout'] if config['numRNN'] > 1 else 0
#     )
#     checkpoint = torch.load("models\\round3\\regressor\\best_model\\251230\\savedModel.pt", weights_only=True)
#     model.load_state_dict(checkpoint["best"])
#     model.to("cuda")
#     del checkpoint

#     # make trainers    
#     trainer_regular = Trainer(
#         dataset_regular,
#         model,
#         device="cuda",
#         batch_size=config['batch_size'],
#         criterion= None,
#         store_best= config['store_best'],
#         use_class_weights = config['weighted'],
#         is_regression = is_regression,
#         regression_bounds_y = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
#         regression_bounds_y_pred = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
#         initial_lr=config['initial_lr'],
#         weight_decay=config['weight_decay'],
#         loss=config['myLoss'],
#     ) 
#     trainer_smooth = Trainer(
#         dataset_smooth,
#         model,
#         device="cuda",
#         batch_size=config['batch_size'],
#         criterion= None,
#         store_best= config['store_best'],
#         use_class_weights = config['weighted'],
#         is_regression = is_regression,
#         regression_bounds_y = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
#         regression_bounds_y_pred = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
#         initial_lr=config['initial_lr'],
#         weight_decay=config['weight_decay'],
#         loss=config['myLoss'],
#     ) 
#     trainer_scramble_mean = Trainer(
#         dataset_scramble_mean,
#         model,
#         device="cuda",
#         batch_size=config['batch_size'],
#         criterion= None,
#         store_best= config['store_best'],
#         use_class_weights = config['weighted'],
#         is_regression = is_regression,
#         regression_bounds_y = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
#         regression_bounds_y_pred = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
#         initial_lr=config['initial_lr'],
#         weight_decay=config['weight_decay'],
#         loss=config['myLoss'],
#     ) 
#     trainer_scramble_time = Trainer(
#         dataset_scramble_time,
#         model,
#         device="cuda",
#         batch_size=config['batch_size'],
#         criterion= None,
#         store_best= config['store_best'],
#         use_class_weights = config['weighted'],
#         is_regression = is_regression,
#         regression_bounds_y = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
#         regression_bounds_y_pred = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
#         initial_lr=config['initial_lr'],
#         weight_decay=config['weight_decay'],
#         loss=config['myLoss'],
#     ) 

#     # make predictions
#     all_distance = {}
#     for name_trainer, trainer in zip(["regular", "smooth", "scramble_mean", "scramble_time"], \
#                                     [trainer_regular, trainer_smooth, trainer_scramble_mean, trainer_scramble_time]):
#         x, y = trainer.dataset.train_batch(True, to_cuda=True)
#         y = y.cpu().detach().numpy().squeeze()
#         y_pred = trainer.model(x).squeeze().cpu().detach().numpy()

#         # find weird repeated value in y_pred and remove from x and y
#         vals, counts = np.unique(y_pred, return_counts=True)

#         if np.any(counts > 500) and len(counts) > 50:
#             weird_value = np.sort(vals[counts > 500])
#             print("Weird repeated value found in predictions, removing from analysis.")
#         else:
#             weird_value = []

#         callbacks = (
#             trainer.dataset.test_batch,
#             trainer.dataset.P14_batch,
#             trainer.dataset.OT3_batch,
#         )

#         perf_metric = {}

#         for name, callable in zip(["OTI", "P14", "OT3"], \
#                                             callbacks):
#             x, y = callable(True, to_cuda=True)   
#             all_classes = np.unique(y.cpu().numpy())

#             # best model
#             trainer.load_best()
#             y_pred = trainer.model(x).squeeze().cpu().detach().numpy()
#             y = y.cpu().detach().numpy()

#             for v in weird_value:
#                     y = y[y_pred != v]
#                     x = x[y_pred != v,:,:]
#                     y_encoded = y_encoded[y_pred != v]
#                     if name == "conc":
#                         y_ec50 = y_ec50[y_pred != v]
#                     y_pred = y_pred[y_pred != v]
            
#             fig, ax = plt.subplots()
#             ax.hist(y_pred, bins=50, alpha=0.5, label='Predicted')
#             ax.hist(y, bins=50, alpha=0.5, label='Actual')
#             ax.legend()
#             plt.savefig(thisPath.joinpath(f"pred_vs_actual_{name_trainer}_{name}.png"))
#             plt.close(fig)
            
#             perf_metric.update({name:np.mean(np.abs(y - y_pred))})
            
#         all_distance.update({name_trainer: perf_metric})
#         this_df = pd.DataFrame.from_dict(all_distance).stack()

#     df = pd.concat([df, this_df.rename(f'distance_iter_{i}')], axis=1)
#     print(df)
# df.to_csv(thisPath.joinpath('metrics.csv'))

# print(df.mean(axis=1))


################## Test smooth time kernel size for prediction ####################
# df = pd.DataFrame()
# for i in np.linspace(0, 30, 121):
#     print(i)
#     dataset_smooth = Dataset(
#         csv_path = csv_path,csv_pos_path=csv_pos_path,  # Optional
#         position_to_displacement=position_to_displacement,
#         remove_mean=config['remove_mean'],
#         replace_nan_by_min=config['replace_nan_by_min'],
#         is_regression=is_regression, #True,
#         EC50 = EC50,
#         smooth_time=i,
#         # Convert the x, y position to a single displacement value (sqrt((x(t+1)-x(t))^2 + (y(t+1)-y(t))^2)
#         )

#     # load best model
#     model = MixedFCTemporalModel(
#         n_classes=4 if not is_regression else 1,
#         n_rnn_layers=config['numRNN'],
#         rnn_hidden_size=config['sizeRNN'],
#         n_fc_layers=config['numFC'],
#         fc_hidden_size=config['sizeFC'],
#         temporal_length=dataset_smooth.length_serie["OTI"],
#         input_size=dataset_smooth.features,
#         bidirectional=config['bidir'],
#         pooling=None,
#         dropout=config['dropout'] if config['numRNN'] > 1 else 0
#     )
#     checkpoint = torch.load("models\\round3\\regressor\\best_model\\251230\\savedModel.pt", weights_only=True)
#     model.load_state_dict(checkpoint["best"])
#     model.to("cuda")
#     del checkpoint

#     # make trainers    
#     trainer_smooth = Trainer(
#         dataset_smooth,
#         model,
#         device="cuda",
#         batch_size=config['batch_size'],
#         criterion= None,
#         store_best= config['store_best'],
#         use_class_weights = config['weighted'],
#         is_regression = is_regression,
#         regression_bounds_y = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
#         regression_bounds_y_pred = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
#         initial_lr=config['initial_lr'],
#         weight_decay=config['weight_decay'],
#         loss=config['myLoss'],
#     ) 
    

#     # make predictions
#     all_distance = {}
#     x, y = trainer_smooth.dataset.train_batch(True, to_cuda=True)
#     y = y.cpu().detach().numpy().squeeze()
#     y_pred = trainer_smooth.model(x).squeeze().cpu().detach().numpy()

#     # find weird repeated value in y_pred and remove from x and y
#     vals, counts = np.unique(y_pred, return_counts=True)

#     if np.any(counts > 500) and len(counts) > 50:
#         weird_value = np.sort(vals[counts > 500])
#         print("Weird repeated value found in predictions, removing from analysis.")
#     else:
#         weird_value = []

#     callbacks = (
#         dataset_smooth.test_batch,
#         dataset_smooth.P14_batch,
#         dataset_smooth.OT3_batch,
#     )

#     perf_metric = {}

#     for name, callable in zip(["OTI", "P14", "OT3"], \
#                                         callbacks):
#         x, y = callable(True, to_cuda=True)   
#         all_classes = np.unique(y.cpu().numpy())

#         # best model
#         trainer_smooth.load_best()
#         y_pred = trainer_smooth.model(x).squeeze().cpu().detach().numpy()
#         y = y.cpu().detach().numpy()

#         for v in weird_value:
#                 y = y[y_pred != v]
#                 x = x[y_pred != v,:,:]
#                 y_encoded = y_encoded[y_pred != v]
#                 if name == "conc":
#                     y_ec50 = y_ec50[y_pred != v]
#                 y_pred = y_pred[y_pred != v]
        
#         # fig, ax = plt.subplots()
#         # ax.hist(y_pred, bins=50, alpha=0.5, label='Predicted')
#         # ax.hist(y, bins=50, alpha=0.5, label='Actual')
#         # ax.legend()
#         # plt.savefig(thisPath.joinpath(f"pred_vs_actual_{name_trainer}_{name}.png"))
#         # plt.close(fig)
        
#         perf_metric.update({name:np.mean(np.abs(y - y_pred))})
        
#     all_distance.update({"smooth": perf_metric})
#     this_df = pd.DataFrame.from_dict(all_distance).stack()

#     df = pd.concat([df, this_df.rename(f'{i}')], axis=1)
# df.to_csv(thisPath.joinpath('smoothness_kernel_metrics.csv'))

# print(df.mean(axis=1))


################## Test smooth time kernel size by retraining ####################
df = pd.DataFrame()
for i in np.linspace(0, 30, 121):
    print(i)
    thisModelPath = thisPath.joinpath(f'smooth_time_{i}')
    thisModelPath.mkdir(parents=True, exist_ok=True)

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
    store_best = config['store_best']
    myLoss = config['myLoss']
    customFilter = None
        
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

    EC50 = pd.read_csv(EC50_path, index_col=None , header=None)
    EC50 = {
        (row.iloc[0]): row.iloc[1]
        for _, (_, row) in enumerate(EC50.iterrows())
    }
    
    dataset = Dataset(
        csv_path = csv_path,
        csv_pos_path=csv_pos_path,  # Optional
        position_to_displacement=position_to_displacement,
        remove_mean=remove_mean,
        replace_nan_by_min=replace_nan_by_min,
        customFilter = customFilter, 
        is_regression=is_regression, #True,
        EC50 = EC50,
        smooth_time=i if i > 0 else False,
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
    
    if customLoss and not is_regression:
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
        regression_bounds_y = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
        regression_bounds_y_pred = torch.Tensor([-13,-10,-9]).to("cuda") if is_regression else None,
        initial_lr=initial_lr,
        weight_decay=weight_decay,
        loss = myLoss,
    ) 
    
    # n_epoch = 10000
    # val_patience = 250
    # trainer.train(n_epoch, val_patience = val_patience)
    # metrics = save_model_generalizability(trainer, f"best_model_test_mean_order_smoothness/smooth_time_{i}", "", save = True, EC50_path = "D:\sebastien\PycalcActivation\EC50.csv")
    # torch.save(
    #         {
    #             "model": trainer.model.state_dict(),
    #             "optim": trainer.optim.state_dict(),
    #             "scheduler": trainer.scheduler.state_dict() if trainer.scheduler else None,
    #             "best": trainer._best_state_dict,
    #             "last": trainer._last_state_dict,
    #         },
    #         Path(thisModelPath.joinpath("model.pt")),
    #     )


    checkpoint = torch.load(thisModelPath.joinpath("model.pt"), weights_only=True)
    trainer.model.load_state_dict(checkpoint["best"])
    trainer.model.to("cuda")
    del checkpoint
    # make predictions

    all_distance = {}
    x, y = dataset.train_batch(True, to_cuda=True)
    x = x.cpu().detach().numpy()
    j = 0
    for row in x:
        if i>0:
            row[:] = gaussian_filter1d(row, sigma = i, axis=0)
    x = torch.tensor(x).to("cuda")
    y = y.cpu().detach().numpy().squeeze()
    y_pred = trainer.model(x).squeeze().cpu().detach().numpy()

    # find weird repeated value in y_pred and remove from x and y
    vals, counts = np.unique(y_pred, return_counts=True)

    if np.any(counts > 500) and len(counts) > 50:
        weird_value = np.sort(vals[counts > 500])
        print("Weird repeated value found in predictions, removing from analysis.")
    else:
        weird_value = []

    callbacks = (
        dataset.test_batch,
        dataset.P14_batch,
        dataset.OT3_batch,
    )

    perf_metric = {}

    for name, callable in zip(["OTI", "P14", "OT3"], \
                                        callbacks):
        x, y = callable(True, to_cuda=True)   

        # SMOOTH OUT with same kernel size
        x = x.cpu().detach().numpy()
        for row in x:
            if i>0:
                row[:] = gaussian_filter1d(row, sigma = i, axis=0)
        x = torch.tensor(x).to("cuda")
        # best model
        y_pred = trainer.model(x).squeeze().cpu().detach().numpy()
        y = y.cpu().detach().numpy()

        for v in weird_value:
                y = y[y_pred != v]
                x = x[y_pred != v,:,:]
                y_pred = y_pred[y_pred != v]
        
        fig, ax = plt.subplots()
        ax.hist(y_pred, bins=50, alpha=0.5, label='Predicted')
        ax.hist(y, bins=50, alpha=0.5, label='Actual')
        ax.legend()
        plt.savefig(thisModelPath.joinpath(f"pred_vs_actual_{name}.png"))
        plt.close(fig)
        
        perf_metric.update({name:np.mean(np.abs(y - y_pred))})
        
    all_distance.update({"smooth": perf_metric})
    this_df = pd.DataFrame.from_dict(all_distance).stack()

    df = pd.concat([df, this_df.rename(f'{i}')], axis=1)
    print(df)
df.to_csv(thisPath.joinpath('smoothness_kernel_metrics_retrained_predSmooth.csv'))

print(df.mean(axis=1))