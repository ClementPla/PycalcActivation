
import wandb
import os
from PycalcAct.train_function import *
import torch
import matplotlib.pyplot as plt

os.environ['WANDB_API_KEY'] = '73246a79f06da26fb325d763bd90ab7fc81bc9e6'

is_regression = True
if is_regression:
    project_name = "my-first-sweep-regressor"
else:
    project_name = "my-first-sweep-classifier"

def objective(config, is_regression, sweep_id):
    model_unique_name, trainer = setupTrainer(config, is_regression, sweep_id, model_unique_name=None, EC50_path = "D:\sebastien\PycalcActivation\EC50.csv")
    n_epoch = 10000
    val_patience = 150
    trainer.train(n_epoch, val_patience = val_patience)
    metric_train, metric_val, metric_test, _, _ = save_model_perf(trainer, model_unique_name, sweep_id, save = True)
    metrics = save_model_generalizability(trainer, model_unique_name, sweep_id, save = True, EC50_path = "D:\sebastien\PycalcActivation\EC50.csv")
    if not is_regression:
        metric_to_optimize = np.sqrt(((metric_test.detach().clone().cpu()-0)/(5.08-0))**2 + \
                                10*((metrics['distance_P14']-0.76)/(5.99-0.76))**2 + \
                                10*((metrics['distance_OT3']-0.35)/(3.75-0.35))**2) # minimize
         # average accuracy of all datasets
    else:
        metric_to_optimize = metric_test  #maximize
    torch.cuda.empty_cache()    
    del trainer
    plt.close('all')

    return model_unique_name, metric_train, metric_val, metric_test, metrics, metric_to_optimize

def main(project_name, is_regression, sweep_id):
    with wandb.init(project=project_name) as run:
        model_unique_name, metric_train, metric_val, metric_test, metrics, metric_to_optimize = objective(run.config, is_regression, sweep_id)
        if is_regression:
            log_dict = {"fScore_train": metric_train,
                        "fScore_test": metric_test,
                        "fScore_val": metric_val, 
                        "metric_to_optimize": metric_to_optimize,
                        "model_unique_name": model_unique_name}
        else:
            log_dict = { "accuracy_train": metric_train,
                        "accuracy_test": metric_test,
                        "accuracy_val": metric_val, 
                        "metric_to_optimize": metric_to_optimize,
                        "model_unique_name": model_unique_name}
        log_dict.update(metrics)
        run.log(log_dict)
        
sweep_configuration = {
    "method": "random" if is_regression else "bayes",# "bayes","random"
    "metric": {
        "goal": "maximize" if is_regression else "minimize", # "maximize", "minimize"
        "name": "metric_to_optimize"},
    "parameters": {
        "whichDataset" : {"values": ["ratio", "ratioNorm", "indiv"]},
        "whichDisplacement" : {"values": ["displacement", "xyPosition", "None"]},
        "replace_nan_by_min" : {"values": [True, False]},
        "remove_mean": {"values": [True, False]},
        "whichFFT" : {"values": [True, False]},
        "numRNN" : {"values": [1, 2, 3]if not is_regression else [1, 2]},
        "sizeRNN": {"values": [8, 16, 32, 64] if not is_regression else [4, 8, 16, 32]},
        "bidir" : {"values": [True, False]},
        "numFC": {"values": [1, 2, 3] if not is_regression else [1, 2]},
        "sizeFC" :  {"values": [8, 16, 32, 64] if not is_regression else [4, 8, 16, 32]},
        "dropout" : {"min": 0.05, "max": 0.20},
        "weighted" : {"values": [True, False]},
        "customLoss" : {"values": [False]},
        "initial_lr": {"values": [0.01, 0.001]},
        "weight_decay": {"values": [1e-5, 1e-4, 1e-3]},
        "batch_size": {"values": [128, 256, 215, 1024, 2048, 4096]},
        "store_best": {"values": ["myFScore", "Accuracy", "mySpearman"] if is_regression else ["Accuracy"]},
        "myLoss": {"values": ["MSE", "L1", "Huber"] if is_regression else [None]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

wandb.agent(sweep_id, function=lambda: main(project_name = project_name , \
                                            is_regression = is_regression, sweep_id = sweep_id), count=5000)