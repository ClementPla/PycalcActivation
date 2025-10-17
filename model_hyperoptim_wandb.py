import wandb
import os
from PycalcAct.train_function import *

os.environ['WANDB_API_KEY'] = '73246a79f06da26fb325d763bd90ab7fc81bc9e6'
project_name = "my-first-sweep-regressor"
is_regression = True

def objective(config, is_regression):
    model_unique_name, trainer = setupTrainer(config, is_regression)
    n_epoch = 1000
    trainer.train(n_epoch)
    metric_train, metric_val, metric_test = save_model_perf(trainer, model_unique_name)
    return model_unique_name, metric_train, metric_val, metric_test

def main(project_name, is_regression):
    with wandb.init(project=project_name) as run:
        model_unique_name, metric_train, metric_val, metric_test = objective(run.config, is_regression)
        run.log({"metric_train": metric_train,
                 "metric_test": metric_test,
                 "metric_val": metric_val,
                 "model_unique_name": model_unique_name})
        
sweep_configuration = {
    "method": "bayes",
    "metric": {
        "goal": "maximize", 
        "name": "metric_test"},
    "parameters": {
        "whichDataset" : {"values": ["ratio", "ratioNorm", "indiv"]},
        "whichDisplacement" : {"values": [True, False]},
        "replace_nan_by_min" : {"values": [True, False]},
        "remove_mean": {"values": [True, False]},
        "whichFFT" : {"values": [True, False]},
        "numRNN" : {"values": [1, 2, 3]},
        "sizeRNN": {"values": [8, 16, 32, 64]},
        "bidir" : {"values": [True, False]},
        "numFC": {"values": [1, 2, 3]},
        "sizeFC" :  {"values": [8, 16, 32, 64]},
        "dropout" : {"min": 0.1, "max": 0.30},
        "weighted" : {"values": [True, False]},
        "customLoss" : {"values": [False]},
        "xyDisplacement" : {"values": [True, False]},
        "initial_lr": {"values": [0.01, 0.001, 0.0001]},
        "weight_decay": {"values": [1e-5, 1e-4, 1e-3]},
        "batch_size": {"values": [128, 256, 215, 1024, 2048, 4096]},
        # "augment_gt" : {"values": [True, False]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

wandb.agent(sweep_id, function=lambda: main(project_name = project_name , is_regression = is_regression), count=1000)