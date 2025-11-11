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
    model_unique_name, trainer = setupTrainer(config, is_regression, sweep_id, model_unique_name=None)
    n_epoch = 1000
    val_patience = 250
    trainer.train(n_epoch, val_patience = val_patience)
    metric_train, metric_val, metric_test, _, _ = save_model_perf(trainer, model_unique_name, sweep_id, save = True)
    metrics = save_model_generalizability(trainer, model_unique_name, sweep_id, save = True)
    if not is_regression:
        metric_to_max = np.sqrt(((metrics['metric_test']-0.5)/0.2)**2 + \
                                ((metrics['distance_P14']-1.7)/0.3)**2 + \
                                ((metrics['distance_OT3']-1.1)/0.4)**2)
         # average accuracy of all datasets
    else:
        metric_to_max = metric_test  #
    torch.cuda.empty_cache()    
    del trainer
    plt.close('all')

    return model_unique_name, metric_train, metric_val, metric_test, metrics, metric_to_max

def main(project_name, is_regression, sweep_id):
    with wandb.init(project=project_name) as run:
        model_unique_name, metric_train, metric_val, metric_test, metrics, metric_to_max = objective(run.config, is_regression, sweep_id)
        if is_regression:
            log_dict = {    "fScore_train": metric_train,
                            "fScore_test": metric_test,
                            "fScore_val": metric_val, 
                            "metric_to_max": metric_to_max,
                            "model_unique_name": model_unique_name}
        else:
            log_dict = { "accuracy_train": metric_train,
                        "accuracy_test": metric_test,
                        "accuracy_val": metric_val, 
                        "metric_to_max": metric_to_max,
                        "model_unique_name": model_unique_name}
        log_dict.update(metrics)
        run.log(log_dict)
        
sweep_configuration = {
    "method": "random",# "bayes",
    "metric": {
        "goal": "maximize", 
        "name": "metric_to_max"},
    "parameters": {
        "whichDataset" : {"values": ["ratio", "ratioNorm", "indiv"]},
        "whichDisplacement" : {"values": ["displacement", "xyPosition"]},
        "replace_nan_by_min" : {"values": [True, False]},
        "remove_mean": {"values": [False]},
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
        "store_best": {"values": ["myFScore", "Accuracy"] if is_regression else ["Accuracy"]},
        "loss": {"values": ["MSE", "L1", "Huber"] if is_regression else None},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

wandb.agent(sweep_id, function=lambda: main(project_name = project_name , \
                                            is_regression = is_regression, sweep_id = sweep_id), count=1000)