import wandb
import os
from PycalcAct.train_function import *
import torch

os.environ['WANDB_API_KEY'] = '73246a79f06da26fb325d763bd90ab7fc81bc9e6'
is_regression = False
if is_regression:
    project_name = "my-first-sweep-regressor"
else:
    project_name = "my-first-sweep-classifier"

def objective(config, is_regression):
    model_unique_name, trainer = setupTrainer(config, is_regression)
    n_epoch = 500
    trainer.train(n_epoch)
    metric_train, metric_val, metric_test = save_model_perf(trainer, model_unique_name)
    metrics = save_model_generalizability(trainer, model_unique_name)

    metric_to_min = sum([metrics[k] for k in ["SL_dist", "P14_dist", "OT3_dist"]]) # need to min
    if is_regression:
        metric_to_min += 1000*sum([metrics[k] for k in ["SL_fScore", "P14_fScore", "OT3_fScore"]])
        
    torch.cuda.empty_cache()    
    del trainer
    return model_unique_name, metric_train, metric_val, metric_test, metrics, metric_to_min

def main(project_name, is_regression):
    with wandb.init(project=project_name) as run:
        model_unique_name, metric_train, metric_val, metric_test, metrics, metric_to_min = objective(run.config, is_regression)
        log_dict = {   "metric_train": metric_train,
                    "metric_test": metric_test,
                    "metric_val": metric_val, 
                    "metric_to_min": metric_to_min,
                    "model_unique_name": model_unique_name}
        log_dict.update(metrics)
        run.log(log_dict)
        
sweep_configuration = {
    "method": "bayes",
    "metric": {
        "goal": "minimize", 
        "name": "metric_to_min"},
    "parameters": {
        "whichDataset" : {"values": ["ratio", "ratioNorm", "indiv"]},
        "whichDisplacement" : {"values": ["displacement", "xyPosition", "None"]},
        "replace_nan_by_min" : {"values": [True, False]},
        "remove_mean": {"values": [False]},
        "whichFFT" : {"values": [True, False]},
        "numRNN" : {"values": [1, 2]}, # {"values": [1, 2, 3]}
        "sizeRNN": {"values": [8, 16, 32]}, #{"values": [8, 16, 32, 64]}
        "bidir" : {"values": [True, False]},
        "numFC": {"values": [1, 2]}, # {"values": [1, 2, 3]}
        "sizeFC" :  {"values": [8, 16, 32]},    #{"values": [8, 16, 32, 64]}
        "dropout" : {"min": 0.05, "max": 0.20},
        "weighted" : {"values": [True, False]},
        "customLoss" : {"values": [False]},
        "initial_lr": {"values": [0.01, 0.001]},
        "weight_decay": {"values": [1e-5, 1e-4, 1e-3]},
        "batch_size": {"values": [128, 256, 215, 1024, 2048, 4096]},
        # "augment_gt" : {"values": [True, False]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

wandb.agent(sweep_id, function=lambda: main(project_name = project_name , is_regression = is_regression), count=200)