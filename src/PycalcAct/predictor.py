from copy import deepcopy
from functools import partial

import matplotlib.pyplot as plt
import torch
from colorama import Fore, Style
from progress_table import ProgressTable
from torchinfo import summary as torch_summary
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, CohenKappa, ConfusionMatrix

from PycalcAct.utils.wrapper import on_keyboard_interrup


class Predictor:
    def __init__(
        self,
        dataset,
        model,
        device="cuda",
        criterion=None,
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.dataset = dataset.to(device)
        self.criterion = criterion if criterion else self.default_criterion()
        self.metrics = MetricCollection(
            dict(
                Accuracy=Accuracy(task="multiclass", num_classes=self.dataset.num_classes),
                CohenKappa=CohenKappa(task="multiclass", num_classes=self.dataset.num_classes),
            )
        ).to(device)
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=self.dataset.num_classes).to(device)

    def summary(self, optimizer=False, model=False, data=True):
        if model:
            print(Fore.BLUE + Style.BRIGHT + Style.DIM + "Model")
            B = 10
            F = self.dataset.features
            N = self.dataset.length_serie
            print(torch_summary(self.model, input_size=(B, N, F)))
            print(Style.RESET_ALL, end="")
        if data:
            print(Fore.GREEN + Style.BRIGHT + Style.DIM + "Data")
            print(Style.RESET_ALL, end="")
            print(Fore.GREEN, end="")
            return self.dataset.summary

    def reset(self, dataset=None, model=None):
        if dataset:
            self.update_dataset(dataset, reset=False)
        self.confmat.reset()
        self.metrics.reset()

        if model:
            self.model = model.to(self.device)
            self._initial_state_dict = deepcopy(self.model.state_dict())
        else:
            self.model.load_state_dict(self._initial_state_dict)

    def predict(self, which="best", show_confmat=True):
        self.model.load_state_dict(self._best_state_dict)
        x,y = self.dataset.batch_from_data(x = self.dataset.data,
                                                y = self.dataset.classes_int,
                                                time_first=True, to_cuda = True)
        loss, scores = self.eval(x, y)




        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        if which == "best":
            self.model.load_state_dict(self._best_state_dict)
            fig.suptitle("Best Model")
        elif which == "last":
            self.model.load_state_dict(self._last_state_dict)
            fig.suptitle("Last Model")

        callbacks = (
            self.dataset.train_batch,
            self.dataset.val_batch,
            self.dataset.test_batch,
        )

        for i, (name, callable) in enumerate(zip(["Train", "Val", "Test"], callbacks)):
            x, y = callable(True, to_cuda=True)
            loss, scores = self.eval(x, y)
            axs[i].set_title(
                f"Accuracy {(name)} {scores['Accuracy'].item():.2%}, Cohen's Kappa {scores['CohenKappa'].item():.1%}"
            )
            self.confmat.plot(
                ax=axs[i],
                cmap="RdYlGn",
                labels=self.dataset.labels,
            )
        if show_confmat:
            fig.show()
        else:
            return fig

    def eval(self, x, y):
        self.model.eval()
        self.metrics.reset()
        self.confmat.reset()

        with torch.no_grad():
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y.long())
            y_pred = torch.softmax(y_pred, dim=1)
            self.metrics(y_pred, y)
            self.confmat(y_pred, y)
        return loss, self.metrics.compute()

    def estimate_uncertainty(self, x, n_samples=100):
        self.model.train()
        with torch.no_grad():
            y_pred = torch.stack([torch.softmax(self.model(x), dim=1) for _ in range(n_samples)], dim=0)
            y_pred_mean = y_pred.mean(dim=0)
            y_pred_std = y_pred.std(dim=0)
        return y_pred_mean, y_pred_std

    def load_best(self):
        self.model.load_state_dict(self._best_state_dict)

    def load_last(self):
        self.model.load_state_dict(self._last_state_dict)

    def update_dataset(self, dataset, reset=True):
        self.dataset = dataset.to(self.device)
        if reset:
            self.reset()

    def default_criterion(self):
        return torch.nn.CrossEntropyLoss(self.dataset.weights).to(self.device)
