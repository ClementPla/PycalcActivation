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


class Trainer:
    def __init__(
        self,
        dataset,
        model,
        optim=None,
        criterion=None,
        scheduler=None,
        initial_lr=0.001,
        weight_decay=1e-4,
        store_best="Accuracy",
        device="cuda",
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.dataset = dataset.to(device)

        self.initial_lr = initial_lr
        self.weight_decay = weight_decay

        self.optim = optim if optim else self.default_optimizer()
        self.criterion = criterion if criterion else self.default_criterion()
        self.scheduler = scheduler

        self._initial_optim_state_dict = deepcopy(self.optim.state_dict())
        self._initial_state_dict = deepcopy(self.model.state_dict())
        self._store_best = store_best

        self.metrics = MetricCollection(
            dict(
                Accuracy=Accuracy(task="multiclass", num_classes=self.dataset.num_classes),
                CohenKappa=CohenKappa(task="multiclass", num_classes=self.dataset.num_classes),
            )
        ).to(device)
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=self.dataset.num_classes).to(device)

        self._best_state_dict = self._initial_state_dict
        self._last_state_dict = self._initial_state_dict

        if scheduler:
            self._initial_scheduler_state_dict = deepcopy(scheduler.state_dict())

    def default_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)

    def default_criterion(self):
        return torch.nn.CrossEntropyLoss(self.dataset.weights).to(self.device)

    def default_scheduler(self):
        return partial(torch.optim.lr_scheduler.CosineAnnealingLR, self.optim, eta_min=1e-6)

    def summary(self, optimizer=False, model=False, data=True):
        if optimizer:
            print(Fore.GREEN + "Optimizer")
            print(Style.RESET_ALL, end="")
            print(self.optim)
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

    def reset(self):
        self.confmat.reset()
        self.metrics.reset()
        self.model.load_state_dict(self._initial_state_dict)
        self.optim.load_state_dict(self._initial_optim_state_dict)
        if self.scheduler:
            self.scheduler.load_state_dict(self._initial_scheduler_state_dict)

    def register_last_state(self):
        self._last_state_dict = deepcopy(self.model.state_dict())

    @on_keyboard_interrup("register_last_state")
    def train(self, n_epochs: int, val_every: int = 1, verbose: bool = True, seed=1234):
        torch.manual_seed(seed)

        if self.scheduler is None:
            self.scheduler = self.default_scheduler()(T_max=n_epochs)

        x, y = self.dataset.train_batch(True, to_cuda=True)
        current_best = 0
        xval, yval = self.dataset.val_batch(True, to_cuda=True)
        table = ProgressTable(
            num_decimal_places=2,
            pbar_show_progress=True,
            pbar_style="square",
            default_column_width=25,
        )

        table.add_column("Epoch")
        table.add_column("Loss")
        table.add_column(self._store_best, color="green")

        for e in table(
            range(n_epochs),
            total=n_epochs,
            description="Epoch",
            show_eta=True,
        ):
            self.model.train()
            self.optim.zero_grad()

            idx = torch.randperm(x.shape[0], device=x.device)
            x = x[idx]
            y = y[idx]

            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)

            loss.backward()
            self.optim.step()
            table.update("train loss", loss.item(), aggregate="mean", color="blue")
            table.update("Epoch", e)
            if self.scheduler:
                self.scheduler.step()
            if e % val_every == 0:
                loss, scores = self.eval(xval, yval)
                if scores[self._store_best] > current_best:
                    current_best = scores[self._store_best]
                    self._best_state_dict = deepcopy(self.model.state_dict())
                    if verbose:
                        table[self._store_best] = scores[self._store_best].item() * 100

                    table.next_row()

        table.close()

        self.register_last_state()

    def test(self, which="best"):
        if which == "best":
            self.model.load_state_dict(self._best_state_dict)
        elif which == "last":
            self.model.load_state_dict(self._last_state_dict)

        callbacks = (
            self.dataset.train_batch,
            self.dataset.val_batch,
            self.dataset.test_batch,
        )
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

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

        return fig

    def eval(self, x, y):
        self.model.eval()
        self.metrics.reset()
        self.confmat.reset()

        with torch.no_grad():
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            y_pred = torch.softmax(y_pred, dim=1)
            self.metrics(y_pred, y)
            self.confmat(y_pred, y)
        return loss, self.metrics.compute()

    def update_dataset(self, dataset, reset=True):
        self.dataset = dataset.to(self.device)
        if reset:
            self.reset()
