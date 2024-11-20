from copy import deepcopy
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
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
        batch_size=32,
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

    def reset(self, dataset=None, model=None, optim=None, scheduler=None):
        if dataset:
            self.update_dataset(dataset, reset=False)
        self.confmat.reset()
        self.metrics.reset()

        if model:
            self.model = model.to(self.device)
            self._initial_state_dict = deepcopy(self.model.state_dict())
        else:
            self.model.load_state_dict(self._initial_state_dict)

        if optim is None and model is not None:
            self.optim = self.default_optimizer()
            self._initial_optim_state_dict = deepcopy(self.optim.state_dict())
        elif optim:
            self.optim = optim
            self._initial_optim_state_dict = deepcopy(self.optim.state_dict())
        else:
            self.optim.load_state_dict(self._initial_optim_state_dict)

        if scheduler is None and self.scheduler is not None:
            self.scheduler = None
        elif scheduler:
            self.scheduler = scheduler

    def register_last_state(self):
        self._last_state_dict = deepcopy(self.model.state_dict())

    @on_keyboard_interrup("register_last_state")
    def train(self, n_epochs: int, val_every: int = 1, verbose: bool = True, seed=1234):
        if self.scheduler is None:
            self.scheduler = self.default_scheduler()(T_max=n_epochs)
            self._initial_scheduler_state_dict = deepcopy(self.scheduler.state_dict())

        x, y = self.dataset.train_batch(True, to_cuda=True)
        current_best = 0
        xval, yval = self.dataset.val_batch(True, to_cuda=True)
        table = ProgressTable(
            num_decimal_places=2,
            print_header_every_n_rows=15,
            pbar_show_progress=True,
            pbar_style="square",
            default_column_width=25,
        )

        table.add_column("Epoch")
        table.add_column("Loss", color="blue", alignment="right")
        table.add_column(self._store_best, color="green", alignment="right")

        for e in table(
            range(n_epochs),
            total=n_epochs,
            description="Epoch",
            show_eta=True,
        ):
            self.model.train()
            self.optim.zero_grad()
            with torch.cuda.amp.autocast():
                idx = torch.randperm(x.shape[0], device=x.device)
                x = x[idx]
                y = y[idx]

                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)

            loss.backward()
            self.optim.step()

            if self.scheduler:
                self.scheduler.step()
            table.update("Loss", loss.item(), aggregate="mean", color="blue")
            table.update("Epoch", e)
            if e % val_every == 0:
                loss, scores = self.eval(xval, yval)
                if scores[self._store_best] > current_best:
                    current_best = scores[self._store_best]
                    self._best_state_dict = deepcopy(self.model.state_dict())
                    if verbose:
                        table.update(self._store_best, scores[self._store_best].item() * 100, color="green")

                    table.next_row()

        table.close()

        self.register_last_state()

    def test(self, which="best", show_confmat=True):
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
            loss = self.criterion(y_pred, y)
            y_pred = torch.softmax(y_pred, dim=1)
            self.metrics(y_pred, y)
            self.confmat(y_pred, y)
        return loss, self.metrics.compute()

    @torch.inference_mode()
    def predict(self, x):
        return torch.softmax(self.model(x), dim=1)

    def get_loss(self, x, y, average=True):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x)
            if average:
                loss = self.criterion(y_pred, y)
            else:
                loss = F.cross_entropy(y_pred, y, reduction="none")
        return loss

    def estimate_uncertainty(self, x, y=None, n_samples=100):
        self.model.train()
        with torch.no_grad():
            all_preds = []
            all_losses = []
            for _ in range(n_samples):
                pred = self.model(x)
                if y is not None:
                    loss = F.cross_entropy(pred, y, reduction="none")
                    all_losses.append(loss)
                pred = torch.softmax(pred, dim=1)
                all_preds.append(pred)

            y_pred = torch.stack(all_preds, dim=0)
            y_pred_mean = y_pred.mean(dim=0)
            y_pred_std = y_pred.var(dim=0)

        if y is not None:
            return y_pred_mean, y_pred_std, torch.stack(all_losses, dim=0)

        return y_pred_mean, y_pred_std

    def get_predictions(self, x):
        pass

    def load_best(self):
        self.model.load_state_dict(self._best_state_dict)

    def load_last(self):
        self.model.load_state_dict(self._last_state_dict)

    def update_dataset(self, dataset, reset=True):
        self.dataset = dataset.to(self.device)
        if reset:
            self.reset()

    def save(self, path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "best": self._best_state_dict,
                "last": self._last_state_dict,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.optim.load_state_dict(checkpoint["optim"])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self._best_state_dict = checkpoint["best"]
        self._last_state_dict = checkpoint["last"]
