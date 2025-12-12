from sklearn.feature_selection import SelectFdr
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
import numpy as np
from scipy.stats import f_oneway

class myCustomCriterion:
    def __init__(
        self,
        weight,
        D,
        device
    ):
        self.weight = weight
        self.D = D
        self.device = device

    def forward(self, input, target)-> torch.Tensor:
        myFunction =  torch.nn.CrossEntropyLoss(weight=self.weight, reduction = 'none').to(self.device)       
        crossEntropyLoss = myFunction(input, target).to(self.device)
        predictedClass = input.argmax(dim = 1).to(self.device)
        correctionFactor = self.D[target,predictedClass]
        correctedLoss = crossEntropyLoss*correctionFactor
        loss = torch.mean(correctedLoss)
        # print(loss)
        
        return loss
    
    def __call__(self, input, target):
        return self.forward(input, target)
    
class myMSELoss:
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean", weights=None, classes=None):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.weights = weights
        self.classes = classes

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        weight_list = target.tolist()
        weight_dict = {i: self.weights[int(k)] for k,i in enumerate(self.classes)} # square self.weight?
        weight_list = torch.Tensor([weight_dict[min(self.classes, key=lambda c: abs(c - y))]
            for y in weight_list]).to(input.device)
        return F.mse_loss(input.squeeze(), target, reduction=self.reduction, weight=weight_list)
    
    def __call__(self, input, target):
        return self.forward(input, target)
    

class myFScore(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Keep 1D float/int tensors for accumulation
        self.add_state("preds",  default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
        self.add_state("target", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # If preds are logits with shape [B, C], convert to class indices [B]
        if preds.ndim > 1 and preds.size(-1) > 1:
            preds = torch.argmax(preds, dim=1)

        # Flatten to 1D and ensure float32 (ANOVA is numeric; int is also fine)
        preds  = preds.detach().float().view(-1)
        target = target.detach().float().view(-1)

        # Concatenate into the running tensor states
        self.preds  = torch.cat([self.preds, preds])
        self.target = torch.cat([self.target, target])

    def compute(self) -> torch.Tensor:
        # Convert to numpy for scipy
        all_preds   = self.preds.detach().cpu().numpy()
        all_targets = self.target.detach().cpu().numpy()

        # Identify unique classes in targets and group predictions by target class
        classes = np.unique(all_targets)
        groups  = [all_preds[all_targets == cls] for cls in classes]

        # If each group is constant or empty, return 0.0 to avoid invalid ANOVA
        if len(groups) < 2 or any(len(g) == 0 for g in groups) or all(np.all(g == g[0]) for g in groups):
            f = 0.0
        else:
            f, _ = f_oneway(*groups)
            if not np.isfinite(f):
                f = 0.0

        return torch.tensor(float(f), dtype=torch.float32)

class mySpearman(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds",  default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
        self.add_state("target", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # If preds are logits [B, C], convert to class indices [B]; otherwise keep as continuous
        if preds.ndim > 1 and preds.size(-1) > 1:
            preds = torch.argmax(preds, dim=1)

        preds  = preds.detach().float().view(-1)
        target = target.detach().float().view(-1)

        self.preds  = torch.cat([self.preds, preds])
        self.target = torch.cat([self.target, target])

    def compute(self) -> torch.Tensor:
        x = self.target.detach().cpu().numpy()
        y = self.preds.detach().cpu().numpy()

        # Spearman is undefined if either vector is constant
        if np.std(x) == 0 or np.std(y) == 0:
            rho = 0.0
        else:
            rho, _ = spearmanr(x, y)
            if not np.isfinite(rho):
                rho = 0.0

        return torch.tensor(float(rho), dtype=torch.float32)

    
# class myFScore(Metric):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.add_state("preds", default=[], dist_reduce_fx="cat")
#         self.add_state("target", default=[], dist_reduce_fx="cat")

#     def update(self, preds: Tensor, target: Tensor) -> None:
#         # Convert logits to class predictions if needed
#         if preds.squeeze().dim() > 1:
#             preds = torch.argmax(preds, dim=1)

#         preds = preds.detach().cpu()
#         target = target.detach().cpu()

#         # Convert to list of tensors
#         self.preds.extend(preds.detach().tolist())
#         self.target.extend(target.detach().tolist())

#     def compute(self) -> Tensor:
#         # Convert lists to numpy arrays
#         all_preds = np.array(self.preds)
#         all_targets = np.array(self.target)

#         # Identify unique classes
#         all_classes = np.unique(all_targets)

#         # Group predictions by true class
#         groups = [all_preds[all_targets == cls] for cls in all_classes]

#         # Perform one-way ANOVA
        
#         if all(np.all(group == group[0]) for group in groups):
#             f = 0.0  # or some default value
#         else:
#             f, _ = f_oneway(*groups)
#             if np.isnan(f) or np.isinf(f):
#                 f = 0.0

#         # Return F-score as tensor
#         return torch.tensor(f, dtype=torch.float32)
    



# class mySpearman(Metric):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.add_state("preds", default=[], dist_reduce_fx="cat")
#         self.add_state("target", default=[], dist_reduce_fx="cat")

#     def update(self, preds: Tensor, target: Tensor) -> None:
#         # Convert logits to class predictions if needed
#         if preds.squeeze().dim() > 1:
#             preds = torch.argmax(preds, dim=1)

#         preds = preds.detach().cpu()
#         target = target.detach().cpu()

#         # Convert to list of tensors
#         self.preds.extend(preds.detach().tolist())
#         self.target.extend(target.detach().tolist())

#     def compute(self) -> Tensor:
#         # Convert lists to numpy arrays
#         all_preds = np.array(self.preds)
#         all_targets = np.array(self.target)

#         # Perform Spearman correlation
#         if np.std(all_preds) == 0 or np.std(all_targets) == 0:
#             f = 0.0  # or some default value
#         else:
#             f, _ = spearmanr(all_targets, all_preds)
#             if np.isnan(f) or np.isinf(f):
#                 f = 0.0

#         # Return F-score as tensor
#         return torch.tensor(f, dtype=torch.float32)