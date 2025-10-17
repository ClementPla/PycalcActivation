from sklearn.feature_selection import SelectFdr
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
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        # Convert logits to class predictions if needed
        if preds.squeeze().dim() > 1:
            preds = torch.argmax(preds, dim=1)

        preds = preds.detach().cpu()
        target = target.detach().cpu()

        # Convert to list of tensors
        self.preds.extend(preds.tolist())
        self.target.extend(target.tolist())

    def compute(self) -> Tensor:
        # Convert lists to numpy arrays
        all_preds = np.array(self.preds)
        all_targets = np.array(self.target)

        # Identify unique classes
        all_classes = np.unique(all_targets)

        # Group predictions by true class
        groups = [all_preds[all_targets == cls] for cls in all_classes]

        # Perform one-way ANOVA
        f, _ = f_oneway(*groups)

        # Return F-score as tensor
        return torch.tensor(f if not np.isnan(f) else 0.0, dtype=torch.float32)