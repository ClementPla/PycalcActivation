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
        self.add_state("f_score", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("p_val", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("num_batches", default=torch.tensor(0.), dist_reduce_fx="mean")

    def update(self, preds: Tensor, target: Tensor) -> None:
        # preds, target = self._input_format(preds, target)
        # if preds.shape != target.shape:
        #     raise ValueError("preds and target must have the same shape")
        all_classes = np.unique(target.cpu().numpy())
        if preds.squeeze().dim() >1:
            preds = torch.argmax(preds, dim=1)
        groups = [preds[target == cls].cpu().detach().numpy() for cls in all_classes]
        
        f,p = f_oneway(*groups)
        # print(f)
        if not np.isnan(f):
            self.f_score += torch.tensor(f, dtype=torch.float32)
            self.p_val += torch.tensor(p, dtype=torch.float32)
        else:
            self.f_score += torch.tensor(0.0, dtype=torch.float32)
            self.p_val += torch.tensor(1.0, dtype=torch.float32)
        self.num_batches += 1
        
    def compute(self) -> Tensor:
        return self.f_score / self.num_batches