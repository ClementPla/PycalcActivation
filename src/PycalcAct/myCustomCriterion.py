from sklearn.feature_selection import SelectFdr
import torch
import torch.nn.functional as F
from torch import Tensor

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
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean", weights=None):
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.weights = weights
    def forward(self, input: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        weight_list = target.tolist()
        weight_dict = {
            0 : weights[0]**2,
            1 : weights[1]**2,
            2 : weights[2]**2,
            3 : weights[3]**2,
            }
        weight_list = torch.Tensor([weight_dict[y] for y in weight_list]).to(input.device)
        return F.mse_loss(input.squeeze(), target, reduction=self.reduction, weight=weight_list)
    
    def __call__(self, input, target):
        return self.forward(input, target, weights=self.weights)