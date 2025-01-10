import torch

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