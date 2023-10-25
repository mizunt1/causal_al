import torch.nn as nn
import torch.nn.functional as F
import torch 

class ModelEnsemble(nn.Module):
    def __init__(self, input_size, num_models):
        super().__init__()
        self.model_backbone = nn.Sequential(
            nn.Linear(input_size, 24, True), nn.ReLU(), nn.Dropout(), nn.Linear(24,1))
        self.models = nn.ModuleList([self.model_backbone for model in range(num_models)])
        
    def forward(self, x):
        return torch.stack([model(x) for model in self.models])

class ModelEnsembleHet(nn.Module):
    def __init__(self, input_size, num_models):
        super().__init__()
        self.modelA = nn.Sequential(
            nn.Linear(input_size, 12, True), nn.ReLU(), nn.Linear(12, 1, True))
        self.modelB = nn.Sequential(
            nn.Linear(input_size, 12, True), nn.ReLU(), nn.Dropout(), nn.Linear(12,1, True))
        self.modelC = nn.Sequential(
            nn.Linear(input_size, 24, True), nn.ReLU(), nn.Dropout(), nn.Linear(24,12, True), nn.ReLU(), nn.Linear(12, 1, True))

        self.models = nn.ModuleList([self.modelA, self.modelB, self.modelC])
        
    def forward(self, x):
        return torch.stack([model(x) for model in self.models])

def ensemble_loss(output, target):
    losses = torch.stack(
        [F.binary_cross_entropy_with_logits(model_result, target) for model_result in output])
    total_loss = torch.sum(losses)
    return total_loss



