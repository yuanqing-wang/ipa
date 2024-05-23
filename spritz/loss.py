import torch

class ConsistencyLoss(torch.nn.Module):
    def forward(
            self,
            X: torch.nn.Tensor
    ):
        variance = torch.var(X, dim=-1).mean()
        return variance
    
