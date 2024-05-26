from random import Random
import torch
from .rotation import RandomRotation
from .loss import ConsistencyLoss

def init(
        model: torch.nn.Module,
        num_samples: int=32,
        num_rotations: int=32,
        optimizer: torch.optim.Optimizer=torch.optim.Adam,
        lr: float=1e-3,
        num_epochs: int=100,
    ):
        optimizer = optimizer(
            model.parameters(),
            lr=lr,
        )
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            X = torch.randn(num_samples, 3)
            X = RandomRotation()(X, num_rotations)
            Y = model(X)
            loss = ConsistencyLoss()(Y)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss {loss}")
        return model
