import torch

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.W = torch.nn.Parameter(
            torch.randn(in_features, out_features)
        )
        self.B = torch.nn.Parameter(
            torch.randn(out_features)
        )
