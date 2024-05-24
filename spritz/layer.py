import torch

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self._W = torch.nn.Parameter(
            torch.randn(in_features, out_features)
        )
        self.B = torch.nn.Parameter(
            torch.randn(out_features)
        )

    @property
    def W(self):
        W = self._W
        W_shape = W.shape
        W_flatten = W.flatten()
        W_flatten_normalized = torch.nn.functional.normalize(
            W_flatten,
            p=2,
        )
        W = W.view(W_shape)
        return W
    
    def forward(
            self,
            X: torch.Tensor,
    ):
        # (N, D) @ (D, C) + (C,) -> (N, C)
        return X @ self.W + self.B

