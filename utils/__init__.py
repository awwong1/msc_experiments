import torch.nn as nn


class View(nn.Module):
    """Utility module for reshaping tensors within nn.Sequential block.
    """
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
