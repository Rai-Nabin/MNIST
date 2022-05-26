from torch.nn import Module, Flatten, Sequential, Linear, ReLU
from . import config


class NeuralNetwork(Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(NeuralNetwork, self).__init__()
        self.flatten = Flatten()
        self.linear_relu_stack = Sequential(
            Linear(28*28*3, 512), ReLU(), Linear(512, 512), ReLU(), Linear(512, num_classes))

    def forward(self, x):
        out = self.flatten(x)
        out = self.linear_relu_stack(out)
        return out
