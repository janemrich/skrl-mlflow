from typing import Sequence

from .module import Module


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        # register modules
        modules = args[0] if len(args) == 1 and isinstance(args[0], Sequence) else args
        for i, module in enumerate(modules):
            self.register_module(str(i), module)

    def forward(self, input):
        for module in self.modules():
            input = module(input)
        return input
