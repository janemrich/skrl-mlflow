from typing import Iterator, Mapping, Optional, Tuple

from abc import ABC
from collections import OrderedDict

import numpy as np
import warp as wp


class _Parameter(ABC):
    pass


class Module(ABC):
    def __init__(self, *args, **kwargs):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

        self.device = wp.get_device("cuda")

    def __post_init__(self) -> None:
        for k, v in self.__dict__.items():
            if isinstance(v, _Parameter):
                self.register_parameter(k, v)
            elif isinstance(v, Module):
                self.register_module(k, v)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _save_to_state_dict(self, destination, prefix):
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param.data

    def forward(self, *args):
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forward" method')

    def register_parameter(self, name: str, param: Optional[wp.array]) -> None:
        if not isinstance(param, _Parameter):
            raise TypeError(f"Class {type(param)} is not a Parameter subclass")
        self._parameters[name] = param

    def register_module(self, name: str, module: Optional["Module"]) -> None:
        if not isinstance(module, Module):
            raise TypeError(f"Class {type(module)} is not a Module subclass")
        if name in self._modules:
            raise KeyError(f"Module with name '{name}' already exists")
        self._modules[name] = module

    def parameters(self) -> Iterator[Optional[wp.array]]:
        modules = self._modules.values()
        if modules:
            parameters = []
            for module in modules:
                parameters += [p.data if isinstance(p, _Parameter) else p for p in module.parameters()]
        else:
            parameters = [p.data for p in self._parameters.values()]
        return parameters

    def modules(self) -> Iterator["Module"]:
        return self._modules.values()

    def named_modules(self) -> Iterator[Tuple[str, "Module"]]:
        return self._modules.items()

    def state_dict(
        self, *, destination: Optional[Mapping[str, wp.array]] = None, prefix: str = ""
    ) -> Mapping[str, wp.array]:
        if destination is None:
            destination = OrderedDict()
        self._save_to_state_dict(destination, prefix)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + ".")
        return destination

    def load_state_dict(self, state_dict: Mapping[str, wp.array]) -> None:
        def _load_from_state_dict(dst, src):
            if isinstance(src, dict):
                for k in src:
                    _load_from_state_dict(dst[k], src[k])
            elif isinstance(src, wp.array):
                wp.copy(dst, src.to(dst.device))
            elif isinstance(src, np.ndarray):
                wp.copy(dst, wp.array(src, dtype=dst.dtype, device=dst.device))
            else:
                raise NotImplementedError(f"Unsupported type: {type(src)}")

        _load_from_state_dict(self.state_dict(), state_dict)
