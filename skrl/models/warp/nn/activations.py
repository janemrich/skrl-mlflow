from typing import Sequence, Tuple

from .module import Module


class ELU(Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self._alpha = alpha

    def parse(self, uid: str) -> Tuple[str, Sequence[str], Sequence[str], Sequence[str], Sequence[str]]:
        impl = "" if self._alpha == 1.0 else f"{self._alpha} * "
        # templates
        name = "_elu"
        template_function = f"""
@wp.func
def {{name}}(x: wp.float32):
    if x > 0.0:
        return x
    else:
        return {impl}(wp.exp(x) - 1.0)
"""
        template_kernel = """
# ELU
{output} = wp.tile_map({name}, {input})
"""
        # generation
        functions = [template_function.strip().format(name=name)]
        kernel_parameters = []
        kernel_arguments = []
        kernel_definitions = [template_kernel.strip().format(name=name, input="{input}", output="{output}")]
        return None, functions, kernel_parameters, kernel_arguments, kernel_definitions


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def parse(self, uid: str) -> Tuple[str, Sequence[str], Sequence[str], Sequence[str], Sequence[str]]:
        # templates
        name = "_relu"
        template_function = """
@wp.func
def {name}(x: wp.float32):
    return wp.max(x, 0.0)
"""
        template_kernel = """
# ReLU
{output} = wp.tile_map({name}, {input})
"""
        # generation
        functions = [template_function.strip().format(name=name)]
        kernel_parameters = []
        kernel_arguments = []
        kernel_definitions = [template_kernel.strip().format(name=name, input="{input}", output="{output}")]
        return None, functions, kernel_parameters, kernel_arguments, kernel_definitions


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def parse(self, uid: str) -> Tuple[str, Sequence[str], Sequence[str], Sequence[str], Sequence[str]]:
        # templates
        name = "_tanh"
        template_function = """
@wp.func
def {name}(x: wp.float32):
    return wp.tanh(x)
"""
        template_kernel = """
# Tanh
{output} = wp.tile_map({name}, {input})
"""
        # generation
        functions = [template_function.strip().format(name=name)]
        kernel_parameters = []
        kernel_arguments = []
        kernel_definitions = [template_kernel.strip().format(name=name, input="{input}", output="{output}")]
        return None, functions, kernel_parameters, kernel_arguments, kernel_definitions
