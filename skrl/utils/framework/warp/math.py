from typing import Any, Union

import numpy as np
import warp as wp


__all__ = ["scalar_mul", "mean", "var", "std"]


@wp.kernel
def _scalar_mul(dst: wp.array2d(dtype=Any), src: wp.array2d(dtype=Any), scalar: Any):
    i, j = wp.tid()
    dst[i, j] = src[i, j] * dst.dtype(scalar)


@wp.kernel
def _mean_1d(src: wp.array(ndim=1), n: int, dst: wp.array(ndim=1)):
    i = wp.tid()
    wp.atomic_add(dst, 0, dst.dtype(src[i]) / dst.dtype(n))


@wp.kernel
def _mean_2d(src: wp.array(ndim=2), n: int, dst: wp.array(ndim=1)):
    i, j = wp.tid()
    wp.atomic_add(dst, 0, dst.dtype(src[i, j]) / dst.dtype(n))


@wp.kernel
def _mean_3d(src: wp.array(ndim=3), n: int, dst: wp.array(ndim=1)):
    i, j, k = wp.tid()
    wp.atomic_add(dst, 0, dst.dtype(src[i, j, k]) / dst.dtype(n))


@wp.kernel
def _mean_4d(src: wp.array(ndim=4), n: int, dst: wp.array(ndim=1)):
    i, j, k, l = wp.tid()
    wp.atomic_add(dst, 0, dst.dtype(src[i, j, k, l]) / dst.dtype(n))


_MEAN = [None, _mean_1d, _mean_2d, _mean_3d, _mean_4d]


@wp.kernel
def _var_1d(src: wp.array(ndim=1), mean: wp.array(ndim=1), n: int, dst: wp.array(ndim=1)):
    i = wp.tid()
    wp.atomic_add(dst, 0, wp.pow(dst.dtype(src[i]) - mean[0], 2.0) / dst.dtype(n))


@wp.kernel
def _var_2d(src: wp.array(ndim=2), mean: wp.array(ndim=1), n: int, dst: wp.array(ndim=1)):
    i, j = wp.tid()
    wp.atomic_add(dst, 0, wp.pow(dst.dtype(src[i, j]) - mean[0], 2.0) / dst.dtype(n))


@wp.kernel
def _var_3d(src: wp.array(ndim=3), mean: wp.array(ndim=1), n: int, dst: wp.array(ndim=1)):
    i, j, k = wp.tid()
    wp.atomic_add(dst, 0, wp.pow(dst.dtype(src[i, j, k]) - mean[0], 2.0) / dst.dtype(n))


@wp.kernel
def _var_4d(src: wp.array(ndim=4), mean: wp.array(ndim=1), n: int, dst: wp.array(ndim=1)):
    i, j, k, l = wp.tid()
    wp.atomic_add(dst, 0, wp.pow(dst.dtype(src[i, j, k, l]) - mean[0], 2.0) / dst.dtype(n))


_VAR = [None, _var_1d, _var_2d, _var_3d, _var_4d]


@wp.kernel
def _std(src: wp.array(ndim=1), dst: wp.array(ndim=1)):
    dst[0] = wp.sqrt(src[0])


def scalar_mul(array: wp.array, scalar: Union[int, float], inplace: bool = False) -> wp.array:
    output = (
        array
        if inplace
        else wp.empty(array.shape, dtype=array.dtype, device=array.device, requires_grad=array.requires_grad)
    )
    wp.launch(_scalar_mul, dim=array.shape, inputs=[output, array, scalar], device=array.device)
    return output


def mean(array: wp.array, *, dtype: type = wp.float32) -> wp.array:
    output = wp.zeros((1,), dtype=dtype, requires_grad=array.requires_grad)
    wp.launch(
        _MEAN[array.ndim],
        dim=array.shape,
        inputs=[array, np.prod(array.shape).item()],
        outputs=[output],
        device=array.device,
    )
    return output


def var(array: wp.array, *, dtype: type = wp.float32, correction: int = 1) -> wp.array:
    output = wp.zeros((1,), dtype=dtype, requires_grad=array.requires_grad)
    wp.launch(
        _VAR[array.ndim],
        dim=array.shape,
        inputs=[array, mean(array, dtype=dtype), np.prod(array.shape).item() - correction],
        outputs=[output],
        device=array.device,
    )
    return output


def std(array: wp.array, *, dtype: type = wp.float32, correction: int = 1) -> wp.array:
    _var = var(array, dtype=dtype, correction=correction)
    output = wp.zeros((1,), dtype=dtype, requires_grad=True) if array.requires_grad else _var
    wp.launch(_std, dim=1, inputs=[_var], outputs=[output], device=array.device)
    return output
