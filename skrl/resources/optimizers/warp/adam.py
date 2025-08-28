from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import math

import warp as wp

from skrl import config
from skrl.utils.framework.warp import ScopedCapture


tiled = wp.constant(config.warp.tiled)
tile_dim_0 = wp.constant(config.warp.tile_dim_0)
block_dim = wp.constant(config.warp.block_dim)


def create_clip_by_total_norm_kernels(max_norm: float):
    @wp.func
    def square(x: wp.float32) -> wp.float32:
        return x * x

    @wp.func
    def clip_by_norm(x: wp.float32, sum_squares: wp.float32) -> wp.float32:
        norm = wp.sqrt(sum_squares)
        if norm > wp.static(max_norm):
            return x / norm * wp.static(max_norm)
        return x

    @wp.kernel(enable_backward=False)
    def sum_squares(src: wp.array(ndim=1), dst: wp.array(ndim=1)):
        # tiled implementation
        if wp.static(tiled):
            tiled_src = wp.tile_load(src, shape=(tile_dim_0,), offset=(wp.tid() * tile_dim_0,))
            wp.tile_atomic_add(dst, wp.tile_sum(wp.tile_map(square, tiled_src)))
        # non-tiled implementation
        else:
            wp.atomic_add(dst, 0, square(src[wp.tid()]))

    @wp.kernel(enable_backward=False)
    def clip_by_total_norm(src: wp.array(ndim=1), sum_squares: wp.array(ndim=1)):
        i = wp.tid()
        # tiled implementation
        if wp.static(tiled):
            tiled_sum_squares = wp.tile_load(sum_squares, shape=(1,), offset=(0,))
            tiled_src = wp.tile_load(src, shape=(tile_dim_0,), offset=(i * tile_dim_0,))
            tiled_src = wp.tile_map(clip_by_norm, tiled_src, wp.tile_broadcast(tiled_sum_squares, shape=(tile_dim_0,)))
            wp.tile_store(src, tiled_src, offset=(i * tile_dim_0,))
        # non-tiled implementation
        norm = wp.sqrt(sum_squares[0])
        if norm > wp.static(max_norm):
            src[i] = src[i] / norm * wp.static(max_norm)

    return sum_squares, clip_by_total_norm


@wp.kernel(enable_backward=False)
def _adam_step(
    param: wp.array(ndim=1),
    grad: wp.array(ndim=1),
    m1: wp.array(ndim=1),
    m2: wp.array(ndim=1),
    t: wp.array(ndim=1),
    lr: wp.array(ndim=1),
    beta1: float,
    beta2: float,
    eps: float,
):
    i = wp.tid()
    m1[i] = beta1 * m1[i] + (1.0 - beta1) * grad[i]
    m2[i] = beta2 * m2[i] + (1.0 - beta2) * grad[i] * grad[i]
    m1_hat = m1[i] / (1.0 - wp.pow(beta1, wp.float32(t[0])))
    m2_hat = m2[i] / (1.0 - wp.pow(beta2, wp.float32(t[0])))
    param[i] = param[i] - lr[0] * m1_hat / (wp.sqrt(m2_hat) + eps)


@wp.kernel(enable_backward=False)
def _increase_timestep(t: wp.array(ndim=1)):
    t[0] += 1


def adam_step(
    params: Sequence[wp.array],
    gradients: Sequence[wp.array],
    m1: Sequence[wp.array],
    m2: Sequence[wp.array],
    t: wp.array,
    lr: wp.array,
    betas: Tuple[float, float],
    eps: float,
) -> None:
    """Perform an optimization step to update parameters.

    :param params: Parameters.
    :param gradients: Gradients.
    :param m1: First moment of the parameters.
    :param m2: Second moment of the parameters.
    :param t: Timestep.
    :param lr: Learning rate.
    :param betas: Beta coefficients.
    :param eps: Term added to the denominator to improve numerical stability.
    """
    wp.launch(_increase_timestep, dim=1, inputs=[t])
    for i in range(len(params)):
        wp.launch(
            _adam_step,
            dim=params[i].shape[0],
            inputs=[params[i], gradients[i], m1[i], m2[i], t, lr, betas[0], betas[1], eps],
        )


class Adam:
    def __init__(
        self,
        params: Sequence[wp.array],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        device: Optional[Union[str, wp.context.Device]] = None,
    ) -> None:
        """Adam optimizer.

        Adapted from Warp implementation of `warp.optim.Adam <https://nvidia.github.io/warp>`_
        to support CUDA graphs, gradient clipping and state dict.

        :param params: Model parameters.
        :param lr: Learning rate.
        :param betas: Coefficients for the running averages of the gradient and its square.
        :param eps: Term added to the denominator to improve numerical stability.
        """
        self.device = config.warp.parse_device(device)
        self.params = [param.flatten() for param in params]
        self.gradients = [param.grad.flatten() for param in self.params]

        self._betas = betas
        self._eps = eps
        self._t = wp.zeros((1,), dtype=wp.int32, device=self.device)
        self._lr = wp.array([lr], dtype=wp.float32, device=self.device)
        self._m1 = [wp.zeros_like(param) for param in self.params]
        self._m2 = [wp.zeros_like(param) for param in self.params]

        self._use_graph = self.device.is_cuda
        self._graph_clip_by_total_norm = None
        self._graph_adam_step = None
        self._max_norm = None

    def step(self, *, lr: Optional[float] = None) -> None:
        """Perform an optimization step to update parameters.

        :param lr: Learning rate.
        """
        if lr is not None:
            self._lr.fill_(lr)
        if self._graph_adam_step is None:
            with ScopedCapture(device=self.device, enabled=self._use_graph) as capture:
                adam_step(self.params, self.gradients, self._m1, self._m2, self._t, self._lr, self._betas, self._eps)
            self._graph_adam_step = capture.graph
        else:
            wp.capture_launch(self._graph_adam_step)

    def state_dict(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def clip_by_total_norm(self, max_norm: float):
        """Clip (scaling down) parameters' gradients in-place by their total norm.

        .. note::

            This method captures, and launches, the computation done by the ``clip_by_total_norm`` function
            on a CUDA graph for performance reasons.

        https://arxiv.org/abs/1211.5063

        :param max_norm: Maximum global norm.
        """
        # create kernels if not already done or if `max_norm` has changed
        if max_norm != self._max_norm:
            self._max_norm = max_norm
            self._graph_clip_by_total_norm = None
            self._sum_squares = wp.zeros((1,), dtype=wp.float32, device=self.device)
            self._sum_squares_kernel, self._clip_by_total_norm_kernel = create_clip_by_total_norm_kernels(max_norm)
        # clip gradients
        self._sum_squares.zero_()
        if self._graph_clip_by_total_norm is None:
            with ScopedCapture(device=self.device, enabled=self._use_graph) as capture:
                for gradient in self.gradients:
                    wp.launch(
                        self._sum_squares_kernel,
                        dim=[math.ceil(gradient.shape[0] / tile_dim_0), block_dim] if tiled else gradient.shape[0],
                        inputs=[gradient],
                        outputs=[self._sum_squares],
                        device=self.device,
                        block_dim=block_dim,
                    )
                for gradient in self.gradients:
                    wp.launch(
                        self._clip_by_total_norm_kernel,
                        dim=[math.ceil(gradient.shape[0] / tile_dim_0), block_dim] if tiled else gradient.shape[0],
                        inputs=[gradient, self._sum_squares],
                        device=self.device,
                        block_dim=block_dim,
                    )
            self._graph_clip_by_total_norm = capture.graph
        else:
            wp.capture_launch(self._graph_clip_by_total_norm)
