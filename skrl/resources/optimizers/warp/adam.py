from typing import Any, Mapping, Optional, Sequence, Tuple

import warp as wp
import warp.optim as optim


@wp.kernel(enable_backward=False)
def _sum_squares(src: wp.array(ndim=1), dst: wp.array(ndim=1)):
    wp.atomic_add(dst, 0, wp.pow(src[wp.tid()], 2.0))


@wp.kernel(enable_backward=False)
def _clip_by_total_norm(src: wp.array(ndim=1), sum_squares: wp.array(ndim=1), max_norm: float):
    i = wp.tid()
    norm = wp.sqrt(sum_squares[0])
    if norm > max_norm:
        src[i] = src[i] / norm * max_norm


class Adam(optim.Adam):
    def __init__(
        self,
        params: Sequence[wp.array],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
    ) -> None:
        """Adam optimizer.

        Adapted from `Warp's Adam <https://nvidia.github.io/warp>`_ to support state dict.

        :param params: Model parameters.
        :param lr: Learning rate.
        :param betas: Coefficients for the running averages of the gradient and its square.
        :param eps: Term added to the denominator to improve numerical stability.
        """
        super().__init__([param.flatten() for param in params], lr=lr, betas=betas, eps=eps)

    def step(self, gradients: Sequence[wp.array], *, lr: Optional[float] = None) -> None:
        """Perform an optimization step to update parameters.

        :param gradients: Gradients of the parameters.
        :param lr: Learning rate.
        """
        if lr is not None:
            self.lr = lr
        super().step(gradients)

    def state_dict(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        raise NotImplementedError


def clip_by_total_norm(arrays: Sequence[wp.array], max_norm: float) -> Sequence[wp.array]:
    """Clip (scaling down) arrays' values in place by their total norm.

    https://arxiv.org/abs/1211.5063

    :param arrays: List of flattened arrays to clip.
    :param max_norm: Maximum global norm.

    :return: Clipped arrays.
    """
    sum_squares = wp.zeros((1,), dtype=wp.float32)
    for array in arrays:
        wp.launch(_sum_squares, dim=array.shape[0], inputs=[array], outputs=[sum_squares])
    for array in arrays:
        wp.launch(_clip_by_total_norm, dim=array.shape[0], inputs=[array, sum_squares, max_norm])
    return arrays
