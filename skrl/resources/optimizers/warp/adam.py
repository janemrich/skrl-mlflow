from typing import Any, Mapping, Sequence, Tuple

import warp as wp
import warp.optim as optim


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

    def state_dict(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        raise NotImplementedError
