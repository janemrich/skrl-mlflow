import hypothesis
import hypothesis.strategies as st
import pytest

import numpy as np
import warp as wp

import skrl.utils.framework.warp.math as math_utils


@hypothesis.given(shape=st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_mean(capsys, ndim, dtype, shape):
    sample = (np.random.rand(*shape[:ndim]) * 100).astype(dtype)
    array = wp.array(sample)

    value = math_utils.mean(array)
    assert np.allclose(value.numpy().item(), np.mean(sample), atol=1e-05, rtol=1e-03)


@hypothesis.given(shape=st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_var(capsys, ndim, dtype, shape):
    sample = (np.random.rand(*shape[:ndim]) * 100).astype(dtype)
    array = wp.array(sample)

    value = math_utils.var(array, correction=0)
    assert np.allclose(value.numpy().item(), np.var(sample, ddof=0), atol=1e-05, rtol=1e-03)

    value = math_utils.var(array, correction=1)
    assert np.allclose(value.numpy().item(), np.var(sample, ddof=1), atol=1e-05, rtol=1e-03, equal_nan=True)


@hypothesis.given(shape=st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_std(capsys, ndim, dtype, shape):
    sample = (np.random.rand(*shape[:ndim]) * 100).astype(dtype)
    array = wp.array(sample)

    value = math_utils.std(array, correction=0)
    assert np.allclose(value.numpy().item(), np.std(sample, ddof=0), atol=1e-05, rtol=1e-03)

    value = math_utils.std(array, correction=1)
    assert np.allclose(value.numpy().item(), np.std(sample, ddof=1), atol=1e-05, rtol=1e-03, equal_nan=True)
