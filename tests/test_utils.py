import numpy as np
import pytest

from cplx_gmm._utils import check_random_state


def test_check_random_state_none_returns_global_random_state() -> None:
    rng = check_random_state(None)
    assert isinstance(rng, np.random.RandomState)


def test_check_random_state_int_returns_seeded_random_state() -> None:
    rng_1 = check_random_state(0)
    rng_2 = check_random_state(0)

    assert rng_1.rand() == rng_2.rand()


def test_check_random_state_invalid_seed_raises() -> None:
    with pytest.raises(ValueError, match="cannot be used to seed"):
        check_random_state("invalid")
