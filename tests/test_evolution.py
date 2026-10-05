import numpy as np
import pytest
from scipy.linalg import expm

from trotterization import exact_unitary, ising_terms, trotter_unitary
from trotterization.models import X, Z


@pytest.mark.parametrize("order", [1, 2])
def test_unitarity_and_zero_time(order):
    terms = ising_terms(3)
    unitary = trotter_unitary(terms, 1.2, 7, order)
    np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(8), atol=1e-12)
    np.testing.assert_allclose(trotter_unitary(terms, 0, 7, order), np.eye(8))


@pytest.mark.parametrize("order", [1, 2])
def test_commuting_terms_are_exact(order):
    terms = [0.3 * Z, -0.8 * Z]
    np.testing.assert_allclose(trotter_unitary(terms, 1.3, 4, order),
                               exact_unitary(terms, 1.3), atol=1e-12)


def test_first_order_application_order():
    expected = expm(-1j * Z * 0.2) @ expm(-1j * X * 0.2)
    np.testing.assert_allclose(trotter_unitary([X, Z], 0.2, 1), expected)


@pytest.mark.parametrize("order,lower,upper", [(1, 1.8, 2.2), (2, 3.7, 4.3)])
def test_global_convergence_order(order, lower, upper):
    terms = [0.7 * X, 0.9 * Z]
    exact = exact_unitary(terms, 0.8)
    errors = [np.linalg.norm(trotter_unitary(terms, 0.8, n, order) - exact, 2)
              for n in (32, 64)]
    assert lower < errors[0] / errors[1] < upper


def test_second_order_time_reversal():
    terms = ising_terms(2)
    forward = trotter_unitary(terms, 0.8, 9, 2)
    backward = trotter_unitary(terms, -0.8, 9, 2)
    np.testing.assert_allclose(backward, forward.conj().T, atol=1e-12)


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_invalid_steps(steps):
    with pytest.raises(ValueError):
        trotter_unitary([X], 1, steps)


@pytest.mark.parametrize("terms", [[], [np.ones((2, 3))], [X, np.eye(3)],
                                   [np.array([[0, 1], [0, 0]])],
                                   [np.full((2, 2), np.nan)]])
def test_invalid_terms(terms):
    with pytest.raises(ValueError):
        exact_unitary(terms, 1)


def test_invalid_time_and_order():
    for time in (np.nan, np.inf, 1j):
        with pytest.raises(ValueError):
            exact_unitary([X], time)
    with pytest.raises(ValueError):
        trotter_unitary([X], 1, 2, order=3)


def test_ising_convention():
    expected = -np.kron(Z, Z) - 0.7 * (np.kron(X, np.eye(2))
                                       + np.kron(np.eye(2), X))
    np.testing.assert_allclose(sum(ising_terms(2)), expected)
    np.testing.assert_allclose(sum(ising_terms(1)), -0.7 * X)


@pytest.mark.parametrize("spins", [0, -1, 2.5, True])
def test_invalid_spin_count(spins):
    with pytest.raises(ValueError):
        ising_terms(spins)

