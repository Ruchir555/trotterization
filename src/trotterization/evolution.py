"""Time-independent Hamiltonian evolution, with hbar = 1."""
from numbers import Integral

import numpy as np
from scipy.linalg import expm


def _validate(terms, time):
    matrices = tuple(np.asarray(term, dtype=complex) for term in terms)
    if not matrices:
        raise ValueError("Provide at least one Hamiltonian term.")
    shape = matrices[0].shape
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] == 0:
        raise ValueError("Hamiltonian terms must be nonempty square matrices.")
    for term in matrices:
        if term.shape != shape or not np.isfinite(term).all():
            raise ValueError("Terms must have identical shapes and finite entries.")
        if not np.allclose(term, term.conj().T, rtol=1e-12, atol=1e-12):
            raise ValueError("Hamiltonian terms must be Hermitian.")
    if not np.isscalar(time) or not np.isreal(time) or not np.isfinite(time):
        raise ValueError("Time must be a finite real scalar.")
    return matrices, float(np.real(time))


def exact_unitary(terms, time):
    """Return exp(-i * sum(terms) * time). Uses dense matrix exponentiation."""
    matrices, time = _validate(terms, time)
    return expm(-1j * time * sum(matrices))


def trotter_unitary(terms, time, steps, order=1):
    """Approximate evolution with Lie-Trotter (1) or symmetric Strang (2).

    Terms are applied to a state in their supplied order, so the first
    term's exponential appears rightmost in the step matrix.
    """
    matrices, time = _validate(terms, time)
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("Steps must be a positive integer.")
    if isinstance(order, bool) or order not in (1, 2):
        raise ValueError("Order must be 1 or 2.")
    dt = time / steps
    step = np.eye(matrices[0].shape[0], dtype=complex)
    if order == 1:
        schedule = [(term, dt) for term in matrices]
    else:
        schedule = ([(term, dt / 2) for term in matrices[:-1]]
                    + [(matrices[-1], dt)]
                    + [(term, dt / 2) for term in reversed(matrices[:-1])])
    for term, duration in schedule:
        step = expm(-1j * duration * term) @ step
    return np.linalg.matrix_power(step, int(steps))

