"""Open-boundary transverse-field Ising chain in the computational basis."""
from numbers import Integral

import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
I = np.eye(2, dtype=complex)


def _product(operators):
    result = np.array([[1]], dtype=complex)
    for operator in operators:
        result = np.kron(result, operator)
    return result


def ising_terms(n_spins, coupling=1.0, field=0.7):
    """Return individual terms of H = -J sum Z_j Z_(j+1) - h sum X_j.

    Site zero is the leftmost tensor factor. Dense matrices scale as 4**N.
    """
    if (isinstance(n_spins, bool) or not isinstance(n_spins, Integral)
            or n_spins < 1):
        raise ValueError("n_spins must be a positive integer.")
    for value in (coupling, field):
        if not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value):
            raise ValueError("Coupling and field must be finite real scalars.")
    terms = []
    for site in range(n_spins - 1):
        operators = [I] * n_spins
        operators[site] = operators[site + 1] = Z
        terms.append(-float(np.real(coupling)) * _product(operators))
    for site in range(n_spins):
        operators = [I] * n_spins
        operators[site] = X
        terms.append(-float(np.real(field)) * _product(operators))
    return terms

