# Trotterization

A hands-on introduction to quantum Hamiltonian simulation using product formulas.
Start with small spin systems, compare against exact evolution, and see how error
changes with the number of Trotter steps.

## Quick start

Python 3.10 or newer:

```bash
git clone https://github.com/Ruchir555/trotterization.git
cd trotterization
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/convergence.py
```

The example prints a CSV-style error table and saves
`results/convergence.png`. Try `--spins 4 --time 2`.

## Physics

Set $\hbar=1$ and write a time-independent Hamiltonian as
$H=\sum_{j=1}^{m}H_j$. Exact evolution is $U(t)=e^{-iHt}$.
For noncommuting terms, exponentiating the sum is generally different from
multiplying the individual exponentials.

With $\Delta t=t/r$, this implementation applies terms to a state in the
supplied order:

$$S_1(\Delta t)=e^{-iH_m\Delta t}\cdots e^{-iH_1\Delta t},
\qquad U_1(t)=S_1(t/r)^r.$$

For two terms, the symmetric second-order formula is

$$S_2(\Delta t)=e^{-iH_1\Delta t/2}e^{-iH_2\Delta t}
e^{-iH_1\Delta t/2},\qquad U_2(t)=S_2(t/r)^r.$$

For more terms, the code uses the corresponding forward/reverse symmetric
sequence with the last term taking a full step. For fixed finite matrices and
fixed total time, first-order global error scales as $O(r^{-1})$ and second-order
as $O(r^{-2})$ in the asymptotic regime. Commuting terms are exact up to numerical
roundoff. More steps cost more work; very small errors eventually meet a
floating-point floor.

## Example model

An open-boundary transverse-field Ising chain:

$$H=-J\sum_{j=0}^{N-2}Z_jZ_{j+1}-h\sum_{j=0}^{N-1}X_j.$$

Here $X$ and $Z$ are Pauli matrices, not spin operators $\sigma/2$.
Defaults are $J=1$, $h=0.7$, $N=3$. Site zero is the leftmost tensor factor.
Each bond and field contributes a separate term to the product formula.

```python
import numpy as np
from trotterization import exact_unitary, ising_terms, trotter_unitary

terms = ising_terms(3, coupling=1.0, field=0.7)
exact = exact_unitary(terms, time=1.0)
approx = trotter_unitary(terms, time=1.0, steps=32, order=2)
psi0 = np.eye(8, dtype=complex)[:, 0]  # |000>
psi = approx @ psi0
error = np.linalg.norm(approx - exact, ord=2)
print(error, np.linalg.norm(psi))
```

The plot reports spectral norm operator error, a worst-case bound on state-vector
error. It is not state infidelity. For normalised pure states, fidelity is
$|\langle\psi_{\rm exact}|\psi_{\rm approx}\rangle|^2$.

## Learning workflow

1. Run the tests and the default convergence example.
2. Read `src/trotterization/evolution.py`; identify the step size and matrix order.
3. Replace the model with one qubit, $H=0.7X+0.9Z$.
4. Double the step count: asymptotic errors should shrink by about 2 and 4.
5. Use two commuting terms, such as $X$ and $2X$, and explain the result.
6. Compare state fidelity and an observable against the operator error.
7. Change a test deliberately, push a branch, and inspect the failed Actions run.
8. Fix it, push again, and check the passing run before opening a pull request.

## Tests and continuous integration

`python -m pytest -q` checks unitarity, zero-time evolution, commuting terms,
application order, convergence rates, time reversal, model conventions, and
invalid inputs. GitHub Actions repeats tests on Python 3.10–3.13 for pushes and
pull requests, runs the plotting example, and uploads its figure as an artifact.

## Scope and next experiments

This is a dense classical reference implementation for small systems.
Storage scales as $4^N$ and dense matrix operations become expensive rapidly.
It does not compile quantum circuits, simulate noise, or claim quantum advantage.

Useful extensions: compare different term orderings; group commuting terms;
measure magnetisation; add a Heisenberg model; implement a circuit backend and
compare its gate counts. Keep these separate from the exact reference.

## Licence

MIT. See [LICENSE](LICENSE).

