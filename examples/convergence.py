"""Run with: python examples/convergence.py"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trotterization import exact_unitary, ising_terms, trotter_unitary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spins", type=int, default=3)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("results/convergence.png"))
    args = parser.parse_args()
    if not 1 <= args.spins <= 8:
        parser.error("Use 1 to 8 spins for this dense demonstration.")
    terms = ising_terms(args.spins)
    reference = exact_unitary(terms, args.time)
    steps = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    print("steps,order,operator_error")
    for order in (1, 2):
        errors = []
        for count in steps:
            approximation = trotter_unitary(terms, args.time, int(count), order)
            error = np.linalg.norm(approximation - reference, ord=2)
            errors.append(error)
            print(f"{count},{order},{error:.8e}")
        ax.loglog(steps, np.maximum(errors, 1e-16), "o-", label=f"Order {order}")
    ax.set(xlabel="Trotter steps", ylabel="Spectral norm error",
           title=f"Ising chain: {args.spins} spins, t = {args.time}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

