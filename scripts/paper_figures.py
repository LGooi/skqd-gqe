
# Recreate Figures 3, 4, 5, 6 of the GQE-for-QSCI paper for
# N2 (10e, 8o).
#
#   Fig 3 (optimization history)      — R = 2.5 A, L = 10, d_max = 170
#   Fig 4 (sampling efficiency)       — R in {1.1, 1.8, 2.5} A
#   Fig 5 (two-qubit-gate efficiency) — R in {1.1, 1.8, 2.5} A
#   Fig 6 (wavefunction compactness)  — R in {1.1, 1.8, 2.5} A
#
# The paper compares six QSCI input states:
#   - LUCJ (Local Unitary Cluster Jastrow), varying layers
#   - SqDRIFT (randomized Trotter), varying qDRIFT length N
#   - single/multiple time-evolved (first-order Trotter)    [not run]
#   - GQE-optimized ansatz                                  [Fig 3 only]
#   - Random-circuit baseline                               [Fig 3 only]
#   - exact CASCI ground-state sampling                     [reference]
#
# For Figs 4-6 we only reproduce LUCJ / SqDRIFT / exact-CASCI because the
# time-evolved / GQE variants require substantial extra machinery. For Fig 3
# we run the GQE training pipeline in-process (without W&B / Hydra) and
# capture per-iteration best-so-far metrics for GQE-optimized, Local-refined,
# Global-refined and a Random-circuit baseline. All quantum "sampling" is done
# by classical state-vector simulation (`cudaq.sample` on `qpp-cpu`), and the
# QSCI post-processing (symmetry completion + PyCI diagonalization) is shared.
#
# Usage:
#   python scripts/paper_figures.py                       # run + plot all
#   python scripts/paper_figures.py --figures 4,5,6       # only Figs 4-6
#   python scripts/paper_figures.py --figures 3           # only Fig 3
#   python scripts/paper_figures.py --skip-run            # re-plot from cache
#   python scripts/paper_figures.py --molecule h2         # H2 sanity check
#
# Results are cached under `outputs/paper_figures/`:
#   fig3_<mol>_<Rtag>.json            — per-seed optimization trajectories
#   <mol>_<Rtag>.json                 — Fig 4/5/6 curves per geometry
# Delete a JSON (or pass a partial `--figures`) to force a re-run.

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np


def log(msg: str) -> None:
    print(msg, flush=True)

# Make gqe_qsci importable when running the script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cudaq
import pyci
import ffsim
import ffsim.qiskit as fqk
from qiskit import QuantumCircuit, transpile

from gqe_qsci.molecule import PySCFMolecule
from gqe_qsci.qsci.subspace import DeterminantSubspace
from gqe_qsci.qsci.determinant import Determinant
from gqe_qsci.skqd.hamiltonian import hamiltonian_to_cudaq_spin
from gqe_qsci.gqe.circuit_metrics import (
    CircuitMetrics,
    _Frontier,
    pauli_evolution_gates,
)


# --- CLI --------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', type=str, default='outputs/paper_figures')
parser.add_argument('--skip-run', action='store_true',
                    help='Skip benchmarks and re-plot from cached JSON.')
parser.add_argument('--molecule', type=str, default='n2', choices=['n2', 'h2'],
                    help='n2: full 3-geometry benchmark; h2: quick sanity check.')
parser.add_argument('--geometries', type=float, nargs='+', default=None,
                    help='Override bond lengths (Angstrom) for the Fig 4-6 sweep.')
parser.add_argument('--figures', type=str, default='all',
                    help='Which figures to (re)generate. Comma-separated ints '
                         'or "all". Example: --figures 3,4,5')
parser.add_argument('--shots-fig4', type=int, nargs='+',
                    default=[1_000, 3_162, 10_000, 31_623, 100_000, 316_228, 1_000_000],
                    help='Shot counts for sampling-efficiency figure.')
parser.add_argument('--shots-fig5-6', type=int, default=1_000_000,
                    help='Fixed shot budget for gate-efficiency and '
                         'compactness figures.')
parser.add_argument('--lucj-layers', type=int, nargs='+', default=[1, 2, 3, 4, 5])
parser.add_argument('--sqdrift-N', type=int, nargs='+', default=[10, 25, 50, 100, 150, 200])
parser.add_argument('--sqdrift-krylov', type=int, default=3,
                    help='Number of Krylov states used by SqDRIFT (paper: 3).')
parser.add_argument('--sqdrift-nr', type=int, default=50,
                    help='Number of qDRIFT realizations per Krylov state '
                         '(paper: 500, reduced here for tractability).')
parser.add_argument('--sqdrift-dt', type=float, default=0.5,
                    help='Reference evolution time for SqDRIFT.')
# Figure 3 (optimization history) knobs
parser.add_argument('--fig3-R', type=float, default=2.5,
                    help='Bond length for Fig 3 (paper: 2.5 A, stretched N2).')
parser.add_argument('--fig3-L', type=int, default=10,
                    help='Number of operators per generated circuit '
                         '(paper: 10).')
parser.add_argument('--fig3-max-dim', type=int, default=170,
                    help='Determinant budget d_max for Fig 3 (paper: 170).')
parser.add_argument('--fig3-shots', type=int, default=100_000,
                    help='Shot budget per circuit for Fig 3 (paper: 1e5).')
parser.add_argument('--fig3-iters', type=int, default=100,
                    help='Optimization iterations for Fig 3 (paper: 100).')
parser.add_argument('--fig3-seeds', type=int, default=3,
                    help='Number of independent optimization seeds '
                         '(paper: 5). Runtime scales linearly.')
parser.add_argument('--fig3-num-samples', type=int, default=10,
                    help='Circuits generated per iteration.')
parser.add_argument('--fig3-step-per-epoch', type=int, default=30,
                    help='GRPO gradient steps per iteration.')
# GQE-SKQD knobs (Fig 3, Fig 4/5/6). Same SKQD reward pipeline used everywhere.
parser.add_argument('--gqe-skqd', dest='gqe_skqd', action='store_true', default=True,
                    help='Include the GQE-SKQD curves (default: on).')
parser.add_argument('--no-gqe-skqd', dest='gqe_skqd', action='store_false',
                    help='Disable the GQE-SKQD curves.')
parser.add_argument('--gqe-skqd-krylov-dim', type=int, default=2,
                    help='Krylov dimension for SKQD post-processing (default: 2).')
parser.add_argument('--gqe-skqd-dt', type=float, default=0.1,
                    help='dt for the SKQD Trotter evolution.')
parser.add_argument('--gqe-skqd-trotter-steps', type=int, default=2,
                    help='Trotter steps per SKQD Krylov step.')
parser.add_argument('--gqe-skqd-shots', type=int, default=1000,
                    help='Per-Krylov-state shots for SKQD sampling during '
                         'training (kept small for tractability).')
parser.add_argument('--fig456-gqe-L', type=int, default=10,
                    help='Circuit length L for the GQE-SKQD ansatz trained '
                         'once per geometry for Figs 4/5/6 (paper: 140).')
parser.add_argument('--fig456-gqe-iters', type=int, default=20,
                    help='Training iterations for the shared Fig 4/5/6 GQE-SKQD '
                         'ansatz (paper: 100).')
parser.add_argument('--fig456-gqe-num-samples', type=int, default=10,
                    help='Circuits generated per iteration during Fig 4/5/6 '
                         'GQE-SKQD ansatz training.')
parser.add_argument('--fig456-gqe-step-per-epoch', type=int, default=15,
                    help='GRPO gradient steps per iteration during Fig 4/5/6 '
                         'GQE-SKQD ansatz training.')
parser.add_argument('--seed', type=int, default=43)
parser.add_argument('--max-dim', type=int, default=100_000)
args = parser.parse_args()


def _parse_figures(spec: str) -> set[int]:
    spec = spec.strip().lower()
    if spec == 'all':
        return {3, 4, 5, 6}
    out: set[int] = set()
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            f = int(tok)
        except ValueError:
            raise SystemExit(f'--figures: cannot parse "{tok}"')
        if f not in (3, 4, 5, 6):
            raise SystemExit(f'--figures: {f} not in {{3, 4, 5, 6}}')
        out.add(f)
    if not out:
        raise SystemExit('--figures: no figures selected')
    return out


FIGURES = _parse_figures(args.figures)
NEED_R_SWEEP = bool({4, 5, 6} & FIGURES)
NEED_SAMPLING_CURVES = bool({4, 6} & FIGURES)
NEED_GATE_CURVES = 5 in FIGURES
NEED_FIG3 = 3 in FIGURES


# --- Environment setup ------------------------------------------------------

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
np.random.seed(args.seed)
rng = np.random.default_rng(args.seed)
cudaq.set_random_seed(args.seed)
cudaq.set_target('qpp-cpu')

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)


# --- Molecule helpers -------------------------------------------------------

if args.molecule == 'n2':
    default_geoms = [1.1, 1.8, 2.5]
    atoms = ['N', 'N']
    nelecas, norbcas = 10, 8
else:  # h2 quick mode
    default_geoms = [0.7474]
    atoms = ['H', 'H']
    nelecas, norbcas = 2, 2

geometries = args.geometries if args.geometries else default_geoms


def build_molecule(bond_length: float) -> PySCFMolecule:
    return PySCFMolecule(
        geometry=SimpleNamespace(
            type='linear_chain', atoms=atoms, bond_length=bond_length,
        ),
        basis='sto-3g',
        nelecas=nelecas,
        norbcas=norbcas,
        spin=0,
        charge=0,
    )


# --- Common QSCI diagonalizer ----------------------------------------------

class QSCIDiagonalizer:
    def __init__(self, molecule: PySCFMolecule):
        self.molecule = molecule
        h2 = np.asarray(
            molecule.cas_hamiltonian.h2.transpose(0, 2, 1, 3), order='C'
        )
        self.pyci_ham = pyci.hamiltonian(
            molecule.cas_hamiltonian.e_core, molecule.cas_hamiltonian.h1, h2,
        )

    def diagonalize(self, subspace: DeterminantSubspace) -> float:
        if subspace.ndet == 0:
            return float('inf')
        wfn = pyci.fullci_wfn(self.pyci_ham.nbasis, *self.molecule.nelec)
        for det in subspace.determinants:
            wfn.add_det(det)
        op = pyci.sparse_op(self.pyci_ham, wfn)
        energy, _ = op.solve(maxiter=200)
        return float(energy[0])

    def process(self, subspace: DeterminantSubspace) -> tuple[int, int, int, float]:
        """Full QSCI post-processing pipeline: post-select, symmetry-complete,
        truncate, diagonalize.

        Returns (n_sampled, n_valid, n_final, energy).
        """
        n_sampled = subspace.ndet
        valid = subspace.post_select_by_nelec(self.molecule.nelec)
        n_valid = valid.ndet
        truncated = valid.enlarge(
            max_dim=args.max_dim, method='symmetry_completion',
        )
        energy = self.diagonalize(truncated)
        return n_sampled, n_valid, truncated.ndet, energy


# --- LUCJ sampler -----------------------------------------------------------

class LUCJSampler:
    """Prepare U_LUCJ|HF> exactly and multinomially sample bitstrings."""

    def __init__(self, molecule: PySCFMolecule, n_layers: int):
        self.molecule = molecule
        self.norb = molecule.norb
        self.nelec = molecule.nelec
        self.n_layers = n_layers
        molecule.compute_ccsd()  # cached
        t1 = molecule.ccsd_amplitude['t1']
        t2 = molecule.ccsd_amplitude['t2']
        # Local UCJ interaction pattern
        pairs_aa = [(p, p + 1) for p in range(self.norb - 1)]
        pairs_ab = [(p, p) for p in range(self.norb)]
        self.op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=t2, t1=t1, n_reps=n_layers,
            interaction_pairs=(pairs_aa, pairs_ab),
        )
        vec = ffsim.hartree_fock_state(self.norb, self.nelec)
        vec = ffsim.apply_unitary(vec, self.op, self.norb, self.nelec)
        self._probs = np.abs(vec) ** 2
        self._probs /= self._probs.sum()

    def sample(self, shots: int) -> DeterminantSubspace:
        addr = rng.choice(len(self._probs), size=shots, p=self._probs)
        strings = ffsim.addresses_to_strings(
            addr, self.norb, self.nelec,
            bitstring_type=ffsim.BitstringType.STRING,
        )
        counter = Counter(strings)
        dets: list[Determinant] = []
        seen: set[tuple[int, int]] = set()
        for bits, _ in counter.most_common():
            alpha = int(bits[self.norb:], 2)
            beta = int(bits[:self.norb], 2)
            key = (alpha, beta)
            if key in seen:
                continue
            seen.add(key)
            dets.append(Determinant([alpha, beta]))
        return DeterminantSubspace(dets)

    def measure_circuit(self) -> CircuitMetrics:
        n_qubits = 2 * self.norb
        qc = QuantumCircuit(n_qubits)
        qc.append(fqk.PrepareHartreeFockJW(self.norb, self.nelec), range(n_qubits))
        qc.append(fqk.UCJOpSpinBalancedJW(self.op), range(n_qubits))
        qct = transpile(
            qc,
            basis_gates=['h', 's', 'sdg', 'x', 'y', 'z', 'rz', 'ry', 'rx', 'cx'],
            optimization_level=0,
        )
        gates = [
            (inst.operation.name,
             tuple(qct.find_bit(q).index for q in inst.qubits))
            for inst in qct.data
        ]
        counts: Counter = Counter(name for name, _ in gates)
        frontier = _Frontier(n_qubits)
        frontier.add_gates(gates)
        return CircuitMetrics(
            n_qubits=n_qubits,
            depth=frontier.depth(),
            gate_counts=counts,
            two_qubit_depth=frontier.two_qubit_depth(),
        )


# --- SqDRIFT sampler --------------------------------------------------------

def _term_coefficients(op: cudaq.SpinOperator) -> list[complex]:
    return [term.evaluate_coefficient() for term in op]


def _term_words(op: cudaq.SpinOperator, n_qubits: int) -> list[cudaq.pauli_word]:
    return [term.get_pauli_word(n_qubits) for term in op]


@cudaq.kernel
def _sqdrift_kernel(
    n_qubits: int,
    n_electrons: int,
    qdrift_coeffs: list[float],
    qdrift_words: list[cudaq.pauli_word],
):
    q = cudaq.qvector(n_qubits)
    for i in range(n_electrons):
        x(q[i])
    for i in range(len(qdrift_coeffs)):
        exp_pauli(qdrift_coeffs[i], q, qdrift_words[i])
    mz(q)


class SqDRIFTSampler:
    """Mirror of pure_skqd.py's SqDRIFT sampling flow, minus MPI/CUDA.

    Runs the union-of-Krylov-subspaces sampling protocol:
      for each k in [0, krylov_dim):
        for each of n_r qDRIFT realizations:
          - draw N Pauli terms i.i.d. from H
          - apply exp(-i sign(c_j) P_j * k * dt * lambda / N)
          - sample S bitstrings, union into determinant set
    """

    def __init__(
        self,
        molecule: PySCFMolecule,
        krylov_dim: int,
        qdrift_length: int,
        n_randomizations: int,
        dt: float,
    ):
        self.molecule = molecule
        self.norb = molecule.norb
        self.nelec = molecule.nelec
        self.n_qubits = 2 * self.norb
        self.n_electrons = int(sum(self.nelec))
        self.krylov_dim = krylov_dim
        self.qdrift_length = qdrift_length
        self.n_randomizations = n_randomizations
        self.dt = dt

        spin_ham = hamiltonian_to_cudaq_spin(molecule)
        coeffs = [c.real for c in _term_coefficients(spin_ham)]
        words = _term_words(spin_ham, self.n_qubits)
        # Drop identity terms
        keep = [i for i, w in enumerate(words) if any(ch != 'I' for ch in str(w))]
        self.h_coeffs = [coeffs[i] for i in keep]
        self.h_words = [words[i] for i in keep]
        abs_coeffs = np.abs(np.asarray(self.h_coeffs, dtype=np.float64))
        self.lam = float(abs_coeffs.sum())
        self.probs = abs_coeffs / self.lam
        self.signs = np.sign(np.asarray(self.h_coeffs, dtype=np.float64))

    def _sample_qdrift(self, k: int, N: int) -> tuple[list[float], list[cudaq.pauli_word]]:
        if k == 0 or N == 0:
            return [], []
        idxs = rng.choice(len(self.h_coeffs), size=N, p=self.probs)
        angle = 2.0 * k * self.dt * self.lam / N
        return [angle * float(self.signs[i]) for i in idxs], [self.h_words[i] for i in idxs]

    def sample(self, shots_per_realization: int) -> DeterminantSubspace:
        seen: set[tuple[int, int]] = set()
        dets: list[Determinant] = []
        for k in range(self.krylov_dim):
            n_reals = 1 if k == 0 else self.n_randomizations
            for _ in range(n_reals):
                coeffs, words = self._sample_qdrift(k, self.qdrift_length)
                result = cudaq.sample(
                    _sqdrift_kernel,
                    self.n_qubits, self.n_electrons, coeffs, words,
                    shots_count=shots_per_realization,
                )
                for det in DeterminantSubspace.from_cudaq_sample_result(result).determinants:
                    key = (int(det[0]), int(det[1]))
                    if key not in seen:
                        seen.add(key)
                        dets.append(det)
        return DeterminantSubspace(dets)

    def measure_circuit(self) -> CircuitMetrics:
        # Deepest realization is k = krylov_dim - 1, N Pauli evolutions.
        idxs = rng.choice(len(self.h_coeffs), size=self.qdrift_length, p=self.probs)
        gates: list = [('x', (q,)) for q in range(self.n_electrons)]
        for i in idxs:
            gates.extend(pauli_evolution_gates(str(self.h_words[i])))
        counts: Counter = Counter(name for name, _ in gates)
        frontier = _Frontier(self.n_qubits)
        frontier.add_gates(gates)
        return CircuitMetrics(
            n_qubits=self.n_qubits,
            depth=frontier.depth(),
            gate_counts=counts,
            two_qubit_depth=frontier.two_qubit_depth(),
        )

    def total_shots_for(self, shots_per_realization: int) -> int:
        # k=0 uses 1 realization, remaining k use n_r realizations.
        return shots_per_realization * (1 + max(self.krylov_dim - 1, 0) * self.n_randomizations)


# --- Exact CASCI reference sampler -----------------------------------------

class CASCIDistribution:
    """Sample from the exact CASCI ground-state distribution p(x)=|<x|Psi>|^2."""

    def __init__(self, molecule: PySCFMolecule):
        from pyscf import fci
        self.molecule = molecule
        self.nelec = molecule.nelec
        self.norb = molecule.norb
        # PySCF FCI to get the CI vector
        cisolver = fci.direct_spin1.FCI()
        e, ci = cisolver.kernel(
            molecule.cas_hamiltonian.h1,
            molecule.cas_hamiltonian.h2,
            self.norb,
            self.nelec,
            ecore=molecule.cas_hamiltonian.e_core,
        )
        self.e_fci = float(e)
        # ci is shape (n_alpha_strings, n_beta_strings), pyscf convention
        self.probs = (np.abs(ci) ** 2).ravel()
        self.probs /= self.probs.sum()

    def sample(self, shots: int) -> DeterminantSubspace:
        idxs = rng.choice(len(self.probs), size=shots, p=self.probs)
        counter = Counter(idxs.tolist())
        dets: list[Determinant] = []
        seen: set[tuple[int, int]] = set()
        for idx, _ in counter.most_common():
            det = Determinant.from_fullci_index(int(idx), self.norb, self.nelec)
            key = (int(det[0]), int(det[1]))
            if key in seen:
                continue
            seen.add(key)
            dets.append(det)
        return DeterminantSubspace(dets)


# --- Benchmark runners ------------------------------------------------------

@dataclass
class Point:
    x: float
    energy: float
    subspace_dim: int
    n_sampled: int
    n_valid: int
    extra: dict = field(default_factory=dict)


def run_lucj_sampling_curve(mol: PySCFMolecule, layers: int,
                            shot_grid: Sequence[int]) -> list[Point]:
    sampler = LUCJSampler(mol, n_layers=layers)
    diag = QSCIDiagonalizer(mol)
    out = []
    for shots in shot_grid:
        t0 = time.time()
        subspace = sampler.sample(shots)
        dt = time.time() - t0
        n_s, n_v, n_f, e = diag.process(subspace)
        log(f'    [lucj L={layers} shots={shots}] E={e:.6f} ndet={n_f} '
            f'sampled_dets={n_s} ({dt:.1f}s)')
        out.append(Point(x=shots, energy=e, subspace_dim=n_f, n_sampled=n_s, n_valid=n_v))
    return out


def run_sqdrift_sampling_curve(mol: PySCFMolecule, N: int,
                               shot_grid: Sequence[int]) -> list[Point]:
    diag = QSCIDiagonalizer(mol)
    out = []
    for total_shots in shot_grid:
        # allocate shots across (1 + (krylov_dim-1)*n_r) realizations
        n_reals = 1 + max(args.sqdrift_krylov - 1, 0) * args.sqdrift_nr
        per = max(total_shots // n_reals, 1)
        sampler = SqDRIFTSampler(
            mol,
            krylov_dim=args.sqdrift_krylov,
            qdrift_length=N,
            n_randomizations=args.sqdrift_nr,
            dt=args.sqdrift_dt,
        )
        t0 = time.time()
        subspace = sampler.sample(per)
        dt = time.time() - t0
        n_s, n_v, n_f, e = diag.process(subspace)
        log(f'    [sqdrift N={N} shots={total_shots}] E={e:.6f} '
            f'ndet={n_f} sampled_dets={n_s} ({dt:.1f}s)')
        out.append(Point(x=total_shots, energy=e, subspace_dim=n_f, n_sampled=n_s, n_valid=n_v,
                         extra={'per_realization_shots': per, 'n_realizations': n_reals}))
    return out


def run_exact_sampling_curve(mol: PySCFMolecule,
                             shot_grid: Sequence[int]) -> tuple[float, list[Point]]:
    dist = CASCIDistribution(mol)
    diag = QSCIDiagonalizer(mol)
    out = []
    for shots in shot_grid:
        t0 = time.time()
        subspace = dist.sample(shots)
        dt = time.time() - t0
        n_s, n_v, n_f, e = diag.process(subspace)
        log(f'    [exact shots={shots}] E={e:.6f} ndet={n_f} sampled_dets={n_s} ({dt:.1f}s)')
        out.append(Point(x=shots, energy=e, subspace_dim=n_f, n_sampled=n_s, n_valid=n_v))
    return dist.e_fci, out


def run_gate_efficiency_curve_lucj(mol: PySCFMolecule,
                                   layer_grid: Sequence[int],
                                   shots: int) -> list[Point]:
    diag = QSCIDiagonalizer(mol)
    out = []
    for L in layer_grid:
        t0 = time.time()
        sampler = LUCJSampler(mol, n_layers=L)
        cm = sampler.measure_circuit()
        subspace = sampler.sample(shots)
        dt = time.time() - t0
        n_s, n_v, n_f, e = diag.process(subspace)
        log(f'    [lucj-gate L={L}] cx={cm.cx_count} depth={cm.depth} '
            f'E={e:.6f} ndet={n_f} ({dt:.1f}s)')
        out.append(Point(
            x=cm.cx_count, energy=e, subspace_dim=n_f, n_sampled=n_s, n_valid=n_v,
            extra={
                'layers': L,
                'depth': cm.depth,
                'two_qubit_depth': cm.two_qubit_depth,
                'total_gates': cm.total_gates,
                'rotation_gates': cm.gate_counts.get('rz', 0)
                                  + cm.gate_counts.get('rx', 0)
                                  + cm.gate_counts.get('ry', 0),
            },
        ))
    return out


def run_gate_efficiency_curve_sqdrift(mol: PySCFMolecule,
                                      N_grid: Sequence[int],
                                      shots: int) -> list[Point]:
    diag = QSCIDiagonalizer(mol)
    out = []
    for N in N_grid:
        n_reals = 1 + max(args.sqdrift_krylov - 1, 0) * args.sqdrift_nr
        per = max(shots // n_reals, 1)
        sampler = SqDRIFTSampler(
            mol,
            krylov_dim=args.sqdrift_krylov,
            qdrift_length=N,
            n_randomizations=args.sqdrift_nr,
            dt=args.sqdrift_dt,
        )
        cm = sampler.measure_circuit()
        t0 = time.time()
        subspace = sampler.sample(per)
        dt = time.time() - t0
        n_s, n_v, n_f, e = diag.process(subspace)
        log(f'    [sqdrift-gate N={N}] cx={cm.cx_count} depth={cm.depth} '
            f'E={e:.6f} ndet={n_f} ({dt:.1f}s)')
        out.append(Point(
            x=cm.cx_count, energy=e, subspace_dim=n_f, n_sampled=n_s, n_valid=n_v,
            extra={
                'N': N,
                'depth': cm.depth,
                'two_qubit_depth': cm.two_qubit_depth,
                'total_gates': cm.total_gates,
                'rotation_gates': cm.gate_counts.get('rz', 0),
            },
        ))
    return out


# --- Serialisation ----------------------------------------------------------

def points_to_json(points: list[Point]) -> list[dict]:
    return [
        {'x': p.x, 'energy': p.energy, 'subspace_dim': p.subspace_dim,
         'n_sampled': p.n_sampled, 'n_valid': p.n_valid, 'extra': p.extra}
        for p in points
    ]


def points_from_json(js: list[dict]) -> list[Point]:
    return [Point(x=p['x'], energy=p['energy'], subspace_dim=p['subspace_dim'],
                  n_sampled=p['n_sampled'], n_valid=p['n_valid'], extra=p.get('extra', {}))
            for p in js]


def geom_stem(R: float) -> str:
    return f'R{R:.3f}'.replace('.', 'p')


def result_file(R: float) -> Path:
    return out_dir / f'{args.molecule}_{geom_stem(R)}.json'


def fig3_result_file() -> Path:
    return out_dir / f'fig3_{args.molecule}_{geom_stem(args.fig3_R)}.json'


# --- Fig 4/5/6: benchmark orchestration ------------------------------------

_FIG456_SAMPLING_KEYS = (
    'lucj3_sampling', 'sqdrift_sampling', 'exact_sampling',
)
_FIG456_GATE_KEYS = (
    'lucj_gate_efficiency', 'sqdrift_gate_efficiency',
)


def _ansatz_cache_file(R: float, L: int) -> Path:
    return out_dir / f'gqe_skqd_ansatz_{args.molecule}_{geom_stem(R)}_L{L}.json'


def _ensure_gqe_skqd_ansatz(R: float, L: int) -> list[int]:
    """Train one GQE-SKQD seed at this (R, L) and cache the best operator
    sequence to disk. Returns the cached `list[int]` on subsequent calls."""
    cache = _ansatz_cache_file(R, L)
    if cache.exists():
        with cache.open() as f:
            payload = json.load(f)
        seq = [int(x) for x in payload['best_seq']]
        log(f'  loaded cached GQE-SKQD ansatz (R={R} L={L}) from {cache}')
        return seq

    log(f'  training GQE-SKQD ansatz for R={R} L={L} '
        f'(iters={args.fig456_gqe_iters}, num_samples={args.fig456_gqe_num_samples})')

    import torch
    if torch.cuda.is_available():
        trainer_kwargs = {'precision': '16-mixed', 'accelerator': 'gpu'}
    else:
        trainer_kwargs = {'precision': '32-true', 'accelerator': 'cpu'}

    seed = args.seed
    run_dir = out_dir / f'fig456_ansatz_runs_{args.molecule}_{geom_stem(R)}_L{L}'
    cfg = _build_fig3_config(
        seed=seed, R=R, L=L,
        max_dim=args.max_dim,
        shots=args.fig3_shots,
        n_iters=args.fig456_gqe_iters,
        num_samples=args.fig456_gqe_num_samples,
        step_per_epoch=args.fig456_gqe_step_per_epoch,
        output_dir=run_dir,
    )
    captured: list[list[int]] = []
    t0 = time.time()
    _run_gqe_seed(seed, cfg, trainer_kwargs,
                  use_skqd=True, keep_best_seq=captured, tag='ansatz')
    log(f'  ansatz training took {time.time() - t0:.1f}s')

    if not captured:
        raise RuntimeError(
            f'GQE-SKQD ansatz training did not produce a best sequence for R={R} L={L}'
        )
    best_seq = captured[0]
    with cache.open('w') as f:
        json.dump({'R': R, 'L': L, 'seed': seed, 'best_seq': best_seq}, f)
    log(f'  wrote {cache}')
    return best_seq


def _run_gqe_skqd_shot_sweep(R: float,
                             best_seq: list[int],
                             shot_grid: Sequence[int]) -> list[Point]:
    """Fresh SKQDPipeline per total-shots target. Applies the trained GQE-SKQD
    ansatz once per shot budget and reports (energy, subspace_dim, cx_count).
    A single Factory is reused so molecule + operator pool are built once.
    """
    from omegaconf import OmegaConf
    from gqe_qsci.factory import Factory
    import torch

    factory = Factory()
    krylov_dim = int(args.gqe_skqd_krylov_dim)
    out = []
    # best_seq is the raw TrainPipeline sample sequence which already includes
    # the leading identity token; feed it directly (no extra prefix).
    seq_tensor = torch.as_tensor([list(best_seq)], dtype=torch.long)
    for total_shots in shot_grid:
        per_krylov = max(total_shots // max(krylov_dim, 1), 1)
        cfg = _build_fig3_config(
            seed=args.seed, R=R, L=len(best_seq),
            max_dim=args.max_dim, shots=total_shots,
            n_iters=1, num_samples=1, step_per_epoch=1,
            output_dir=out_dir / f'fig456_skqd_R{geom_stem(R)}_shots{total_shots}',
        )
        cfg.skqd = OmegaConf.create({
            'krylov_dim': krylov_dim,
            'dt': float(args.gqe_skqd_dt),
            'trotter_steps': int(args.gqe_skqd_trotter_steps),
            'shots_per_krylov_state': int(per_krylov),
        })
        pipeline = _build_skqd_pipeline(factory, cfg)
        t0 = time.time()
        result = pipeline.process({'idx': seq_tensor})
        dt = time.time() - t0
        s = result.samples[0]
        e = float(s.energy)
        n_final = int(s.subspace_dim)
        n_sampled = int(s.num_sampled_basis or 0)
        n_valid = int(s.num_symmetry_preserving_basis or 0)
        cx = int(s.cx_count or 0)
        n_ops = max(len(best_seq) - 1, 0)
        log(f'    [gqe-skqd R={R} shots={total_shots} per_k={per_krylov}] '
            f'E={e:.6f} ndet={n_final} cx={cx} L={n_ops} ({dt:.1f}s)')
        out.append(Point(x=total_shots, energy=e, subspace_dim=n_final,
                         n_sampled=n_sampled, n_valid=n_valid,
                         extra={'cx_count': cx, 'per_krylov_shots': per_krylov,
                                'krylov_dim': krylov_dim, 'L': n_ops}))
    return out


def _run_gqe_skqd_gate_point(R: float,
                             best_seq: list[int], shots: int) -> Point:
    """Single (cx_count, energy) point for Fig 5 using the same trained ansatz."""
    from omegaconf import OmegaConf
    from gqe_qsci.factory import Factory
    import torch

    factory = Factory()
    krylov_dim = int(args.gqe_skqd_krylov_dim)
    per_krylov = max(shots // max(krylov_dim, 1), 1)
    cfg = _build_fig3_config(
        seed=args.seed, R=R, L=len(best_seq),
        max_dim=args.max_dim, shots=shots,
        n_iters=1, num_samples=1, step_per_epoch=1,
        output_dir=out_dir / f'fig456_skqd_gate_R{geom_stem(R)}',
    )
    cfg.skqd = OmegaConf.create({
        'krylov_dim': krylov_dim,
        'dt': float(args.gqe_skqd_dt),
        'trotter_steps': int(args.gqe_skqd_trotter_steps),
        'shots_per_krylov_state': int(per_krylov),
    })
    pipeline = _build_skqd_pipeline(factory, cfg)
    # best_seq is the raw TrainPipeline sample sequence which already includes
    # the leading identity token; feed it directly (no extra prefix).
    seq_tensor = torch.as_tensor([list(best_seq)], dtype=torch.long)
    t0 = time.time()
    result = pipeline.process({'idx': seq_tensor})
    dt = time.time() - t0
    s = result.samples[0]
    cx = int(s.cx_count or 0)
    n_ops = max(len(best_seq) - 1, 0)
    log(f'    [gqe-skqd-gate R={R} L={n_ops}] '
        f'cx={cx} E={float(s.energy):.6f} ndet={int(s.subspace_dim)} ({dt:.1f}s)')
    return Point(
        x=cx, energy=float(s.energy), subspace_dim=int(s.subspace_dim),
        n_sampled=int(s.num_sampled_basis or 0),
        n_valid=int(s.num_symmetry_preserving_basis or 0),
        extra={'L': n_ops, 'krylov_dim': krylov_dim,
               'shots': shots, 'per_krylov_shots': per_krylov,
               'total_gates': int(s.total_gates or 0)},
    )


def _load_geometry(R: float) -> dict | None:
    path = result_file(R)
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _save_geometry(R: float, data: dict) -> None:
    path = result_file(R)
    with path.open('w') as f:
        json.dump(data, f, indent=2)
    log(f'  wrote {path}')


def ensure_geometry_data(R: float,
                         need_sampling: bool,
                         need_gates: bool) -> dict:
    """Load cached per-R JSON if present; compute any missing curves needed
    for the requested figures and persist the merged result.

    When --skip-run is set, missing curves are left absent (plotters handle
    empty lists gracefully).
    """
    cached = _load_geometry(R) or {}
    if cached:
        log(f'  loaded cache {result_file(R)}')

    missing_sampling = need_sampling and not all(k in cached for k in _FIG456_SAMPLING_KEYS)
    missing_gates = need_gates and not all(k in cached for k in _FIG456_GATE_KEYS)
    need_gqe_skqd_sampling = need_sampling and args.gqe_skqd and 'gqe_skqd_sampling' not in cached
    need_gqe_skqd_gate = need_gates and args.gqe_skqd and 'gqe_skqd_gate' not in cached

    if args.skip_run and (missing_sampling or missing_gates
                          or need_gqe_skqd_sampling or need_gqe_skqd_gate):
        which = []
        if missing_sampling:
            which.append('sampling')
        if missing_gates:
            which.append('gates')
        if need_gqe_skqd_sampling or need_gqe_skqd_gate:
            which.append('gqe-skqd')
        log(f'  --skip-run: leaving {"/".join(which)} curves absent for R={R}')
        return cached

    if not (missing_sampling or missing_gates
            or need_gqe_skqd_sampling or need_gqe_skqd_gate):
        return cached

    mol = build_molecule(R)
    casci_e = float(mol.compute_casci())
    log(f'  CASCI energy: {casci_e:.8f}')

    data: dict = dict(cached)
    data.setdefault('R', R)
    data['casci_energy'] = casci_e

    if missing_sampling:
        log(f'  [fig4/6] LUCJ (3 layers) sampling scan...')
        data['lucj3_sampling'] = points_to_json(
            run_lucj_sampling_curve(mol, layers=3, shot_grid=args.shots_fig4)
        )
        log(f'  [fig4/6] SqDRIFT (N=100) sampling scan...')
        data['sqdrift_sampling'] = points_to_json(
            run_sqdrift_sampling_curve(mol, N=100, shot_grid=args.shots_fig4)
        )
        log(f'  [fig4/6] Exact CASCI sampling scan...')
        _e_fci, exact_pts = run_exact_sampling_curve(mol, args.shots_fig4)
        data['exact_sampling'] = points_to_json(exact_pts)

    if missing_gates:
        log(f'  [fig5] LUCJ layers scan {args.lucj_layers}...')
        data['lucj_gate_efficiency'] = points_to_json(
            run_gate_efficiency_curve_lucj(mol, args.lucj_layers, args.shots_fig5_6)
        )
        log(f'  [fig5] SqDRIFT N scan {args.sqdrift_N}...')
        data['sqdrift_gate_efficiency'] = points_to_json(
            run_gate_efficiency_curve_sqdrift(mol, args.sqdrift_N, args.shots_fig5_6)
        )

    if need_gqe_skqd_sampling or need_gqe_skqd_gate:
        best_seq = _ensure_gqe_skqd_ansatz(R, args.fig456_gqe_L)
        if need_gqe_skqd_sampling:
            log(f'  [fig4/6] GQE-SKQD sampling scan (L={args.fig456_gqe_L})...')
            data['gqe_skqd_sampling'] = points_to_json(
                _run_gqe_skqd_shot_sweep(R, best_seq, args.shots_fig4)
            )
        if need_gqe_skqd_gate:
            log(f'  [fig5] GQE-SKQD gate point (L={args.fig456_gqe_L})...')
            data['gqe_skqd_gate'] = points_to_json(
                [_run_gqe_skqd_gate_point(R, best_seq, args.shots_fig5_6)]
            )

    # Compactness aliases (fig 6 reuses the sampling scans)
    if 'lucj3_sampling' in data:
        data['lucj3_compactness'] = data['lucj3_sampling']
    if 'sqdrift_sampling' in data:
        data['sqdrift_compactness'] = data['sqdrift_sampling']
    if 'exact_sampling' in data:
        data['exact_compactness'] = data['exact_sampling']
    if 'gqe_skqd_sampling' in data:
        data['gqe_skqd_compactness'] = data['gqe_skqd_sampling']

    _save_geometry(R, data)
    return data


# --- Fig 3: GQE optimization history ---------------------------------------

def _build_fig3_config(seed: int, R: float, L: int, max_dim: int, shots: int,
                       n_iters: int, num_samples: int, step_per_epoch: int,
                       output_dir: Path):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        'project': {'name': 'paper_figures_fig3'},
        'exp_tag': f'fig3-{args.molecule}-{geom_stem(R)}-seed{seed}',
        'output': str(output_dir),
        'ngates': L,
        'reference_keys': [],
        'sampler': {'mpi': False, 'shots': shots},
        'operator_pool': {
            'spec': 'pauli_evolution',
            'params': None,
            'ccsd_threshold': 1e-6,
            'remove_z_ladder': True,
            'only_use_first_pauli': True,
        },
        'qsci': {
            'max_dim': max_dim,
            'enlarge_method': 'symmetry_completion',
            'max_cycle': 1000,
            'eigsh_kwargs': {'tol': 1e-14, 'maxiter': 5000},
        },
        'skqd': {
            'krylov_dim': int(args.gqe_skqd_krylov_dim),
            'dt': float(args.gqe_skqd_dt),
            'trotter_steps': int(args.gqe_skqd_trotter_steps),
            'shots_per_krylov_state': int(args.gqe_skqd_shots),
        },
        'molecule': {
            '_target_': 'gqe_qsci.molecule.PySCFMolecule',
            'geometry': {'type': 'linear_chain',
                         'atoms': list(atoms), 'bond_length': R},
            'basis': 'sto-3g',
            'nelecas': nelecas, 'norbcas': norbcas,
            'spin': 0, 'charge': 0,
        },
        'model': {
            '_target_': 'gqe_qsci.gqe.models.gpt2.GPT2Model',
            'small': True,
            'repetition_penalty': 1.2,
        },
        'trainer': {
            'seed': seed,
            'max_iters': n_iters,
            'num_samples': num_samples,
            'batch_size': num_samples,
            'step_per_epoch': step_per_epoch,
            'warmup_size': num_samples,
            'buffer_size': num_samples,
            'load_checkpoint': False,
            'checkpoint_every_n_iters': 10 ** 9,
            'optimizer': {'lr': 5e-6, 'cls': 'AdamW', 'weight_decay': 0.01},
            'loss': {'type': 'grpo', 'clip_grpo_low': 0.2, 'clip_grpo_high': 0.28},
            'temperature_scheduler': {
                '_target_': 'gqe_qsci.gqe.scheduler.VarBasedScheduler',
                'initial': 0.5, 'delta': 0.02, 'target_var': 1e-5,
            },
        },
        'vocab_size': 0,
    })
    return cfg


def _record_best_so_far(traj: list[dict], iteration: int,
                        gqe_sample, gqe_local, gqe_global) -> None:
    def _e(s): return None if s is None else float(s.energy)
    def _n(s): return None if s is None else int(s.subspace_dim)
    traj.append({
        'iter': int(iteration),
        'energy': _e(gqe_sample), 'ndet': _n(gqe_sample),
        'local_energy': _e(gqe_local), 'local_ndet': _n(gqe_local),
        'global_energy': _e(gqe_global), 'global_ndet': _n(gqe_global),
    })


def _build_skqd_pipeline(factory, cfg):
    """Construct an SKQDPipeline compatible with TrainPipeline's qsci_pipeline
    slot, reusing `factory`'s cached molecule + operator pool.
    """
    import cudaq
    from gqe_qsci.skqd import SKQDPipeline
    from gqe_qsci.skqd.sampler import SKQDSampler

    molecule = factory.create_molecule(cfg)
    pool = factory.create_operator_pool(cfg)
    numQPUs = cudaq.get_target().num_qpus()
    sampler = SKQDSampler(
        pool, mpi=cfg.sampler.mpi, numQPUs=numQPUs,
        shots_count=int(cfg.skqd.shots_per_krylov_state),
    )
    return SKQDPipeline(
        molecule, pool, sampler,
        krylov_dim=int(cfg.skqd.krylov_dim),
        dt=float(cfg.skqd.dt),
        trotter_steps=int(cfg.skqd.trotter_steps),
        max_dim=int(cfg.qsci.max_dim),
        max_cycle=int(cfg.qsci.max_cycle),
        eigsh_kwargs=dict(cfg.qsci.eigsh_kwargs),
    )


def _run_gqe_seed(seed: int, cfg, trainer_kwargs: dict,
                  *, use_skqd: bool = False,
                  keep_best_seq: list | None = None,
                  tag: str = 'gqe') -> list[dict]:
    """Run one GQE training seed and return per-iteration best-so-far metrics.

    When `use_skqd=True`, the reward pipeline is swapped for `SKQDPipeline`
    (so trajectory metrics reflect SKQD post-processing).

    If `keep_best_seq` is provided (as an empty list), the best operator
    sequence found across the run is appended to it as the sole element.
    """
    import pytorch_lightning as pl
    from gqe_qsci.factory import Factory
    from gqe_qsci.train_pipeline import TrainPipeline

    Path(cfg.output).mkdir(parents=True, exist_ok=True)

    trajectory: list[dict] = []
    factory = Factory()

    class _CapturePipeline(TrainPipeline):
        def on_fit_start(self):
            while len(self.buffer) < self.warmup_size:
                self.collect_rollout(log=False)
            pl.LightningModule.on_fit_start(self)

        def on_train_epoch_start(self):
            self.collect_rollout(log=True)
            _record_best_so_far(
                trajectory, self.current_epoch,
                self.best_sample, self.best_local_refined, self.best_global_refined,
            )
            bs = self.best_sample
            log(f'    [{tag} seed={seed} iter={self.current_epoch}] '
                f'best_E={bs.energy:.6f} ndet={bs.subspace_dim}')
            pl.LightningModule.on_train_epoch_start(self)

        def on_save_checkpoint(self, checkpoint):
            return

    pipeline = _CapturePipeline(factory, cfg)
    pipeline.set_seed(seed)

    if use_skqd:
        pipeline.qsci_pipeline = _build_skqd_pipeline(factory, cfg)

    trainer = pl.Trainer(
        logger=False,
        max_epochs=cfg.trainer.max_iters,
        deterministic=True,
        reload_dataloaders_every_n_epochs=1,
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=False,
        enable_checkpointing=False,
        **trainer_kwargs,
    )
    trainer.fit(pipeline)

    if keep_best_seq is not None and pipeline.best_sample is not None:
        seq = pipeline.best_sample.seq
        if seq is not None:
            keep_best_seq.append(list(int(s) for s in seq))

    return trajectory


def _run_random_seed(seed: int, cfg) -> list[dict]:
    """Random-circuit baseline: draw L random operator indices each iteration."""
    import random as _random
    import torch
    from gqe_qsci.factory import Factory

    Path(cfg.output).mkdir(parents=True, exist_ok=True)

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_rng = np.random.default_rng(seed)

    factory = Factory()
    pool = factory.create_operator_pool(cfg)
    vocab = pool.get_vocab_size()
    pipeline = factory.create_qsci_pipeline(cfg)

    L = int(cfg.ngates)
    M = int(cfg.trainer.num_samples)
    n_iters = int(cfg.trainer.max_iters)

    best_sample = None
    best_local = None
    best_global = None
    trajectory: list[dict] = []
    for it in range(n_iters):
        seqs = seed_rng.integers(0, vocab, size=(M, L + 1))
        seqs[:, 0] = 0  # match TrainPipeline: index 0 = identity prefix
        state = {'idx': torch.as_tensor(seqs, dtype=torch.long)}
        result = pipeline.process(state)

        for s in result.samples:
            if best_sample is None or s.energy < best_sample.energy:
                best_sample = s
        if result.local_refined is not None:
            if best_local is None or result.local_refined.energy < best_local.energy:
                best_local = result.local_refined
        if result.global_refined is not None:
            if best_global is None or result.global_refined.energy < best_global.energy:
                best_global = result.global_refined

        _record_best_so_far(trajectory, it, best_sample, best_local, best_global)
        log(f'    [random seed={seed} iter={it}] '
            f'best_E={best_sample.energy:.6f} ndet={best_sample.subspace_dim}')
    return trajectory


def run_fig3() -> dict:
    """Run (or load) Figure 3 optimization-history trajectories.

    Progress is checkpointed after each per-seed run (gqe, random, gqe-skqd),
    so an interrupted invocation can be resumed by re-running the same command:
    completed (seed, method) slots are read back from the cache and skipped.
    """
    cache_path = fig3_result_file()

    if args.skip_run:
        if cache_path.exists():
            with cache_path.open() as f:
                return json.load(f)
        log(f'  --skip-run: {cache_path} not found; Fig 3 will be empty')
        return {}

    R = args.fig3_R
    L = args.fig3_L
    max_dim = args.fig3_max_dim
    shots = args.fig3_shots
    n_iters = args.fig3_iters
    n_seeds = args.fig3_seeds
    num_samples = args.fig3_num_samples
    step_per_epoch = args.fig3_step_per_epoch

    log(f'\n=== Figure 3: N2 (10e,8o) R={R} A, L={L}, d_max={max_dim}, '
        f'shots={shots}, iters={n_iters}, seeds={n_seeds} ===')

    # Reference CASCI energy (cached inside PySCFMolecule)
    mol = build_molecule(R)
    casci_e = float(mol.compute_casci())
    log(f'  CASCI energy: {casci_e:.8f}')

    # Trainer args that differ by hardware. Use fp32 on CPU (fp16 mixed only
    # helps on GPU and can error on CPU).
    import torch
    if torch.cuda.is_available():
        trainer_kwargs = {'precision': '16-mixed', 'accelerator': 'gpu'}
    else:
        trainer_kwargs = {'precision': '32-true', 'accelerator': 'cpu'}

    per_run_out = out_dir / f'fig3_runs_{args.molecule}_{geom_stem(R)}'
    per_run_out.mkdir(parents=True, exist_ok=True)

    trajectories: dict[str, list[list[dict]]] = {'gqe': [], 'random': []}
    if args.gqe_skqd:
        trajectories['gqe_skqd'] = []

    # Resume from a previous checkpoint if the run configuration matches.
    # Mismatched config invalidates the cache to avoid mixing incompatible runs.
    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            with cache_path.open() as f:
                cached = json.load(f)
            same_cfg = (
                cached.get('R') == R
                and cached.get('L') == L
                and cached.get('max_dim') == max_dim
                and cached.get('shots') == shots
                and cached.get('n_iters') == n_iters
                and cached.get('num_samples') == num_samples
            )
            if same_cfg:
                cached_trajs = cached.get('trajectories', {}) or {}
                for key in trajectories:
                    if isinstance(cached_trajs.get(key), list):
                        trajectories[key] = cached_trajs[key]
                done = ', '.join(f'{k}={len(v)}' for k, v in trajectories.items())
                log(f'  resumed from {cache_path} ({done})')
            else:
                log(f'  {cache_path} exists but run config differs; starting fresh')
        except (json.JSONDecodeError, OSError) as e:
            log(f'  could not read {cache_path} ({e}); starting fresh')

    def _build_data() -> dict:
        return {
            'R': R,
            'L': L,
            'max_dim': max_dim,
            'shots': shots,
            'n_iters': n_iters,
            'num_samples': num_samples,
            'seeds': [args.seed + j for j in range(n_seeds)],
            'casci_energy': casci_e,
            'trajectories': trajectories,
            'skqd_settings': {
                'krylov_dim': int(args.gqe_skqd_krylov_dim),
                'dt': float(args.gqe_skqd_dt),
                'trotter_steps': int(args.gqe_skqd_trotter_steps),
                'shots_per_krylov_state': int(args.gqe_skqd_shots),
            } if args.gqe_skqd else None,
        }

    def _checkpoint() -> None:
        # Atomic write: never leave the cache half-written on crash/kill.
        tmp = cache_path.with_suffix(cache_path.suffix + '.tmp')
        with tmp.open('w') as f:
            json.dump(_build_data(), f, indent=2)
        tmp.replace(cache_path)

    for i in range(n_seeds):
        seed = args.seed + i

        if len(trajectories['gqe']) > i:
            log(f'\n  --- GQE seed {seed} ({i+1}/{n_seeds}): cached, skipping ---')
        else:
            log(f'\n  --- GQE seed {seed} ({i+1}/{n_seeds}) ---')
            cfg = _build_fig3_config(
                seed=seed, R=R, L=L, max_dim=max_dim, shots=shots,
                n_iters=n_iters, num_samples=num_samples,
                step_per_epoch=step_per_epoch,
                output_dir=per_run_out / f'gqe-seed{seed}',
            )
            t0 = time.time()
            traj = _run_gqe_seed(seed, cfg, trainer_kwargs, tag='gqe')
            log(f'  GQE seed {seed} done in {time.time() - t0:.1f}s '
                f'({len(traj)} iterations)')
            trajectories['gqe'].append(traj)
            _checkpoint()

        if len(trajectories['random']) > i:
            log(f'\n  --- Random seed {seed} ({i+1}/{n_seeds}): cached, skipping ---')
        else:
            log(f'\n  --- Random seed {seed} ({i+1}/{n_seeds}) ---')
            cfg = _build_fig3_config(
                seed=seed, R=R, L=L, max_dim=max_dim, shots=shots,
                n_iters=n_iters, num_samples=num_samples,
                step_per_epoch=step_per_epoch,
                output_dir=per_run_out / f'random-seed{seed}',
            )
            t0 = time.time()
            traj = _run_random_seed(seed, cfg)
            log(f'  Random seed {seed} done in {time.time() - t0:.1f}s')
            trajectories['random'].append(traj)
            _checkpoint()

        if args.gqe_skqd:
            if len(trajectories['gqe_skqd']) > i:
                log(f'\n  --- GQE-SKQD seed {seed} ({i+1}/{n_seeds}): cached, skipping ---')
            else:
                log(f'\n  --- GQE-SKQD seed {seed} ({i+1}/{n_seeds}) ---')
                cfg = _build_fig3_config(
                    seed=seed, R=R, L=L, max_dim=max_dim, shots=shots,
                    n_iters=n_iters, num_samples=num_samples,
                    step_per_epoch=step_per_epoch,
                    output_dir=per_run_out / f'gqe-skqd-seed{seed}',
                )
                t0 = time.time()
                traj = _run_gqe_seed(seed, cfg, trainer_kwargs,
                                     use_skqd=True, tag='gqe-skqd')
                log(f'  GQE-SKQD seed {seed} done in {time.time() - t0:.1f}s '
                    f'({len(traj)} iterations)')
                trajectories['gqe_skqd'].append(traj)
                _checkpoint()

    data = _build_data()
    # Ensure the cache file reflects the final state even when nothing ran.
    _checkpoint()
    log(f'  wrote {cache_path}')
    return data


# --- Plotting ---------------------------------------------------------------

CHEMICAL_PRECISION = 1.5936e-3  # Ha (1 kcal/mol)

# Colour/marker choices are intended to visually mirror the paper's Figures 3-6
_STYLE = {
    'exact':     dict(color='#377eb8', marker='s', ls='-',  label='Exact ground state'),
    'lucj':      dict(color='#ff7f00', marker='o', ls='-',  label='LUCJ'),
    'sqdrift':   dict(color='#e41a1c', marker='^', ls='-',  label='SqDRIFT'),
    'random':    dict(color='#7f7f7f', ls='-',              label='Random circuit (best-so-far)'),
    'gqe':       dict(color='#377eb8', ls='-',              label='GQE-optimized circuit (best-so-far)'),
    'local':     dict(color='#4daf4a', ls='-',              label='Local refined (best-so-far)'),
    'global':    dict(color='#984ea3', ls='-',              label='Global refined'),
    'gqe_skqd':  dict(color='#e6ab02', marker='D', ls='-',  label='GQE-SKQD ansatz'),
}
_PANEL_LABEL = ['(a)', '(b)', '(c)', '(d)', '(e)']


def _finalize_error_axis(ax, ylabel=True):
    # Draw the chemical-precision band using the current y-axis bottom, so we
    # never end up with a band that spans 8 decades of empty log space.
    ax.autoscale_view()
    y0, y1 = ax.get_ylim()
    band_bottom = min(y0, CHEMICAL_PRECISION / 5)
    band_top = CHEMICAL_PRECISION
    # Extend axis a little below the band so it's visible as a strip, not a
    # halfplane filling the whole panel.
    new_y0 = min(y0, band_bottom)
    new_y1 = max(y1, CHEMICAL_PRECISION * 3)
    ax.set_ylim(new_y0, new_y1)
    ax.axhspan(band_bottom, band_top, color='#bcd8ec', alpha=0.55,
               label='Chemical precision', zorder=0)
    if ylabel:
        ax.set_ylabel('Energy error [Ha]')
    ax.grid(True, which='both', ls=':', color='0.6', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def _panel_label(ax, i):
    ax.text(0.02, 0.97, _PANEL_LABEL[i], transform=ax.transAxes,
            ha='left', va='top', fontsize=11, fontweight='bold')


def _plot_pts(ax, pts, xattr, yfn, style):
    if not pts:
        return
    xs = np.array([getattr(p, xattr) for p in pts], dtype=float)
    ys = np.array([yfn(p) for p in pts], dtype=float)
    # Sort so lines are monotone in x (matters for compactness plot).
    order = np.argsort(xs)
    ax.plot(xs[order], ys[order], markersize=6, lw=1.4, mfc='white', mew=1.4,
            **style)


def _stack_trajectory(trajs: list[list[dict]], key: str) -> np.ndarray | None:
    """Turn a list of trajectories (one per seed) into a (n_seeds, n_iters)
    matrix for the given key. Returns None if no trajectories have that key
    populated. Missing entries (None) are ignored per column."""
    if not trajs:
        return None
    n_iters = max(len(t) for t in trajs)
    n_seeds = len(trajs)
    mat = np.full((n_seeds, n_iters), np.nan, dtype=float)
    any_populated = False
    for i, t in enumerate(trajs):
        for j, row in enumerate(t):
            v = row.get(key)
            if v is None:
                continue
            mat[i, j] = float(v)
            any_populated = True
    if not any_populated:
        return None
    return mat


def _plot_mean_std(ax, mat: np.ndarray, style: dict, transform=None):
    """Plot mean ± std across the seed axis (rows). transform is applied to
    values before aggregating (e.g. absolute-error transform)."""
    if mat is None:
        return
    iters = np.arange(mat.shape[1])
    vals = mat if transform is None else transform(mat)
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(vals, axis=0)
        std = np.nanstd(vals, axis=0)
    ax.plot(iters, mean, lw=1.6, **style)
    if mat.shape[0] > 1:
        col = style.get('color', 'C0')
        ax.fill_between(iters, mean - std, mean + std, color=col, alpha=0.18,
                        linewidth=0)


def _import_plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
    })
    return plt


def make_fig3(fig3_data: dict) -> None:
    if not fig3_data:
        log('Fig 3: no data available; skipping plot')
        return
    plt = _import_plt()

    casci_e = fig3_data['casci_energy']
    trajs = fig3_data.get('trajectories', {})
    gqe_trajs = trajs.get('gqe', [])
    rnd_trajs = trajs.get('random', [])
    skqd_trajs = trajs.get('gqe_skqd', [])
    max_dim = int(fig3_data.get('max_dim', 0)) or None
    R = fig3_data.get('R', args.fig3_R)
    L = fig3_data.get('L', args.fig3_L)
    mol_tag = args.molecule.upper()

    fig, (ax_e, ax_n) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    # (a) Energy error vs iteration
    def _err(mat): return np.maximum(np.abs(mat - casci_e), 1e-8)

    _plot_mean_std(ax_e, _stack_trajectory(rnd_trajs, 'energy'),
                   _STYLE['random'], transform=_err)
    _plot_mean_std(ax_e, _stack_trajectory(gqe_trajs, 'energy'),
                   _STYLE['gqe'], transform=_err)
    _plot_mean_std(ax_e, _stack_trajectory(gqe_trajs, 'local_energy'),
                   _STYLE['local'], transform=_err)
    _plot_mean_std(ax_e, _stack_trajectory(gqe_trajs, 'global_energy'),
                   _STYLE['global'], transform=_err)
    if skqd_trajs:
        gqe_skqd_style = {k: v for k, v in _STYLE['gqe_skqd'].items()
                          if k != 'marker'}
        gqe_skqd_style['label'] = 'GQE-SKQD circuit (best-so-far)'
        _plot_mean_std(ax_e, _stack_trajectory(skqd_trajs, 'energy'),
                       gqe_skqd_style, transform=_err)
    ax_e.set_yscale('log')
    _finalize_error_axis(ax_e, ylabel=True)
    ax_e.set_xlabel('Iteration')
    _panel_label(ax_e, 0)
    ax_e.legend(loc='upper right', framealpha=0.9)

    # (b) Number of determinants vs iteration
    _plot_mean_std(ax_n, _stack_trajectory(rnd_trajs, 'ndet'), _STYLE['random'])
    _plot_mean_std(ax_n, _stack_trajectory(gqe_trajs, 'ndet'), _STYLE['gqe'])
    _plot_mean_std(ax_n, _stack_trajectory(gqe_trajs, 'local_ndet'), _STYLE['local'])
    _plot_mean_std(ax_n, _stack_trajectory(gqe_trajs, 'global_ndet'), _STYLE['global'])
    if skqd_trajs:
        gqe_skqd_style = {k: v for k, v in _STYLE['gqe_skqd'].items()
                          if k != 'marker'}
        gqe_skqd_style['label'] = 'GQE-SKQD circuit (best-so-far)'
        _plot_mean_std(ax_n, _stack_trajectory(skqd_trajs, 'ndet'), gqe_skqd_style)
    if max_dim:
        ax_n.axhline(max_dim, color='k', ls='--', lw=0.8,
                     label=fr'$d_\mathrm{{max}}$ = {max_dim}')
    ax_n.set_xlabel('Iteration')
    ax_n.set_ylabel('Number of Slater determinants')
    ax_n.grid(True, ls=':', color='0.6', alpha=0.6)
    ax_n.set_axisbelow(True)
    _panel_label(ax_n, 1)
    ax_n.legend(loc='lower right', framealpha=0.9)

    fig.suptitle(fr'Figure 3 — Optimization history for {mol_tag} ({nelecas}e, '
                 fr'{norbcas}o) at $R_\mathrm{{NN}} = {R}$ Å, $L = {L}$',
                 y=1.0)
    fig.tight_layout()
    out = out_dir / 'figure3_optimization_history.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    log(f'wrote {out}')
    plt.close(fig)


def make_fig4(results: dict) -> None:
    plt = _import_plt()
    Rs = [R for R in geometries if str(R) in results]
    if not Rs:
        log('Fig 4: no results; skipping')
        return
    ncols = len(Rs)
    system_tag = f'({args.molecule.upper()}, {nelecas}e, {norbcas}o)'

    fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 6.4),
                              sharex='col', sharey='row', squeeze=False)
    for i, R in enumerate(Rs):
        r = results[str(R)]
        e_fci = r['casci_energy']
        ax_e, ax_n = axes[0, i], axes[1, i]

        curves = [
            ('lucj3_sampling',   'lucj',    ' (3 layers)'),
            ('sqdrift_sampling', 'sqdrift', ' (N=100)'),
            ('exact_sampling',   'exact',   ''),
        ]
        if 'gqe_skqd_sampling' in r:
            curves.append(('gqe_skqd_sampling', 'gqe_skqd',
                           f' (L={args.fig456_gqe_L})'))
        for key, style_key, label_extra in curves:
            pts = points_from_json(r.get(key, []))
            s = dict(_STYLE[style_key])
            s['label'] = s['label'] + label_extra
            _plot_pts(ax_e, pts, 'x', lambda p, e=e_fci: max(abs(p.energy - e), 1e-8), s)
            _plot_pts(ax_n, pts, 'x', lambda p: max(p.subspace_dim, 1), s)

        ax_e.set_xscale('log'); ax_e.set_yscale('log')
        ax_n.set_xscale('log'); ax_n.set_yscale('log')
        _finalize_error_axis(ax_e, ylabel=(i == 0))
        ax_n.grid(True, which='both', ls=':', color='0.6', alpha=0.6)
        ax_n.set_axisbelow(True)
        if i == 0:
            ax_n.set_ylabel('Number of determinants')
        ax_n.set_xlabel(r'Total number of shots $N_\mathrm{shot}$')
        ax_e.set_title(fr'$R_\mathrm{{NN}} = {R}$ Å')
        _panel_label(ax_e, i)
        if i == ncols - 1:
            ax_e.legend(loc='lower left', framealpha=0.9)

    fig.suptitle(f'Figure 4 — Sampling efficiency {system_tag}', y=0.995)
    fig.tight_layout()
    out = out_dir / 'figure4_sampling_efficiency.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    log(f'wrote {out}')
    plt.close(fig)


def make_fig5(results: dict) -> None:
    plt = _import_plt()
    Rs = [R for R in geometries if str(R) in results]
    if not Rs:
        log('Fig 5: no results; skipping')
        return
    ncols = len(Rs)
    system_tag = f'({args.molecule.upper()}, {nelecas}e, {norbcas}o)'

    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.8),
                              sharey=True, squeeze=False)
    axes = axes[0]
    for i, R in enumerate(Rs):
        r = results[str(R)]
        e_fci = r['casci_energy']
        ax = axes[i]
        for key, style_key in [
            ('lucj_gate_efficiency',    'lucj'),
            ('sqdrift_gate_efficiency', 'sqdrift'),
        ]:
            pts = points_from_json(r.get(key, []))
            _plot_pts(ax, pts, 'x',
                      lambda p, e=e_fci: max(abs(p.energy - e), 1e-8),
                      _STYLE[style_key])
        if 'gqe_skqd_gate' in r:
            pts = points_from_json(r.get('gqe_skqd_gate', []))
            s = dict(_STYLE['gqe_skqd'])
            s['label'] = f"GQE-SKQD ansatz (L={args.fig456_gqe_L})"
            _plot_pts(ax, pts, 'x',
                      lambda p, e=e_fci: max(abs(p.energy - e), 1e-8), s)
        ax.set_xscale('log'); ax.set_yscale('log')
        _finalize_error_axis(ax, ylabel=(i == 0))
        ax.set_xlabel('Number of two-qubit gates')
        ax.set_title(fr'$R_\mathrm{{NN}} = {R}$ Å')
        _panel_label(ax, i)
        if i == ncols - 1:
            ax.legend(loc='lower left', framealpha=0.9)

    fig.suptitle(f'Figure 5 — Two-qubit-gate efficiency {system_tag}', y=1.0)
    fig.tight_layout()
    out = out_dir / 'figure5_gate_efficiency.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    log(f'wrote {out}')
    plt.close(fig)


def make_fig6(results: dict) -> None:
    plt = _import_plt()
    Rs = [R for R in geometries if str(R) in results]
    if not Rs:
        log('Fig 6: no results; skipping')
        return
    ncols = len(Rs)
    system_tag = f'({args.molecule.upper()}, {nelecas}e, {norbcas}o)'

    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.8),
                              sharey=True, squeeze=False)
    axes = axes[0]
    for i, R in enumerate(Rs):
        r = results[str(R)]
        e_fci = r['casci_energy']
        ax = axes[i]
        curves = [
            ('lucj3_compactness',   'lucj',    ' (3 layers)'),
            ('sqdrift_compactness', 'sqdrift', ' (N=100)'),
            ('exact_compactness',   'exact',   ''),
        ]
        if 'gqe_skqd_compactness' in r:
            curves.append(('gqe_skqd_compactness', 'gqe_skqd',
                           f' (L={args.fig456_gqe_L})'))
        for key, style_key, label_extra in curves:
            pts = points_from_json(r.get(key, []))
            s = dict(_STYLE[style_key])
            s['label'] = s['label'] + label_extra
            _plot_pts(ax, pts, 'subspace_dim',
                      lambda p, e=e_fci: max(abs(p.energy - e), 1e-8), s)
        ax.set_xscale('log'); ax.set_yscale('log')
        _finalize_error_axis(ax, ylabel=(i == 0))
        ax.set_xlabel('Number of determinants')
        ax.set_title(fr'$R_\mathrm{{NN}} = {R}$ Å')
        _panel_label(ax, i)
        if i == ncols - 1:
            ax.legend(loc='lower left', framealpha=0.9)

    fig.suptitle(f'Figure 6 — Wavefunction compactness {system_tag}', y=1.0)
    fig.tight_layout()
    out = out_dir / 'figure6_compactness.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    log(f'wrote {out}')
    plt.close(fig)


# --- Main -------------------------------------------------------------------

def main() -> None:
    log(f'Figures selected: {sorted(FIGURES)}')

    if NEED_FIG3:
        fig3_data = run_fig3()
        make_fig3(fig3_data)

    if NEED_R_SWEEP:
        results: dict = {}
        for R in geometries:
            log(f'\n=== {args.molecule} R={R} A ===')
            data = ensure_geometry_data(
                R,
                need_sampling=NEED_SAMPLING_CURVES,
                need_gates=NEED_GATE_CURVES,
            )
            if data:
                results[str(R)] = data

        if 4 in FIGURES:
            make_fig4(results)
        if 5 in FIGURES:
            make_fig5(results)
        if 6 in FIGURES:
            make_fig6(results)

    log(f'\nDone. See {out_dir}')


if __name__ == '__main__':
    main()
