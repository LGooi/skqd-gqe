
# Pure LUCJ ground-state energy for a small molecule.
#
# Implements the Local Unitary Cluster Jastrow (LUCJ) ansatz
# [Motta et al. 2023, Kaliakin et al. 2025] as a QSCI input state:
#
#   |Psi> = U_LUCJ |HF>
#   U_LUCJ = prod_k U_k exp(i J_k) U_k^dagger
#
# Each U_k is a number-preserving orbital rotation and each J_k is a
# diagonal Coulomb operator J = 1/2 sum_{pq,st} J^(st)_{pq} n_{p,s} n_{q,t}.
# For LOCAL UCJ we restrict the interaction pattern of J to
#   alpha-alpha : (p, p+1)   (nearest-neighbor spatial orbitals)
#   alpha-beta  : (p, p)     (on-site cross-spin)
# The tensors {J_k, U_k} are initialised from a double factorisation of the
# CCSD (t1, t2) amplitudes on the CAS space, matching the LUCJ baseline
# used in the paper.
#
# For each of `n_shots` measurements we prepare |Psi> and draw a
# computational-basis bitstring. The unique bitstrings define a
# determinant subspace, in which H is diagonalized classically with PyCI.
# The subspace can be grown by increasing `n_layers` (n_reps in ffsim) or
# by increasing `shots`.
#
# Run with:
#   python3 pure_lucj.py
#
# Vary the number of ansatz layers (paper sweeps 1..5):
#   python3 pure_lucj.py --layers 3 --shots 100000

import argparse
from collections import Counter
from types import SimpleNamespace

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--shots', type=int, default=100_000,
                    help='Total shots S drawn from the LUCJ state.')
parser.add_argument('--layers', type=int, default=3,
                    help='LUCJ ansatz repetitions (paper sweeps 1..5).')
parser.add_argument('--max-dim', type=int, default=100_000,
                    help='Max determinants passed to classical diagonalization.')
parser.add_argument('--seed', type=int, default=43,
                    help='RNG seed for bitstring sampling.')
parser.add_argument('--molecule', type=str, default='h2',
                    choices=['h2', 'n2', 'n2-eq', 'n2-stretched'],
                    help='Molecule preset. n2/n2-eq use R=1.1A, '
                         'n2-stretched uses R=2.5A, both in (10e,8o).')
parser.add_argument('--bond-length', type=float, default=None,
                    help='Override the molecule bond length (Angstrom).')
parser.add_argument('--optimize-factorization', action='store_true',
                    help='Post-optimize the double factorization of t2 with '
                         'scipy.minimize (matches ffsim defaults).')
args = parser.parse_args()

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ffsim
import ffsim.qiskit as fqk
from qiskit import QuantumCircuit, transpile

import pyci

from gqe_qsci.molecule import PySCFMolecule
from gqe_qsci.qsci.subspace import DeterminantSubspace
from gqe_qsci.qsci.determinant import Determinant
from gqe_qsci.gqe.circuit_metrics import CircuitMetrics, _Frontier

np.random.seed(args.seed)
rng = np.random.default_rng(args.seed)


# --- Molecule ---------------------------------------------------------------

_MOL_PRESETS = {
    'h2':            (['H', 'H'], 0.7474, 2, 2),
    'n2':            (['N', 'N'], 1.1,   10, 8),
    'n2-eq':         (['N', 'N'], 1.1,   10, 8),
    'n2-stretched':  (['N', 'N'], 2.5,   10, 8),
}
atoms, default_bond, nelecas, norbcas = _MOL_PRESETS[args.molecule]
bond = args.bond_length if args.bond_length is not None else default_bond

molecule = PySCFMolecule(
    geometry=SimpleNamespace(type='linear_chain', atoms=atoms, bond_length=bond),
    basis='sto-3g',
    nelecas=nelecas,
    norbcas=norbcas,
    spin=0,
    charge=0,
)
molecule.compute_ccsd()

norb = molecule.norb
nelec = molecule.nelec
n_qubits = 2 * norb


# --- LUCJ ansatz ------------------------------------------------------------

# Local interaction pattern: alpha-alpha nearest-neighbor + on-site alpha-beta.
alpha_alpha_pairs = [(p, p + 1) for p in range(norb - 1)]
alpha_beta_pairs = [(p, p) for p in range(norb)]

t1 = molecule.ccsd_amplitude['t1']
t2 = molecule.ccsd_amplitude['t2']

lucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
    t2=t2,
    t1=t1,
    n_reps=args.layers,
    interaction_pairs=(alpha_alpha_pairs, alpha_beta_pairs),
    optimize=args.optimize_factorization,
)


# --- State-vector sampling --------------------------------------------------

def sample_lucj_state(shots: int) -> DeterminantSubspace:
    """Prepare U_LUCJ |HF> and draw `shots` computational-basis bitstrings.

    The state vector is built exactly on the (norb, nelec) fixed-symmetry
    subspace using ffsim; sampling then reduces to a multinomial draw over
    that subspace with numpy.
    """
    vec = ffsim.hartree_fock_state(norb, nelec)
    vec = ffsim.apply_unitary(vec, lucj_op, norb, nelec)
    probs = np.abs(vec) ** 2
    probs /= probs.sum()  # guard against tiny numerical drift

    addresses = rng.choice(len(probs), size=shots, p=probs)
    strings = ffsim.addresses_to_strings(
        addresses, norb, nelec,
        bitstring_type=ffsim.BitstringType.STRING,
    )

    # ffsim bitstring layout (MSB left): [ beta_{norb-1} ... beta_0 |
    #                                      alpha_{norb-1} ... alpha_0 ]
    counter = Counter(strings)
    dets: list[Determinant] = []
    seen: set[tuple[int, int]] = set()
    for bits, _ in counter.most_common():
        alpha = int(bits[norb:], 2)
        beta = int(bits[:norb], 2)
        key = (alpha, beta)
        if key in seen:
            continue
        seen.add(key)
        dets.append(Determinant([alpha, beta]))
    return DeterminantSubspace(dets)


# --- Classical diagonalization ---------------------------------------------

pyci_h2_tensor = np.asarray(
    molecule.cas_hamiltonian.h2.transpose(0, 2, 1, 3), order='C'
)
pyci_ham = pyci.hamiltonian(
    molecule.cas_hamiltonian.e_core,
    molecule.cas_hamiltonian.h1,
    pyci_h2_tensor,
)


def diagonalize(subspace: DeterminantSubspace) -> float:
    wfn = pyci.fullci_wfn(pyci_ham.nbasis, *molecule.nelec)
    for det in subspace.determinants:
        wfn.add_det(det)
    op = pyci.sparse_op(pyci_ham, wfn)
    energy, _ = op.solve(maxiter=200)
    return float(energy[0])


# --- Circuit metrics --------------------------------------------------------

def _lucj_gate_list() -> list[tuple[str, tuple[int, ...]]]:
    """Transpile LUCJ (HF + UCJ) to a Clifford + Rz + Rx + Ry basis and
    flatten to (name, qubit_tuple) pairs consumed by `_Frontier`.
    """
    qc = QuantumCircuit(n_qubits)
    qc.append(fqk.PrepareHartreeFockJW(norb, nelec), range(n_qubits))
    qc.append(fqk.UCJOpSpinBalancedJW(lucj_op), range(n_qubits))
    qct = transpile(
        qc,
        basis_gates=['h', 's', 'sdg', 'x', 'y', 'z', 'rz', 'ry', 'rx', 'cx'],
        optimization_level=0,
    )
    gates: list[tuple[str, tuple[int, ...]]] = []
    for inst in qct.data:
        name = inst.operation.name
        qubits = tuple(qct.find_bit(q).index for q in inst.qubits)
        gates.append((name, qubits))
    return gates


def measure_circuit_depth() -> CircuitMetrics:
    """Gate counts and ASAP depth for the LUCJ state-prep circuit.

    Uses the same _Frontier ASAP scheduler as SqDRIFT/GQE circuit_metrics
    so the numbers here are directly comparable to `pure_skqd.py` and
    `pure_gqe.py`.
    """
    gates = _lucj_gate_list()
    counts: Counter = Counter(name for name, _ in gates)
    frontier = _Frontier(n_qubits)
    frontier.add_gates(gates)
    return CircuitMetrics(
        n_qubits=n_qubits,
        depth=frontier.depth(),
        gate_counts=counts,
        two_qubit_depth=frontier.two_qubit_depth(),
    )


# --- Run --------------------------------------------------------------------

print(f'LUCJ: molecule={args.molecule} (R={bond} A), n_qubits={n_qubits}, '
      f'nelec={nelec}, layers={args.layers}, shots={args.shots}')

subspace = sample_lucj_state(args.shots)
print(f'  sampled: unique dets={subspace.ndet}')

valid = subspace.post_select_by_nelec(molecule.nelec)
truncated = valid.enlarge(max_dim=args.max_dim, method='symmetry_completion')
energy = diagonalize(truncated)

print(f'  subspace_dim={truncated.ndet:>6}  E={energy:.10f}')
print(f'\nGround Energy = {energy}')

metrics = measure_circuit_depth()
print(f'\nCircuit metrics for LUCJ (n_reps={args.layers}):')
print(metrics.summary())
