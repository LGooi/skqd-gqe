
# GQE is an optional component of the CUDA-QX Solvers Library. To install its
# dependencies, run:
# pip install cudaq-solvers[gqe]
#
# Run this script with
# python3 pure_gqe.py
#
# In order to leverage CUDA-Q MQPU and distribute the work across
# multiple QPUs (thereby observing a speed-up), run with:
#
# mpiexec -np N and vary N to see the speedup...
# e.g. PMIX_MCA_gds=hash mpiexec -np 2 python3 pure_gqe.py --mpi

import argparse
from types import SimpleNamespace

import cudaq

parser = argparse.ArgumentParser()
parser.add_argument('--mpi', action='store_true')
args = parser.parse_args()

def _nvidia_target_usable(option: str | None = None) -> bool:
    """`set_target('nvidia', ...)` succeeds even on GPUs the shipped CUDA-Q
    binary wasn't compiled for; the failure only shows up when a kernel
    actually runs ("RuntimeError: architecture mismatch"). Probe with a
    trivial kernel so we can fall back before real work starts.
    """
    try:
        if option is None:
            cudaq.set_target('nvidia')
        else:
            cudaq.set_target('nvidia', option=option)
    except RuntimeError:
        return False

    @cudaq.kernel
    def _probe():
        q = cudaq.qvector(1)
        x(q[0])

    try:
        cudaq.sample(_probe)
        return True
    except RuntimeError:
        return False


if args.mpi:
    if _nvidia_target_usable(option='mqpu'):
        cudaq.mpi.initialize()
    else:
        print(
            'Warning: NVIDIA GPUs or MPI not available, unable to use CUDA-Q MQPU. Skipping...'
        )
        exit(0)
else:
    if not _nvidia_target_usable(option='fp64'):
        cudaq.set_target('qpp-cpu')

import cudaq_solvers as solvers

from lightning.fabric.loggers import CSVLogger
from cudaq_solvers.gqe_algorithm.gqe import get_default_config

from gqe_qsci.molecule import PySCFMolecule
from gqe_qsci.gqe.operator_pool import PauliEvolutionPool
from gqe_qsci.gqe.circuit_metrics import compute_circuit_metrics
from gqe_qsci.skqd.hamiltonian import hamiltonian_to_cudaq_spin

# Set deterministic seed and environment variables for deterministic behavior
# Disable this section for non-deterministic behavior
import os, torch

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.manual_seed(3047)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create the molecular hamiltonian using PySCFMolecule directly,
# avoiding solvers.create_molecule which requires a PySCF server process.
pool_molecule = PySCFMolecule(
    geometry=SimpleNamespace(type='linear_chain', atoms=['H', 'H'], bond_length=0.7474),
    basis='sto-3g',
    nelecas=2,
    norbcas=2,
    spin=0,
    charge=0,
)

spin_ham = hamiltonian_to_cudaq_spin(pool_molecule)
n_qubits = pool_molecule.norb * 2
n_electrons = sum(pool_molecule.nelec)

params = [
    0.003125, -0.003125, 0.00625, -0.00625, 0.0125, -0.0125, 0.025, -0.025,
    0.05, -0.05, 0.1, -0.1
]

operator_pool = PauliEvolutionPool(
    pool_molecule,
    params=params,
    threshold=1e-6,
    remove_z_ladder=True,
    only_use_first_pauli=True,
)
op_pool = list(operator_pool)


def term_coefficients(op: cudaq.SpinOperator) -> list[complex]:
    return [term.evaluate_coefficient() for term in op]


def term_words(op: cudaq.SpinOperator) -> list[cudaq.pauli_word]:
    return [term.get_pauli_word(n_qubits) for term in op]


# Kernel that applies the selected operators
@cudaq.kernel
def kernel(n_qubits: int, n_electrons: int, coeffs: list[float],
           words: list[cudaq.pauli_word]):
    q = cudaq.qvector(n_qubits)

    for i in range(n_electrons):
        x(q[i])

    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])


def cost(sampled_ops: list[cudaq.SpinOperator], **kwargs):

    full_coeffs = []
    full_words = []

    for op in sampled_ops:
        full_coeffs += [c.real for c in term_coefficients(op)]
        full_words += term_words(op)

    if args.mpi:
        handle = cudaq.observe_async(kernel,
                                     spin_ham,
                                     n_qubits,
                                     n_electrons,
                                     full_coeffs,
                                     full_words,
                                     qpu_id=kwargs['qpu_id'])
        return handle, lambda res: res.get().expectation()
    else:
        return cudaq.observe(kernel, spin_ham, n_qubits, n_electrons,
                             full_coeffs, full_words).expectation()


def measure_circuit_depth(seq: list[int]):
    """Measure gate counts and depth of the GQE state-prep circuit.

    Mirrors gqe_qsci.gqe.circuit_metrics.compute_circuit_metrics: it
    synthesises exp(-iθP) into primitive gates (H / Sdg+H basis change,
    CX ladder, Rz, uncompute) and computes ASAP depth.

    Args:
        seq: sequence of operator-pool indices, e.g. `best_ops` returned by
             `solvers.gqe`.

    Returns:
        CircuitMetrics with n_qubits, depth, two_qubit_depth, gate_counts.
    """
    return compute_circuit_metrics(
        operator_pool,
        seq,
        n_electrons=n_electrons,
    )


# Configure GQE
cfg = get_default_config()
cfg.use_fabric_logging = False
logger = CSVLogger("gqe_h2_logs/gqe.csv")
cfg.fabric_logger = logger
cfg.save_trajectory = False
cfg.verbose = True

# Run GQE
minE, best_ops = solvers.gqe(cost, op_pool, max_iters=25, ngates=10, config=cfg)

# Only print results from rank 0 when using MPI
if not args.mpi or cudaq.mpi.rank() == 0:
    print(f'Ground Energy = {minE}')
    print('Ansatz Ops')
    for idx in best_ops:
        # Get the first (and only) term since these are simple operators
        term = next(iter(op_pool[idx]))
        print(term.evaluate_coefficient().real, term.get_pauli_word(n_qubits))

    metrics = measure_circuit_depth(best_ops)
    print('\nCircuit metrics for the selected ansatz:')
    print(metrics.summary())

if args.mpi:
    cudaq.mpi.finalize()
