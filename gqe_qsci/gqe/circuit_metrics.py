from collections import Counter
from copy import copy
from dataclasses import dataclass, field
from typing import Iterable

import cudaq

from gqe_qsci.gqe.operator_pool import OperatorPool


Gate = tuple[str, tuple[int, ...]]


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class CircuitMetrics:
    """Gate counts and depth for a single circuit (GQE state-prep, or one Krylov vector)."""
    n_qubits: int
    depth: int
    gate_counts: Counter
    two_qubit_depth: int
    gates: list[Gate] = field(default_factory=list)

    @property
    def total_gates(self) -> int:
        return sum(self.gate_counts.values())

    @property
    def cx_count(self) -> int:
        return self.gate_counts.get("cx", 0)

    def summary(self) -> str:
        lines = [
            f"qubits              : {self.n_qubits}",
            f"total gates         : {self.total_gates}",
            f"circuit depth       : {self.depth}",
            f"two-qubit depth     : {self.two_qubit_depth}",
            "gates by type       :",
        ]
        for name in sorted(self.gate_counts):
            lines.append(f"  {name:<6}: {self.gate_counts[name]}")
        return "\n".join(lines)


@dataclass
class SKQDCircuitMetrics:
    """Circuit metrics for the full GQE+SKQD pipeline.

    The SKQD kernel for Krylov index k is:
        HF prep  →  GQE block  →  k * trotter_steps Trotter sweeps of H

    Attributes:
        gqe_metrics     : metrics for k=0 (pure GQE state-prep circuit)
        per_k           : list of CircuitMetrics, one per k in [0, krylov_dim)
        krylov_dim      : number of Krylov vectors
        trotter_steps   : Trotter steps per time increment
        hamiltonian_terms: number of Pauli terms in H
    """
    gqe_metrics: CircuitMetrics
    per_k: list[CircuitMetrics]
    krylov_dim: int
    trotter_steps: int
    hamiltonian_terms: int

    @property
    def n_qubits(self) -> int:
        return self.gqe_metrics.n_qubits

    @property
    def max_depth(self) -> int:
        return self.per_k[-1].depth

    @property
    def max_two_qubit_depth(self) -> int:
        return self.per_k[-1].two_qubit_depth

    @property
    def max_cx_count(self) -> int:
        return self.per_k[-1].cx_count

    def summary(self) -> str:
        lines = [
            "=== GQE state-prep circuit (k=0) ===",
            self.gqe_metrics.summary(),
            "",
            f"=== SKQD full pipeline (krylov_dim={self.krylov_dim}, "
            f"trotter_steps={self.trotter_steps}, H terms={self.hamiltonian_terms}) ===",
            f"{'k':>4}  {'depth':>8}  {'cx_depth':>10}  {'cx_count':>10}  {'total_gates':>12}",
            "-" * 52,
        ]
        for k, m in enumerate(self.per_k):
            lines.append(
                f"{k:>4}  {m.depth:>8}  {m.two_qubit_depth:>10}  "
                f"{m.cx_count:>10}  {m.total_gates:>12}"
            )
        lines += [
            "-" * 52,
            f"deepest circuit (k={self.krylov_dim - 1}): "
            f"depth={self.max_depth}, cx_depth={self.max_two_qubit_depth}, "
            f"cx_count={self.max_cx_count}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gate synthesis
# ---------------------------------------------------------------------------

def pauli_evolution_gates(pauli: str) -> list[Gate]:
    """Synthesize exp(-iθP) into primitive gates.

    Matches gqe.utils.get_pauli_evolution_gate_count:
      basis change (H / Sdg+H) → CX ladder down → Rz → CX ladder up → undo basis change.
    """
    active = [(q, p) for q, p in enumerate(pauli) if p in ("X", "Y", "Z")]
    if not active:
        return []

    gates: list[Gate] = []
    for q, p in active:
        if p == "X":
            gates.append(("h", (q,)))
        elif p == "Y":
            gates.append(("sdg", (q,)))
            gates.append(("h", (q,)))

    qubits = [q for q, _ in active]
    for i in range(len(qubits) - 1):
        gates.append(("cx", (qubits[i], qubits[i + 1])))

    gates.append(("rz", (qubits[-1],)))

    for i in range(len(qubits) - 2, -1, -1):
        gates.append(("cx", (qubits[i], qubits[i + 1])))

    for q, p in active:
        if p == "X":
            gates.append(("h", (q,)))
        elif p == "Y":
            gates.append(("h", (q,)))
            gates.append(("s", (q,)))

    return gates


def _spin_op_to_pauli_words(op: cudaq.SpinOperator, n_qubits: int) -> list[str]:
    return [term.get_pauli_word(n_qubits) for term in op]


# ---------------------------------------------------------------------------
# Gate list construction
# ---------------------------------------------------------------------------

def build_gate_list(
    operator_pool: OperatorPool,
    seq: Iterable[int],
    n_electrons: int | None = None,
) -> list[Gate]:
    """Gate list for the GQE sampler kernel (HF prep + GQE block).

    Mirrors gqe.sampler.Sampler.kernel.
    """
    n_qubits = operator_pool.n_qubits
    if n_electrons is None:
        n_electrons = operator_pool.n_electrons

    gates: list[Gate] = [("x", (q,)) for q in range(n_electrons)]
    for i in seq:
        operator = operator_pool[i]
        for term in operator:
            gates.extend(pauli_evolution_gates(term.get_pauli_word(n_qubits)))
    return gates


def build_trotter_step_gates(h_pauli_words: list[str]) -> list[Gate]:
    """Gate list for one Trotter sweep of the Hamiltonian."""
    gates: list[Gate] = []
    for word in h_pauli_words:
        gates.extend(pauli_evolution_gates(word))
    return gates


# ---------------------------------------------------------------------------
# Depth computation (ASAP scheduling)
# ---------------------------------------------------------------------------

class _Frontier:
    """Per-qubit time frontier for incremental ASAP scheduling."""

    def __init__(self, n_qubits: int):
        self._f: list[int] = [0] * n_qubits
        self._f2: list[int] = [0] * n_qubits

    def add_gates(self, gates: list[Gate]) -> None:
        f, f2 = self._f, self._f2
        for _, qubits in gates:
            t = max(f[q] for q in qubits) + 1
            for q in qubits:
                f[q] = t
            if len(qubits) >= 2:
                t2 = max(f2[q] for q in qubits) + 1
                for q in qubits:
                    f2[q] = t2

    def depth(self) -> int:
        return max(self._f) if self._f else 0

    def two_qubit_depth(self) -> int:
        return max(self._f2) if self._f2 else 0

    def snapshot(self) -> tuple[list[int], list[int]]:
        return self._f.copy(), self._f2.copy()

    def restore(self, snap: tuple[list[int], list[int]]) -> None:
        self._f, self._f2 = snap[0].copy(), snap[1].copy()


def compute_depth(gates: list[Gate], n_qubits: int) -> tuple[int, int]:
    """ASAP schedule; returns (overall depth, two-qubit-only depth)."""
    frontier = _Frontier(n_qubits)
    frontier.add_gates(gates)
    return frontier.depth(), frontier.two_qubit_depth()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_circuit_metrics(
    operator_pool: OperatorPool,
    seq: Iterable[int],
    n_electrons: int | None = None,
    keep_gates: bool = False,
) -> CircuitMetrics:
    """Metrics for the GQE state-prep circuit (GQE+QSCI pipeline)."""
    n_qubits = operator_pool.n_qubits
    gates = build_gate_list(operator_pool, seq, n_electrons=n_electrons)
    counts: Counter = Counter(name for name, _ in gates)
    depth, two_q_depth = compute_depth(gates, n_qubits)
    return CircuitMetrics(
        n_qubits=n_qubits,
        depth=depth,
        gate_counts=counts,
        two_qubit_depth=two_q_depth,
        gates=gates if keep_gates else [],
    )


def compute_skqd_circuit_metrics(
    operator_pool: OperatorPool,
    seq: Iterable[int],
    h_spin_op: cudaq.SpinOperator,
    krylov_dim: int,
    trotter_steps: int,
    n_electrons: int | None = None,
) -> SKQDCircuitMetrics:
    """Metrics for the full GQE+SKQD pipeline.

    Reconstructs the skqd_kernel circuit for every Krylov index k in
    [0, krylov_dim) and reports gate counts and depth for each.

    Args:
        operator_pool : the GQE operator pool
        seq           : sequence of pool indices selected by the GQE model
        h_spin_op     : molecular Hamiltonian as a cudaq.SpinOperator
                        (from skqd.hamiltonian.hamiltonian_to_cudaq_spin)
        krylov_dim    : number of Krylov vectors (matches SKQDPipeline.krylov_dim)
        trotter_steps : Trotter steps per time step (matches SKQDPipeline.trotter_steps)
        n_electrons   : override for n_electrons (defaults to pool.n_electrons)
    """
    n_qubits = operator_pool.n_qubits
    seq = list(seq)

    # GQE state-prep gates (shared across all k)
    gqe_gates = build_gate_list(operator_pool, seq, n_electrons=n_electrons)
    gqe_counts: Counter = Counter(name for name, _ in gqe_gates)

    frontier = _Frontier(n_qubits)
    frontier.add_gates(gqe_gates)
    gqe_depth = frontier.depth()
    gqe_two_q = frontier.two_qubit_depth()
    gqe_metrics = CircuitMetrics(
        n_qubits=n_qubits,
        depth=gqe_depth,
        gate_counts=gqe_counts,
        two_qubit_depth=gqe_two_q,
    )

    # One Trotter sweep of H
    h_words = _spin_op_to_pauli_words(h_spin_op, n_qubits)
    one_sweep_gates = build_trotter_step_gates(h_words)
    one_sweep_counts: Counter = Counter(name for name, _ in one_sweep_gates)

    # Incrementally extend the frontier, recording metrics at each k boundary
    per_k: list[CircuitMetrics] = []
    running_counts = Counter(gqe_counts)
    sweep_num = 0  # total Trotter sweeps applied so far

    for k in range(krylov_dim):
        # k=0 needs 0 sweeps, k=1 needs trotter_steps sweeps, etc.
        target_sweeps = k * trotter_steps
        while sweep_num < target_sweeps:
            frontier.add_gates(one_sweep_gates)
            running_counts += one_sweep_counts
            sweep_num += 1

        per_k.append(CircuitMetrics(
            n_qubits=n_qubits,
            depth=frontier.depth(),
            gate_counts=Counter(running_counts),
            two_qubit_depth=frontier.two_qubit_depth(),
        ))

    return SKQDCircuitMetrics(
        gqe_metrics=gqe_metrics,
        per_k=per_k,
        krylov_dim=krylov_dim,
        trotter_steps=trotter_steps,
        hamiltonian_terms=len(h_words),
    )
