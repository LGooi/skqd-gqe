import numpy as np
import torch
import pyci
from typing import Any

from gqe_qsci.molecule import PySCFMolecule
from gqe_qsci.gqe.operator_pool import OperatorPool
from gqe_qsci.skqd.schema import SKQDResult, SKQDSampleResult
from gqe_qsci.skqd.sampler import SKQDSampler
from gqe_qsci.skqd.hamiltonian import hamiltonian_to_cudaq_spin
from gqe_qsci.qsci.subspace import DeterminantSubspace
from gqe_qsci.qsci.refine.pipeline import RefinePipeline
from gqe_qsci.qsci.statevector import as_scivector, SCIVector


class SKQDPipeline:
    def __init__(
        self,
        molecule: PySCFMolecule,
        operator_pool: OperatorPool,
        sampler: SKQDSampler,
        krylov_dim: int = 12,
        dt: float = 0.1,
        trotter_steps: int = 4,
        max_dim: int = 100_000,
        max_cycle: int = 200,
        eigsh_kwargs: dict[str, Any] | None = None,
    ):
        self.mol = molecule
        self.hamiltonian = molecule.cas_hamiltonian
        self.norb = molecule.norb
        self.spin = molecule.spin
        self.nelec = molecule.nelec
        self.operator_pool = operator_pool
        self.sampler = sampler
        self.krylov_dim = krylov_dim
        self.dt = dt
        self.trotter_steps = trotter_steps
        self.max_dim = max_dim
        self.max_cycle = max_cycle
        self.eigsh_kwargs = eigsh_kwargs or {}

        self.subspace_refiner = RefinePipeline(
            hamiltonian=self.hamiltonian, nelec=self.nelec, norb=self.norb
        )
        self.global_refined_scistates = None

        self._pyci_h2 = np.asarray(
            self.hamiltonian.h2.transpose(0, 2, 1, 3), order="C"
        )
        self._pyci_ham = pyci.hamiltonian(
            self.hamiltonian.e_core, self.hamiltonian.h1, self._pyci_h2
        )

        h_spin_op = hamiltonian_to_cudaq_spin(molecule)
        self._h_coeffs = [c.real for c in self.sampler.term_coefficients(h_spin_op)]
        self._h_words = self.sampler.term_words(h_spin_op)

    def diagonalize(self, subspace: DeterminantSubspace) -> tuple[float, SCIVector]:
        wfn = pyci.fullci_wfn(self._pyci_ham.nbasis, *self.nelec)
        for det in subspace.determinants:
            wfn.add_det(det)
        op = pyci.sparse_op(self._pyci_ham, wfn)
        energy, coeffs = op.solve(maxiter=self.max_cycle)
        return float(energy[0]), as_scivector(coeffs[0], subspace.determinants)

    def process(self, states: dict[str, torch.Tensor]) -> SKQDResult:
        sampled_per_batch = self.sampler.run_krylov(
            states,
            self.dt,
            self.trotter_steps,
            self.krylov_dim,
            self._h_coeffs,
            self._h_words,
        )

        samples: list[SKQDSampleResult] = []
        sci_states: list[SCIVector] = []

        for seq, krylov_counts in zip(states["idx"].tolist(), sampled_per_batch):
            circuit_gate_numbers = self.operator_pool.get_gate_count(seq)

            per_k_subspaces = [
                DeterminantSubspace.from_cudaq_sample_result(c) for c in krylov_counts
            ]
            unioned = per_k_subspaces[0].union(per_k_subspaces[1:])
            num_unioned = unioned.ndet

            valid_subspace = unioned.post_select_by_nelec(self.nelec)
            num_valid = valid_subspace.ndet

            truncated = valid_subspace.enlarge(max_dim=self.max_dim, method="symmetry_completion")

            energy, ci = self.diagonalize(truncated)
            sci_states.append(ci)

            samples.append(
                SKQDSampleResult(
                    seq=tuple(seq),
                    energy=energy,
                    num_sampled_basis=num_unioned,
                    num_symmetry_preserving_basis=num_valid,
                    subspace_dim=truncated.ndet,
                    cx_count=circuit_gate_numbers.get("cx", 0),
                    total_gates=circuit_gate_numbers.get("total", 0),
                    krylov_dim=self.krylov_dim,
                )
            )

        local_refined_subspace = self.subspace_refiner.process(
            sci_states, max_dim=self.max_dim
        )
        local_refined_energy, local_refined_ci = self.diagonalize(local_refined_subspace)
        local_refined_result = SKQDSampleResult(
            seq=None,
            energy=local_refined_energy,
            num_sampled_basis=None,
            num_symmetry_preserving_basis=None,
            subspace_dim=local_refined_subspace.ndet,
            cx_count=None,
            total_gates=None,
            krylov_dim=self.krylov_dim,
        )

        if self.global_refined_scistates is None:
            self.global_refined_scistates = local_refined_ci
            global_refined_result: SKQDSampleResult | None = None
        else:
            target_sci_states = [self.global_refined_scistates, local_refined_ci]
            global_refined_subspace = self.subspace_refiner.process(
                target_sci_states, max_dim=self.max_dim
            )
            global_refined_energy, global_refined_ci = self.diagonalize(
                global_refined_subspace
            )
            self.global_refined_scistates = global_refined_ci
            global_refined_result = SKQDSampleResult(
                seq=None,
                energy=global_refined_energy,
                num_sampled_basis=None,
                num_symmetry_preserving_basis=None,
                subspace_dim=global_refined_subspace.ndet,
                cx_count=None,
                total_gates=None,
                krylov_dim=self.krylov_dim,
            )

        return SKQDResult(
            samples=tuple(samples),
            local_refined=local_refined_result,
            global_refined=global_refined_result,
        )
