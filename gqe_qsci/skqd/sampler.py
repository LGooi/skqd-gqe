import torch
import cudaq
from mpi4py import MPI
from tqdm import tqdm

from gqe_qsci.gqe.operator_pool import OperatorPool
from gqe_qsci.gqe.sampler import Sampler
from gqe_qsci.skqd.kernel import skqd_kernel


class SKQDSampler(Sampler):
    def __init__(self, operator_pool: OperatorPool, mpi: bool, numQPUs: int, shots_count: int):
        super().__init__(operator_pool, mpi, numQPUs, shots_count)

    @torch.no_grad()
    def run_krylov(
        self,
        state: dict,
        dt: float,
        trotter_steps: int,
        krylov_dim: int,
        h_coeffs: list[float],
        h_words: list[cudaq.pauli_word],
    ) -> list[list]:
        """Sample S-KQD Krylov vectors |psi_k> = U^k |psi_0>, k=0..krylov_dim-1.

        Returns list[list[cudaq.SampleResult]] with outer dim = batch sample, inner dim = k.
        """
        idx_output = state["idx"]
        pool = self.pool
        dt_per_step = dt / trotter_steps
        h_coeffs_scaled = [2.0 * c * dt_per_step for c in h_coeffs]

        if cudaq.mpi.is_initialized():
            rank = cudaq.mpi.rank()
            num_ranks = cudaq.mpi.num_ranks()
            total = len(idx_output)
            per_rank = total // num_ranks
            remainder = total % num_ranks
            start = rank * per_rank + min(rank, remainder)
            end = start + per_rank + (1 if rank < remainder else 0)
            local_rows = idx_output[start:end]
        else:
            rank = 0
            local_rows = idx_output

        is_main = rank == 0
        n_samples = len(local_rows)

        local_results: list[list] = []
        for i, row in enumerate(tqdm(local_rows, desc="SKQD samples", disable=not is_main)):
            sampled_ops = [pool[j] for j in row]
            gqe_coeffs: list[float] = []
            gqe_words: list[cudaq.pauli_word] = []
            for op in sampled_ops:
                gqe_coeffs += [c.real for c in self.term_coefficients(op)]
                gqe_words += self.term_words(op)

            per_k = []
            for k in tqdm(range(krylov_dim), desc=f"  sample {i+1}/{n_samples} Krylov", leave=False, disable=not is_main):
                handle = cudaq.sample_async(
                    skqd_kernel,
                    self.pool.n_qubits,
                    self.pool.n_electrons,
                    gqe_coeffs,
                    gqe_words,
                    h_coeffs_scaled,
                    h_words,
                    k,
                    trotter_steps,
                    shots_count=self.shots_count,
                    qpu_id=i % self.numQPUs if self.mpi else 0,
                )
                per_k.append(handle)
            local_results.append(per_k)

        total_handles = sum(len(per_k) for per_k in local_results)
        with tqdm(total=total_handles, desc="Resolving SKQD results", disable=not is_main) as pbar:
            for i, per_k in enumerate(local_results):
                resolved = []
                for h in per_k:
                    resolved.append(h.get() if hasattr(h, "get") else h)
                    pbar.update(1)
                local_results[i] = resolved

        if cudaq.mpi.is_initialized():
            gathered = MPI.COMM_WORLD.allgather(local_results)
            return [batch for shard in gathered for batch in shard]
        return local_results
